import json
import math
import re

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
import os

# 식당 정보 url 을 위해서
# url 에 한글 식당 이름이 들어가면 에러가 날 수 있어서 인코딩을 위한 모듈을 가져옴
from urllib.parse import quote

load_dotenv()

# =========================
# 설정
# =========================
st.set_page_config(page_title="사용자 조건 기반 맛집 추천", page_icon="🍽️", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY가 환경변수(.env)에서 로드되지 않았습니다. .env 또는 환경변수를 확인하세요.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ 이미 만들어진 리뷰/감성 임베딩 Chroma DB
DB_PATH = r"./vector_db"
COLLECTION_NAME = "hongdae_restaurants"

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY
)

db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=DB_PATH
)

# 검색 파라미터
CHROMA_K = 80     # Chroma에서 가져오는 문서 수(리뷰 문서 중복 대비 넉넉히)
FINAL_K = 5       # 최종 후보(= next용)

# ✅ 거리(도보시간) 가중치(패널티)
LAMBDA_DISTANCE = 0.15  # 0~1 권장 (0이면 거리 영향 없음)

# =========================
# CSV 로드 (메타데이터 전용)
# =========================
@st.cache_data
def load_meta_csv():
    df = pd.read_csv("/Users/ijunseong/Downloads/식당DB_통합_도보추가_최최종수정.csv")

    must_text = ["사업장명", "업태구분명", "대표메뉴_메뉴", "대표메뉴_가격", "지번주소"]
    for c in must_text:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    must_num = ["도보거리_km", "도보시간_분"]
    for c in must_num:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["사업장명"] = df["사업장명"].astype(str).str.strip()
    return df

df_meta = load_meta_csv()

@st.cache_data
def build_name_index(df):
    temp = df.copy()
    temp = temp[temp["사업장명"] != ""]
    temp = temp.drop_duplicates(subset=["사업장명"], keep="first")
    return temp.set_index("사업장명")

meta_index = build_name_index(df_meta)

# =========================
# 유틸
# =========================
def clean_llm_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("요약:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()

def safe_num(x):
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None

# ✅ 메뉴/가격 문자열을 "메뉴 : 가격" 형태로 깔끔하게 만들기
def _split_items(s: str):
    """'A | B | C' 또는 'A,B,C' 같은 문자열을 아이템 리스트로 분리"""
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []

    # | 우선, 없으면 쉼표로 분리
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
    else:
        parts = [p.strip() for p in s.split(",")]

    return [p for p in parts if p]

def format_menu_price_lines(menu: str, price: str, max_items: int = 8):
    menus = _split_items(menu)
    prices = _split_items(price)

    lines = []
    n = min(len(menus), len(prices))

    if n > 0:
        for i in range(n):
            lines.append(f"- {menus[i]} : {prices[i]}")
    else:
        # 짝이 안 맞거나 한쪽만 있으면 있는 것만 표시
        if menus and not prices:
            for m in menus[:max_items]:
                lines.append(f"- {m}")
        elif prices and not menus:
            for p in prices[:max_items]:
                lines.append(f"- {p}")
        elif menu or price:
            lines.append(f"- {menu} : {price}".strip(" :"))

    if max_items is not None:
        lines = lines[:max_items]
    return lines

# =========================
# 1) 사용자 문장 -> CSV 하드필터 조건 추출
# =========================
def parse_filter_condition(q: str):
    cond = {}

    m = re.search(r"(\d+)\s*분", q)
    if m:
        cond["max_time"] = int(m.group(1))

    m = re.search(r"(\d+(\.\d+)?)\s*km", q.lower())
    if m:
        cond["max_dist"] = float(m.group(1))

    # 업태: CSV 업태구분명 unique 중 문장에 포함되는 걸 찾음
    for keyword in df_meta["업태구분명"].dropna().unique():
        kw = str(keyword).strip()
        if kw and kw in q:
            cond["업태"] = kw
            break

    return cond

# =========================
# 2) CSV에서 1차 하드필터로 allowed_names 만들기
# =========================
def hard_filter_names_from_csv(user_text: str, max_minutes=None):
    cond = parse_filter_condition(user_text)
    if max_minutes is not None:
        cond["max_time"] = int(max_minutes)

    temp = df_meta.copy()

    if "max_time" in cond:
        temp = temp[temp["도보시간_분"].notna() & (temp["도보시간_분"] <= cond["max_time"])]

    if "max_dist" in cond:
        temp = temp[temp["도보거리_km"].notna() & (temp["도보거리_km"] <= cond["max_dist"])]

    if "업태" in cond:
        temp = temp[temp["업태구분명"].astype(str).str.contains(cond["업태"], na=False)]

    names = set(temp["사업장명"].astype(str).str.strip().tolist())
    names.discard("")
    return names, cond

# =========================
# 3) Chroma 검색을 allowed_names 집합 내부로 제한($in)
#    + ✅ 거리(도보시간) 가중치로 재랭킹
# =========================
def chroma_search_only_allowed(
    query_text: str,
    allowed_names: set,
    exclude_names: set,
    top_k=FINAL_K
):
    where = {"사업장명": {"$in": list(allowed_names)}}

    results = db.similarity_search_with_relevance_scores(
        query_text,
        k=CHROMA_K,
        filter=where
    )

    rows = []
    for doc, rel in results:
        meta = doc.metadata or {}
        name = str(meta.get("사업장명", "")).strip()
        if not name:
            continue
        if name in exclude_names:
            continue

        if name in meta_index.index:
            walk = safe_num(meta_index.loc[name].get("도보시간_분", None))
        else:
            walk = safe_num(meta.get("도보시간_분", None))

        rows.append((doc, float(rel), walk))

    if not rows:
        return []

    times = np.array([r[2] if r[2] is not None else np.nan for r in rows], dtype=float)
    valid = np.isfinite(times)

    if valid.any():
        tmin, tmax = float(np.nanmin(times)), float(np.nanmax(times))
        if tmax == tmin:
            tnorm = np.zeros_like(times)
        else:
            tnorm = (times - tmin) / (tmax - tmin)
        tnorm[~valid] = 1.0
    else:
        tnorm = np.zeros_like(times)

    scored = []
    for (doc, rel, _walk), pen in zip(rows, tnorm):
        final_score = rel - (LAMBDA_DISTANCE * float(pen))
        scored.append((final_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    picked = []
    seen = set()
    for _score, doc in scored:
        name = str((doc.metadata or {}).get("사업장명", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        picked.append(doc)
        if len(picked) >= top_k:
            break

    return picked

# =========================
# GPT로 의도/조건 추출
# =========================
def parse_user_message(history_messages, user_message):
    recent = history_messages[-8:]
    recent_text = ""
    for m in recent:
        role = "사용자" if m["role"] == "user" else "챗봇"
        recent_text += f"{role}: {m['content']}\n"

    system = """
너는 "맛집 추천 챗봇"의 의도 파서다. 사용자의 자연어 발화를 아래 JSON으로 구조화한다.
반드시 JSON만 출력한다. 다른 텍스트 금지.

JSON 스키마:
{
  "search_query": string|null,
  "max_minutes": number|null,
  "want_next": boolean,
  "reset": boolean,
  "hard_constraints_text": string|null
}

규칙:
- want_next=true: 사용자가 '다시 추천', '다른 데', '별로야', '마음에 안 들어', '다음', '다른 곳' 등으로
  "바로 다음 후보"를 원하면 true.
  단, 사용자가 메뉴/분위기/조건을 크게 바꾸면 want_next=false로 두고 search_query를 새로 구성.
- max_minutes: '15분', '20분 이내', '10분 안쪽' 등 도보시간 제한이 있으면 숫자만.
- reset=true: '처음부터', '리셋', '조건 초기화', '새로 찾자' 등.
- search_query: 지금 턴의 추천 의도(메뉴/분위기/상황 포함) 한 문장 요약.
- hard_constraints_text: 강조 조건(가성비/조용함/인테리어/데이트 등) 짧게 요약.
"""

    user = f"""
최근 대화:
{recent_text}

이번 사용자 발화:
{user_message}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0
    )

    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except Exception:
        data = {"search_query": None, "max_minutes": None, "want_next": False, "reset": False, "hard_constraints_text": None}

    data.setdefault("search_query", None)
    data.setdefault("max_minutes", None)
    data.setdefault("want_next", False)
    data.setdefault("reset", False)
    data.setdefault("hard_constraints_text", None)
    return data

# =========================
# 추천 문장 생성 (메타데이터는 CSV 기준으로 출력)
# =========================
def generate_reco_text(user_message, state, doc):
    meta = doc.metadata or {}
    name = str(meta.get("사업장명", "")).strip()

    if name in meta_index.index:
        row = meta_index.loc[name]
        category = str(row.get("업태구분명", "")).strip()
        addr = str(row.get("지번주소", "")).strip()
        walk_min = safe_num(row.get("도보시간_분", None))
        menu = str(row.get("대표메뉴_메뉴", "")).strip()
        price = str(row.get("대표메뉴_가격", "")).strip()
    else:
        category = str(meta.get("업태구분명", "")).strip()
        addr = str(meta.get("지번주소", "")).strip()
        walk_min = safe_num(meta.get("도보시간_분", None))
        menu = str(meta.get("대표메뉴_메뉴", "")).strip()
        price = str(meta.get("대표메뉴_가격", "")).strip()

    walk_line = f"🚶 도보 약 {walk_min:g}분" if walk_min is not None else "🚶 도보시간 정보 없음"

    # 네이버 지도 검색 url 생성
    search_keyword = name
    if addr:
        splits = addr.split()
        dong_name = next((s for s in splits if s.endswith("동")), "")
        if dong_name:
            search_keyword = f"{dong_name} {name}"

    query_encoded = quote(search_keyword)
    map_url = f"https://map.naver.com/p/search/{query_encoded}"

    system = "너는 홍대 맛집 추천 전문 챗봇이다."
    prompt = f"""
사용자 발화: {user_message}
현재 조건(state): {json.dumps(state, ensure_ascii=False)}

추천할 식당 정보:
- 식당명: {name}
- 업태: {category}
- 주소: {addr}

요구사항:
- 식당은 이 1곳만 추천한다.
- 3~5문장으로 자연스럽고 현실적으로 추천 이유를 말한다.
- "요약:" 같은 요약 라인은 절대 쓰지 마라.
- 광고 문구처럼 과장하지 마라.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=0.4
    )

    body = clean_llm_text(resp.choices[0].message.content.strip())

    out = []
    out.append(f"### 🍽️ {name}")
    out.append(walk_line)
    out.append("")
    out.append(body)

    # ✅ 여기만 바뀜: 메뉴/가격을 "메뉴 : 가격" 리스트로 출력
    if menu or price:
        out.append("")
        out.append("**대표메뉴 / 가격**")
        out.extend(format_menu_price_lines(menu, price, max_items=8))

    if addr:
        out.append(f"\n📍[식당 자세히 보기]({map_url})")

    return "\n".join(out).strip(), name

# =========================
# UI
# =========================
st.title("사용자 조건 기반 맛집 추천")
st.caption("조건을 말하면 1곳만 추천해줘요. 마음에 안 들면 ‘다른 곳 추천해줘’라고 말하면 다음 후보로 넘어갑니다.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "state" not in st.session_state:
    st.session_state["state"] = {"search_query": None, "max_minutes": None, "hard_constraints_text": None}

if "result_docs" not in st.session_state:
    st.session_state["result_docs"] = []
if "result_idx" not in st.session_state:
    st.session_state["result_idx"] = 0

if "shown_names" not in st.session_state:
    st.session_state["shown_names"] = set()
if "reco_history" not in st.session_state:
    st.session_state["reco_history"] = []

with st.sidebar:
    st.subheader("📌 추천 내역")
    if st.session_state["reco_history"]:
        for i, nm in enumerate(reversed(st.session_state["reco_history"]), start=1):
            st.write(f"{i}. {nm}")
    else:
        st.write("아직 추천한 식당이 없습니다.")

    if st.button("대화/상태 초기화"):
        st.session_state.clear()
        st.rerun()

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("(예: 20분 안에 갈 수 있는 가성비 좋은 파스타 집 추천해줘)")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    parsed = parse_user_message(st.session_state["messages"], user_input)

    if parsed.get("reset"):
        st.session_state["state"] = {
            "search_query": parsed.get("search_query"),
            "max_minutes": parsed.get("max_minutes"),
            "hard_constraints_text": parsed.get("hard_constraints_text"),
        }
        st.session_state["result_docs"] = []
        st.session_state["result_idx"] = 0
        st.session_state["shown_names"] = set()
        st.session_state["reco_history"] = []
    else:
        if parsed.get("search_query"):
            st.session_state["state"]["search_query"] = parsed.get("search_query")
        if parsed.get("hard_constraints_text"):
            st.session_state["state"]["hard_constraints_text"] = parsed.get("hard_constraints_text")
        if parsed.get("max_minutes") is not None:
            st.session_state["state"]["max_minutes"] = parsed.get("max_minutes")

    want_next = bool(parsed.get("want_next"))

    with st.chat_message("assistant"):
        with st.spinner("조건에 맞는 식당을 찾는 중..."):
            state = st.session_state["state"]
            search_query = state.get("search_query")
            max_minutes = state.get("max_minutes")
            hard_text = state.get("hard_constraints_text")

            if not search_query:
                msg = "원하는 메뉴/분위기를 조금만 더 구체적으로 말해줄래요?"
                st.markdown(msg)
                st.session_state["messages"].append({"role": "assistant", "content": msg})
            else:
                effective_query = search_query if not hard_text else f"{search_query}. 조건: {hard_text}"

                if want_next and st.session_state["result_docs"] and st.session_state["result_idx"] < len(st.session_state["result_docs"]):
                    doc = st.session_state["result_docs"][st.session_state["result_idx"]]
                    st.session_state["result_idx"] += 1
                else:
                    allowed_names, cond = hard_filter_names_from_csv(user_input, max_minutes=max_minutes)

                    if not allowed_names:
                        msg = "CSV 조건(시간/거리/업태)에 맞는 식당이 없어요 😢 조건을 완화해주시면 다시 찾아볼게요."
                        st.markdown(msg)
                        st.session_state["messages"].append({"role": "assistant", "content": msg})
                        doc = None
                    else:
                        try:
                            docs = chroma_search_only_allowed(
                                query_text=effective_query,
                                allowed_names=allowed_names,
                                exclude_names=st.session_state["shown_names"],
                                top_k=FINAL_K
                            )
                        except Exception as e:
                            st.error(f"Chroma $in 필터가 지원되지 않거나 오류가 발생했습니다: {e}")
                            docs = []

                        st.session_state["result_docs"] = docs
                        st.session_state["result_idx"] = 0

                        if not docs:
                            msg = "조건에 맞는 식당은 있는데, 감성/리뷰 기준으로 매칭이 약헸어요 😢 표현을 바꿔서 말해줄래요?"
                            st.markdown(msg)
                            st.session_state["messages"].append({"role": "assistant", "content": msg})
                            doc = None
                        else:
                            doc = docs[st.session_state["result_idx"]]
                            st.session_state["result_idx"] += 1

                if doc is not None:
                    name = (doc.metadata or {}).get("사업장명")
                    if name:
                        st.session_state["shown_names"].add(name)

                    reco_text, reco_name = generate_reco_text(user_input, state, doc)
                    st.markdown(reco_text)
                    st.session_state["messages"].append({"role": "assistant", "content": reco_text})

                    if reco_name and (not st.session_state["reco_history"] or st.session_state["reco_history"][-1] != reco_name):
                        st.session_state["reco_history"].append(reco_name)

                    # 이전 답변 잔상 제거
                    st.rerun()
