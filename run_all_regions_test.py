"""
run_all_regions_test.py
--------------------------------
✅ 모든 지역(region_name)에 대해
   /api/analysis/diagnosis/{region_name},
   /api/rag/action/{region_name}
   결과를 콘솔로 출력
"""

import os
import json
from dotenv import load_dotenv
from app.services.vector_service import find_top_gap_topics, load_region_vectors
from openai import OpenAI
from datetime import datetime
from urllib.parse import unquote
from difflib import get_close_matches

# ✅ .env 파일 로드
load_dotenv()

# ✅ OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ 테스트할 지역 리스트
regions = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
]


# ✅ diagnosis + rag_action 실행 함수
def run_for_region(region_name: str):
    print(f"\n==============================")
    print(f"🏙️  {region_name} 지역 결과 시작")
    print(f"==============================")

    # -----------------------------
    # ✅ 1️⃣ diagnosis 부분
    # -----------------------------
    try:
        region_name = unquote(region_name).strip()
        vectors = load_region_vectors(region_name)
        top_topics_info = find_top_gap_topics(region_name=region_name, top_k=3)
        top_topics = [t["topic"] for t in top_topics_info]
        top_topic_str = ", ".join(top_topics)
        diagnosed_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[Diagnosis] 🔹 Top Topics: {top_topic_str}")
        print(f"[Diagnosis] 🔹 Time: {diagnosed_time}")

    except Exception as e:
        print(f"[Diagnosis] ❌ 오류 발생: {e}")
        return

    # -----------------------------
    # ✅ 2️⃣ rag_action 부분
    # -----------------------------
    try:
        # ✅ RAG가 사용하는 핵심 main_topic = gap 1위 주제
        main_topic = top_topics_info[0]["topic"]
        main_topic_en = top_topics_info[0]["topic_en"]
        print(f"[RAG] 🔹 Main Topic: {main_topic}")

        region_vectors = vectors
        # 영어/한글 topic 키 탐색 (자동 매칭)
        if main_topic in region_vectors:
            topic_vec = region_vectors[main_topic]["vector"]
        elif main_topic_en in region_vectors:
            topic_vec = region_vectors[main_topic_en]["vector"]
        else:
            possible_keys = list(region_vectors.keys())
            close_match = get_close_matches(main_topic, possible_keys, n=1, cutoff=0.4)
            if close_match:
                topic_vec = region_vectors[close_match[0]]["vector"]
                print(f"[RAG] ⚠️ '{main_topic}' 대신 '{close_match[0]}' 키를 사용함")
            else:
                raise KeyError(f"'{main_topic}' 주제 벡터를 찾을 수 없습니다.")

        # GPT 요청 (요약 + 정책 제안)
        prompt = f"""
        === 역할 정의 ===
        너는 'Welling' 프로젝트의 정책 분석 엔진이다.
        아래 지역의 주요 주제에 대한 정책 개선 제안을 요약하라.

        [지역명: {region_name}]
        [핵심 주제(Main Topic): {main_topic}]
        [관련 Top3 주제: {top_topic_str}]

        === 출력 형식 (JSON만 반환) ===
        {{
          "problem_summary": "(핵심 문제 요약)",
          "policy_suggestion": "(정책 개선 제안)"
        }}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 정책 분석 및 행정 전문가이다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        print(f"[RAG] ✅ 정책 제안 생성 완료")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"[RAG] ❌ 오류 발생: {e}")


# ✅ 실행 시작
if __name__ == "__main__":
    print("🚀 모든 지역에 대한 diagnosis + rag_action 테스트 시작\n")
    for region in regions:
        run_for_region(region)
    print("\n🎉 모든 지역 결과 출력 완료!")
