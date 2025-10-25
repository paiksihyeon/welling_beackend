from fastapi import APIRouter, HTTPException
from app.services.vector_service import (
    load_policy_vectors,
    load_region_vectors,
    find_top_gap_topics,
    cosine_similarity,
    aggregate_topic_vectors
)
from openai import OpenAI
import os, json

router = APIRouter(prefix="/rag/action", tags=["RAG - Policy Action"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@router.get("/{region_name}")
def recommend_policy_action(region_name: str):
    """
    ✅ LLM + RAG 정책 기반 정책 액션 제안 API (Cross-Region 비교 포함)
    1️⃣ app/files/{region_name}_vectors.json → 갭이 큰 주제 1개 선택
    2️⃣ 다른 지역 *_vectors.json 파일 중 동일 주제의 유사도 계산 → 상위 3개 지역 선택
    3️⃣ app/files/policy_vectors.json → 정책 벡터 유사도 계산
    4️⃣ GPT → JSON 형식의 정책 개선 카드 생성
    """

    base_path = "app/files"

    # 1️⃣ 지역 벡터 로드 및 변환
    try:
        region_vectors_raw = load_region_vectors(region_name)
        region_vectors = (
            aggregate_topic_vectors(region_vectors_raw)
            if isinstance(region_vectors_raw, list)
            else region_vectors_raw
        )
        top_topic = find_top_gap_topics(region_vectors, top_k=1)[0]
        topic_vec = region_vectors[top_topic]["vector"]
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{region_name} 지역 벡터를 불러오지 못했습니다: {e}")

    # 2️⃣ Cross-Region 비교
    try:
        similarities = []
        for file in os.listdir(base_path):
            if not file.endswith("_vectors.json"):
                continue
            other_region = file.replace("_vectors.json", "")
            if other_region == region_name:
                continue

            try:
                other_vectors_raw = load_region_vectors(other_region)
                other_vectors = (
                    aggregate_topic_vectors(other_vectors_raw)
                    if isinstance(other_vectors_raw, list)
                    else other_vectors_raw
                )

                if top_topic in other_vectors:
                    sim = cosine_similarity(topic_vec, other_vectors[top_topic]["vector"])
                    similarities.append((other_region, sim))
            except Exception:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_related_regions = similarities[:3]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"다른 지역 정책 비교 중 오류 발생: {e}")

    # 3️⃣ 정책 벡터 로드 및 유사도 계산 (list/dict 자동 대응)
    try:
        policy_vectors_raw = load_policy_vectors()
        similarities = []

        # ✅ list 구조
        if isinstance(policy_vectors_raw, list):
            for entry in policy_vectors_raw:
                name = entry.get("policy_name") or entry.get("title") or "Unknown Policy"
                vec = entry.get("vector")
                if not vec:
                    continue
                score = cosine_similarity(topic_vec, vec)
                similarities.append((name, score))

        # ✅ dict 구조
        elif isinstance(policy_vectors_raw, dict):
            for name, vec in policy_vectors_raw.items():
                score = cosine_similarity(topic_vec, vec)
                similarities.append((name, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_policies = similarities[:3]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"정책 벡터 비교 중 오류 발생: {e}")

    # 4️⃣ GPT 프롬프트 구성
    region_refs = "\n".join(
        [f"- {r} 지역 (유사도 {round(s,3)})" for r, s in top_related_regions]
    )
    policy_refs = "\n".join(
        [f"[참고 정책 {i+1}] {name}" for i, (name, _) in enumerate(top_policies)]
    )

    prompt = f"""
=== 역할 정의 ===
너는 'Welling' 프로젝트의 AI 정책 분석 엔진이다.
너의 임무는 제공된 [지역 정보], [다른 지역의 유사 정책], [참고 정책 문서]를 바탕으로
'{region_name}' 지역의 실행 가능한 정책 개선 액션을 생성하는 것이다.

=== 입력 1: 지역 정보 ===
[지역명: {region_name}]
[핵심 주제: {top_topic}]

=== 입력 2: 다른 지역의 동일 주제 유사도 상위 3개 ===
{region_refs}

=== 입력 3: 참고 정책 문서 (RAG 검색 결과) ===
{policy_refs}

=== 지시 사항 ===
1. 다른 지역의 정책 성공 요인을 분석하고, {region_name} 지역에 맞는 개선 방안을 제안하라.
2. [다른 지역]과 [정책 문서]를 근거로 한 실행 가능한 "rag_action_card"를 1~2줄로 작성하라.
3. 반드시 아래 JSON 형식으로만 반환하라.

=== 출력 형식 (JSON만 반환) ===
{{
  "rag_action_card": "(AI가 생성한 정책 제언)",
  "reference_regions": ["지역명1", "지역명2", "지역명3"],
  "reference_policies": ["정책명1", "정책명2", "정책명3"]
}}
"""

    # 5️⃣ GPT 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 지역정책 분석 및 기획 전문가이다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        result_json = json.loads(response.choices[0].message.content)

        print(f"[rag_action] ✅ '{region_name}' 지역 '{top_topic}' 정책 액션 제안 완료")

        return {
            "region": region_name,
            "main_topic": top_topic,
            "related_regions": [
                {"region_name": r, "similarity": round(s, 3)} for r, s in top_related_regions
            ],
            "similar_policies": [
                {"policy_name": p[0], "similarity": round(p[1], 3)} for p in top_policies
            ],
            "result": result_json
        }

    except Exception as e:
        print(f"[rag_action] ❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"GPT 요청 실패: {e}")
