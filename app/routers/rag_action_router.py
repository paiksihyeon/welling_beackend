from fastapi import APIRouter, HTTPException
from app.services.vector_service import (
    load_policy_vectors,
    find_top_gap_topics,
    cosine_similarity,
    aggregate_topic_vectors
)
from openai import OpenAI
import os, json

router = APIRouter(prefix="/rag/action", tags=["RAG - Policy Action"])
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def safe_load_region_vectors(region_name: str):
    """다양한 파일명 패턴으로 지역 벡터 JSON 로드"""
    base_path = "app/files"
    candidates = [
        os.path.join(base_path, f"{region_name}_vectors_e5.json"),
        os.path.join(base_path, f"{region_name}_vectors.json"),
        os.path.join(base_path, f"{region_name}.json"),
        os.path.join(base_path, f"{region_name.lower()}_vectors_e5.json"),
        os.path.join(base_path, f"{region_name.capitalize()}_vectors_e5.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[rag_action] ✅ 지역 벡터 로드 완료: {path}")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"⚠️ {region_name} 지역 벡터 파일을 찾을 수 없습니다.")


@router.get("/{region_name}")
def recommend_policy_action(region_name: str):
    """LLM + RAG 기반 정책 개선 제안 API"""
    base_path = "app/files"

    # 1️⃣ 지역 벡터 로드 및 주제 선택
    try:
        region_vectors_raw = safe_load_region_vectors(region_name)
        region_vectors = (
            aggregate_topic_vectors(region_vectors_raw)
            if isinstance(region_vectors_raw, list)
            else region_vectors_raw
        )

        top_topic_info = find_top_gap_topics(region_vectors=region_vectors, region_name=region_name, top_k=1)[0]
        top_topic_en = top_topic_info.get("topic_en")
        top_topic_kr = top_topic_info.get("topic")

        # ✅ 유연한 키 매칭 (띄어쓰기, 대소문자, 한글/영문 모두 대응)
        normalized_keys = {k.replace(" ", "").lower(): k for k in region_vectors.keys()}
        target_candidates = [
            top_topic_en.replace(" ", "").lower(),
            top_topic_kr.replace(" ", "").lower()
        ]

        found_key = None
        for cand in target_candidates:
            if cand in normalized_keys:
                found_key = normalized_keys[cand]
                break

        if found_key:
            topic_vec = region_vectors[found_key]["vector"]
            top_topic = found_key
        else:
            raise KeyError(f"'{top_topic_en}' 또는 '{top_topic_kr}' 주제를 region_vectors에서 찾을 수 없습니다.")

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"{region_name} 지역 벡터를 불러오지 못했습니다: {e}")

    # 2️⃣ Cross-Region 비교
    try:
        similarities = []
        for file in os.listdir(base_path):
            if not file.endswith("_vectors_e5.json"):
                continue
            other_region = file.replace("_vectors_e5.json", "")
            if other_region == region_name:
                continue

            try:
                other_vectors_raw = safe_load_region_vectors(other_region)
                other_vectors = (
                    aggregate_topic_vectors(other_vectors_raw)
                    if isinstance(other_vectors_raw, list)
                    else other_vectors_raw
                )

                # ✅ 동일한 방식으로 키 매칭 수행
                normalized_other = {k.replace(" ", "").lower(): k for k in other_vectors.keys()}
                for cand in target_candidates:
                    if cand in normalized_other:
                        matched_key = normalized_other[cand]
                        sim = cosine_similarity(topic_vec, other_vectors[matched_key]["vector"])
                        similarities.append((other_region, sim))
                        break

            except Exception:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_related_regions = similarities[:3]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"다른 지역 정책 비교 중 오류 발생: {e}")

    # 3️⃣ 정책 벡터 로드 및 유사도 계산
    try:
        policy_vectors_raw = load_policy_vectors()
        similarities = []

        if isinstance(policy_vectors_raw, list):
            for entry in policy_vectors_raw:
                name = entry.get("policy_name") or entry.get("title") or "Unknown Policy"
                vec = entry.get("vector")
                if not vec:
                    continue
                score = cosine_similarity(topic_vec, vec)
                similarities.append((name, score))

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
