# app/services/rag_service.py

import json
import numpy as np
from datetime import datetime
from openai import OpenAI
from sqlalchemy.orm import Session
from app.utils.models import RegionData, RagSummary

"""
rag_service.py
AI 모델로부터 받은 정책 요약(RAG 결과) 저장, 추천, 인사이트 분석을 통합 관리하는 서비스 로직입니다.
"""

client = OpenAI()  # OPENAI_API_KEY 필요 (.env에 설정)


# =========================================================
# 1. 기본 RAG 요약 저장 로직
# =========================================================
def save_rag_summary(db: Session, region_name: str, topic: str, summary: str):
    """RAG 요약 결과를 저장하며 지역 정보를 자동 연동"""
    try:
        # 지역 정보 조회 또는 신규 생성
        region = db.query(RegionData).filter(RegionData.region_name == region_name).first()
        if not region:
            print(f"[rag_service] {region_name} 지역이 존재하지 않아 새로 생성합니다.")
            region = RegionData(
                region_name=region_name,
                policy_avg_score=0.0,
                sentiment_avg_score=0.0,
                gap_score=0.0,
                updated_at=datetime.utcnow(),
            )
            db.add(region)
            db.commit()
            db.refresh(region)

        # 동일 주제의 기존 요약 삭제 (중복 방지)
        db.query(RagSummary).filter(
            RagSummary.region_id == region.id,
            RagSummary.topic == topic,
        ).delete()

        # 새로운 요약 삽입
        new_summary = RagSummary(
            region_id=region.id,
            topic=topic,
            summary=summary,
            created_at=datetime.utcnow(),
        )
        db.add(new_summary)
        region.updated_at = datetime.utcnow()
        db.commit()

        print(f"[rag_service] '{region_name}' 지역의 '{topic}' 요약 저장 완료")
        return {"status": "success", "region": region_name, "topic": topic}

    except Exception as e:
        db.rollback()
        print(f"[rag_service] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}


# =========================================================
# 2. 벡터 로드 및 유사도 계산 유틸
# =========================================================
def load_vectors(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


# =========================================================
# 3. 정책 추천 (기존 rag_recommend_service)
# =========================================================
def recommend_policies(region_name: str, topic: str, db: Session, top_k: int = 5):
    """
    1️⃣ policy_vectors.json 불러오기
    2️⃣ 유사도 기반 상위 N개 정책 선택
    3️⃣ ChatGPT API를 통해 요약 및 추천 생성
    4️⃣ 결과를 DB에 저장 후 반환
    """
    print(f"[RAG Recommend] {region_name} 지역 / 주제: {topic} 추천 시작")

    vectors = load_vectors("app/files/policy_vectors.json")

    # ✅ JSON 구조에 따라 분기 처리
    if isinstance(vectors, dict):
        target_vector = vectors.get(topic)
    elif isinstance(vectors, list):
        # 리스트 형태일 경우 topic 키를 가진 항목 찾기
        match = next((v for v in vectors if v.get("topic") == topic), None)
        target_vector = match["vector"] if match else None
    else:
        target_vector = None

    if not target_vector:
        raise ValueError(f"주제 '{topic}' 에 대한 벡터를 찾을 수 없습니다.")

    # DB에서 모든 정책 벡터 불러오기
    all_summaries = db.query(RagSummary).all()
    scored = []
    for s in all_summaries:
        if not getattr(s, "embedding", None):
            continue
        try:
            vec = json.loads(s.embedding)
            score = cosine_similarity(target_vector, vec)
            scored.append((s, score))
        except Exception as e:
            print(f"[RAG Recommend] 유사도 계산 오류: {e}")
            continue

    # 유사도 순 정렬 후 상위 N개 선택
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    top_policies = [s.summary for s, _ in scored] if scored else ["유사 정책 없음"]

    # ChatGPT API 호출 → 정책 추천 생성
    prompt = (
        f"지역 '{region_name}'의 '{topic}' 주제와 유사한 타 지역 정책 사례를 분석하고 "
        f"해당 지역에 적용 가능한 정책 제안을 간결히 요약하세요.\n\n"
        f"유사 정책 요약:\n" + "\n".join(top_policies)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 사회정책 연구 전문가입니다."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
        )
        recommendation = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[RAG Recommend] ChatGPT 호출 실패: {e}")
        recommendation = "정책 제안 생성 실패"

    print(f"[RAG Recommend] 추천 결과 생성 완료")

    # 결과를 DB에 저장
    new_summary = RagSummary(
        region_id=None,
        topic=f"추천:{topic}",
        summary=recommendation,
        created_at=datetime.utcnow(),
    )
    db.add(new_summary)
    db.commit()

    return {
        "topic": topic,
        "recommendation": recommendation,
        "similar_examples": top_policies,
    }

# =========================================================
# 4. 종합 인사이트 생성 (기존 rag_insight_service)
# =========================================================
def generate_rag_insight(region_name: str, topic: str, db: Session):
    """
    RAG 종합 인사이트 생성 파이프라인
    1️⃣ 시민 불만 요약
    2️⃣ 유사 정책 검색 및 추천
    3️⃣ 최종 정책 개선 방향 생성
    """
    print(f"[RAG Insight] {region_name} 지역 / {topic} 주제 분석 시작")

    # 1️⃣ 여론 벡터 로드
    sentiment_vectors = load_vectors("app/files/sentiment_vectors.json")
    region_vec = sentiment_vectors.get(region_name)
    topic_vec = region_vec.get(topic) if region_vec else None
    if topic_vec is None:
        raise ValueError(f"{region_name} 지역의 {topic} 벡터를 찾을 수 없습니다.")

    # 2️⃣ 유사한 여론 문장 10~15개 추출
    all_opinions = sentiment_vectors.get("opinions", [])
    similarities = []
    for op in all_opinions:
        sim = cosine_similarity(topic_vec, op["vector"])
        similarities.append((op["text"], sim))
    top_opinions = [t for t, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:10]]

    # 3️⃣ 시민 불만 요약 생성
    citizen_prompt = (
        f"{region_name} 시민들의 '{topic}' 관련 주요 의견을 다음 문장에서 요약하세요.\n\n"
        + "\n".join(top_opinions)
    )
    citizen_summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "사회정책 전문가로서 시민 의견을 요약하세요."},
            {"role": "user", "content": citizen_prompt},
        ],
        max_tokens=300,
    ).choices[0].message.content.strip()

    # 4️⃣ 유사 정책 검색
    policy_vectors = load_vectors("app/files/policy_vectors.json")
    topic_vec_policy = policy_vectors.get(topic)
    all_summaries = db.query(RagSummary).all()
    similarities = []
    for s in all_summaries:
        if not getattr(s, "embedding", None):
            continue
        try:
            vec = json.loads(s.embedding)
            sim = cosine_similarity(topic_vec_policy, vec)
            similarities.append((s.summary, sim))
        except Exception:
            continue
    top_policies = [t for t, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:5]]

    # 5️⃣ 최종 종합 요약 생성
    final_prompt = (
        f"지역 '{region_name}'의 '{topic}'에 대한 시민 불만과 유사 정책을 종합하여 "
        f"정책 개선 방향을 제안하세요.\n\n"
        f"시민 불만 요약:\n{citizen_summary}\n\n"
        f"유사 정책 사례:\n" + "\n".join(top_policies)
    )
    final_summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "정책분석가로서 개선 방향을 제시하세요."},
            {"role": "user", "content": final_prompt},
        ],
        max_tokens=400,
    ).choices[0].message.content.strip()

    print(f"[RAG Insight] {region_name} / {topic} 결과 생성 완료")

    return {
        "citizen_summary": citizen_summary,
        "policy_recommendation": top_policies,
        "final_summary": final_summary,
    }


# =========================================================
# 5. 모듈 로드 확인
# =========================================================
if __name__ == "__main__":
    print("[rag_service.py] 모듈이 정상적으로 로드되었습니다.")
