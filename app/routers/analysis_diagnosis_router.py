# app/routers/analysis_diagnosis_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.utils.models import SentimentAnalysisLog
from app.services.vector_service import load_region_vectors, find_top_gap_topics
from openai import OpenAI
import os, json

router = APIRouter(prefix="/analysis/diagnosis", tags=["Analysis - Diagnosis"])
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@router.get("/{region_name}")
def diagnose_region(region_name: str, db: Session = Depends(get_db)):
    """
    ✅ 지역별 시민 여론 + 벡터 갭 기반 문제진단 API
    1️⃣ SentimentAnalysisLog에서 시민 여론 불러오기
    2️⃣ app/files/{region_name}_vectors.json 에서 갭이 큰 주제 추출
    3️⃣ GPT에게 Welling 표준 프롬프트(JSON 구조)로 요청
    """

    # 1️⃣ 시민 여론 데이터 불러오기
    records = db.query(SentimentAnalysisLog).filter(
        SentimentAnalysisLog.region == region_name
    ).all()
    if not records:
        raise HTTPException(status_code=404, detail=f"{region_name} 지역의 여론 데이터가 없습니다.")

    texts = [r.text for r in records if r.text]
    combined_text = "\n".join(texts[:30])  # 상위 30개까지만 사용

    # 2️⃣ 갭이 큰 주제 추출
    try:
        vectors = load_region_vectors(region_name)
        top_topics = find_top_gap_topics(vectors, top_k=3)
        top_topic_str = ", ".join(top_topics)
    except Exception:
        top_topic_str = "교통, 주거, 의료 등 생활 전반"

    # 3️⃣ 여론 희소성 판단용 데이터
    record_count = len(records)
    if record_count > 50:
        scarcity_level = "여론이 매우 활발함"
    elif record_count > 20:
        scarcity_level = "일정 수준의 여론 활동이 존재함"
    else:
        scarcity_level = "여론 데이터가 부족하거나 정보 접근성이 낮은 지역임"

    # 4️⃣ Welling 프롬프트 구성
    prompt = f"""
=== 역할 정의 ===
너는 'Welling' 프로젝트의 AI 정책 분석 엔진이다.
너의 임무는 제공된 [지역 여론]을 분석하고, 이 지역의 '문제 진단'을 수행하는 것이다.

=== 입력 1: 지역 여론 데이터 ===
[지역: {region_name}]
[주제: {top_topic_str}]
[댓글 수: {record_count}개]
[댓글 내용]
{combined_text}

=== 지시 사항 ===
1. [지역 여론 데이터]를 읽고, 주민들의 핵심 불만 사항을 3~4줄로 요약하여 "problem_summary"를 생성하라.
2. [댓글 수]를 기반으로 여론의 활발도·희소성을 평가하여 "scarcity_insight"를 생성하라.

=== 출력 형식 (JSON으로만 반환) ===
{{
  "problem_summary": "(AI가 생성한 요약문)",
  "scarcity_insight": "(AI가 생성한 희소성 판단 문장)"
}}
"""

    # 5️⃣ GPT API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 사회정책 및 여론 분석 전문가이다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        print(f"[analysis_diagnosis] ✅ '{region_name}' 문제진단 완료")

        return {
            "region": region_name,
            "top_topics": top_topic_str,
            "record_count": record_count,
            "scarcity_level": scarcity_level,
            "result": result
        }

    except Exception as e:
        print(f"[analysis_diagnosis] ❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"GPT 요청 실패: {e}")
