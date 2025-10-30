from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.utils.models import SentimentAnalysisLog
from app.services.vector_service import load_region_vectors, find_top_gap_topics
from openai import OpenAI
from urllib.parse import unquote
from datetime import datetime
import os, json

router = APIRouter(prefix="/analysis/diagnosis", tags=["Analysis - Diagnosis"])
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@router.get("/{region_name}")
def diagnose_region(region_name: str, db: Session = Depends(get_db)):
    """
    ✅ 지역별 시민 여론 + 갭 기반 문제진단 API
    1️⃣ SentimentAnalysisLog에서 시민 여론 불러오기
    2️⃣ gap_score.csv 기반 상위 3개 주제 추출
    3️⃣ GPT에게 분석 요청
    """

    # ✅ 한글 URL 복원 및 공백 제거
    region_name = unquote(region_name).strip()

    # 1️⃣ 시민 여론 데이터 불러오기 (부분 일치 허용)
    records = db.query(SentimentAnalysisLog).filter(
        SentimentAnalysisLog.region.like(f"%{region_name}%")
    ).all()

    if not records:
        raise HTTPException(status_code=404, detail=f"{region_name} 지역의 여론 데이터가 없습니다.")

    texts = [r.text for r in records if r.text]
    combined_text = "\n".join(texts[:30])  # 상위 30개까지만 사용

    # 2️⃣ gap_score.csv 기반 갭이 큰 주제 추출
    try:
        top_topics = find_top_gap_topics(region_name=region_name, top_k=3)
        top_topic_str = ", ".join([t["topic"] for t in top_topics])
    except Exception as e:
        print(f"[analysis_diagnosis] ⚠️ 주제 추출 실패: {e}")
        top_topic_str = "교통, 주거, 의료 등 생활 전반"

    # 3️⃣ 여론 희소성 판단용 데이터
    record_count = len(records)
    if record_count > 50:
        scarcity_level = "여론이 매우 활발함"
    elif record_count > 20:
        scarcity_level = "일정 수준의 여론 활동이 존재함"
    else:
        scarcity_level = "여론 데이터가 부족하거나 정보 접근성이 낮은 지역임"

    # 4️⃣ GPT 프롬프트 구성
    prompt = f"""
    === 역할 정의 ===
    너는 'Welling' 프로젝트의 AI 정책 분석 엔진이다.
    너의 임무는 제공된 [지역 여론]을 정밀 분석하여, 해당 지역의 정책 문제와 여론 특성을 깊이 있게 진단하는 것이다.

    === 입력 데이터 ===
    [지역: {region_name}]
    [주제: {top_topic_str}]
    [여론 내용]
    {combined_text}

    === 작성 지침 ===
    1. **"problem_summary"** 항목에서는 단순 요약이 아니라 다음 요소를 포함하라:
       - 주민들의 불만과 요구를 구체적으로 기술하되, 표면적 진술에 그치지 말고 **원인·맥락·파급효과**를 함께 서술하라.
       - 6~8줄 분량으로 작성하되, 정책·사회적 배경을 연결하여 **심층 분석형 문단**으로 표현하라.
       - “무엇이 문제인가 → 왜 발생했는가 → 어떤 사회적 영향이 있는가”의 구조로 작성하라.
       - 예시: “부산 지역은 △△의 구조적 문제로 인해 청년층의 이탈이 심화되고 있으며, 이는 △△정책의 한계와 연결된다.”

    2. **"scarcity_insight"** 항목에서는 여론의 질적 특성과 감정 흐름을 분석하라:
       - 단순히 활발함을 진술하지 말고, **감정의 방향(분노·피로·실망·희망)** 과 **세대별 차이, 논조 변화, 참여 계층** 등을 분석하라.
       - 주민들의 심리 상태가 정책 개선 요구와 어떤 관계를 가지는지까지 확장하여 설명하라.
       - 4~6줄 이상으로 작성하라.
       - 수치나 댓글 개수는 언급하지 말고, 오직 정성적 판단으로 기술하라.

    === 출력 형식 (JSON으로만 반환) ===
    {{
      "problem_summary": "(심층 분석 요약문)",
      "scarcity_insight": "(정성적 여론 분석 결과)"
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

        # ✅ 진단 시간 추가
        diagnosed_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "region": region_name,
            "top_topics": top_topic_str,
            "record_count": record_count,
            "scarcity_level": scarcity_level,
            "diagnosed_at": diagnosed_time,
            "result": result
        }

    except Exception as e:
        print(f"[analysis_diagnosis] ❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"GPT 요청 실패: {e}")
