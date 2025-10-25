from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.utils.database import get_db
from app.utils.models import RegionData, RagSummary
from datetime import datetime
import openai
import os
from app.services.rag_service import recommend_policies, generate_rag_insight


"""
rag_router.py
ChatGPT API를 통해 정책 요약(summary)과 정책 제안(proposal_list)을 생성하고
DB에 저장하는 라우터입니다.
"""

# ------------------------------------------------------
# Router 설정
# ------------------------------------------------------
router = APIRouter(prefix="/rag", tags=["RAG"])

# OpenAI API Key 로드
openai.api_key = os.getenv("OPENAI_API_KEY")

class RagRequest(BaseModel):
    region_name: str
    topic: str
    text: str


class RAGRecommendRequest(BaseModel):
    region_name: str
    topic: str

class RAGInsightRequest(BaseModel):
    region_name: str
    topic: str
# ------------------------------------------------------
# ChatGPT 기반 RAG 생성 함수
# ------------------------------------------------------
def generate_rag_summary_from_gpt(region_name: str, topic: str, text: str):
    """
    ChatGPT API를 호출하여 요약(summary)과 정책 제안(proposal_list)을 생성
    """
    try:
        prompt = f"""
        아래는 '{region_name}' 지역의 '{topic}' 관련 정책 내용입니다.
        내용을 간결하게 요약한 후, 정책 개선을 위한 2~3가지 구체적인 제안사항을 제시하세요.

        내용:
        {text}

        출력 형식:
        1. Summary: (요약문)
        2. Proposals: (쉼표로 구분된 정책 제안 리스트)
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 복지정책 분석 전문가입니다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        gpt_output = response.choices[0].message.content.strip()

        # 결과 파싱
        summary_text = ""
        proposals = ""
        for line in gpt_output.split("\n"):
            if line.lower().startswith("1.") or "summary" in line.lower():
                summary_text += line.replace("1.", "").replace("Summary:", "").strip()
            elif line.lower().startswith("2.") or "proposals" in line.lower():
                proposals += line.replace("2.", "").replace("Proposals:", "").strip()

        if not summary_text:
            summary_text = gpt_output[:300]  # fallback

        if not proposals:
            proposals = "정책 제안 없음"

        return summary_text, proposals

    except Exception as e:
        raise RuntimeError(f"ChatGPT 요청 실패: {e}")


# ------------------------------------------------------
# RAG 요약 생성 및 DB 저장 엔드포인트
# ------------------------------------------------------
@router.post("/generate/")
def generate_rag_summary(request: RagRequest, db: Session = Depends(get_db)):
    """
    ChatGPT를 호출하여 RAG 요약(summary)과 제안(proposal_list)을 생성 후 DB에 저장
    """
    try:
        # 1️⃣ ChatGPT 요약 생성
        summary, proposals = generate_rag_summary_from_gpt(
            region_name=request.region_name,
            topic=request.topic,
            text=request.text
        )

        # 2️⃣ RegionData 조회
        region = db.query(RegionData).filter_by(region_name=request.region_name).first()
        if not region:
            raise ValueError(f"{request.region_name} 지역 데이터가 존재하지 않습니다.")

        # 3️⃣ RagSummary에 저장
        rag_entry = RagSummary(
            region_id=region.id,
            topic=request.topic,
            summary=summary,
            proposal_list=proposals,
            created_at=datetime.utcnow(),
        )
        db.add(rag_entry)
        db.commit()

        print(f"[rag_router] {request.region_name} '{request.topic}' RAG 생성 완료")
        print(f"  ▶ Summary: {summary}")
        print(f"  ▶ Proposals: {proposals}")

        return {
            "status": "success",
            "region": request.region_name,
            "topic": request.topic,
            "summary": summary,
            "proposal_list": proposals,
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        print(f"[rag_router] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/recommend/")
def rag_recommend(request: RAGRecommendRequest, db: Session = Depends(get_db)):
    """
    RAG 정책 추천 API
    - 주제(topic)에 대한 유사 정책 검색 및 ChatGPT 요약 추천
    """
    try:
        result = recommend_policies(
            region_name=request.region_name,
            topic=request.topic,
            db=db,
            top_k=5
        )
        return {
            "status": "success",
            "region": request.region_name,
            "topic": request.topic,
            "data": result
        }

    except Exception as e:
        print(f"[RAG Router] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/insight/")
def rag_insight(request: RAGInsightRequest, db: Session = Depends(get_db)):
    """
    RAG 종합 인사이트 API
    - 시민 불만 요약 + 유사 정책 매칭 + 최종 종합 요약
    """
    try:
        result = generate_rag_insight(
            region_name=request.region_name,
            topic=request.topic,
            db=db,
        )
        return {
            "status": "success",
            "region": request.region_name,
            "topic": request.topic,
            "data": result
        }

    except Exception as e:
        print(f"[RAG Insight API] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}
# ------------------------------------------------------
# 모듈 실행 확인
# ------------------------------------------------------
if __name__ == "__main__":
    print("[rag_router.py] ChatGPT RAG 연동 라우터 로드 완료.")
