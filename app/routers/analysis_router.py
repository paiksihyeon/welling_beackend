from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.utils.database import get_db
from app.services.sentiment_service import save_sentiment_result
from app.services.rag_service import save_rag_summary
from app.services.gap_calculator import update_all_gap_scores
from datetime import datetime
import random

router = APIRouter(tags=["Analysis"])

# 요청 body용 Pydantic 모델
class AnalysisRequest(BaseModel):
    region_name: str
    topic: str
    summary: str


@router.post("/run_analysis/")
def run_analysis(request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    특정 지역(region_name)에 대해 AI 분석 파이프라인 실행
    1. 감정분석 점수 저장
    2. 정책 요약(RAG) 저장
    3. Gap Score 재계산
    """
    try:
        # 감정 분석 점수 계산 (테스트용 랜덤값)
        sentiment_score = round(random.uniform(0, 100), 2)

        save_sentiment_result(
            db=db,
            region_name=request.region_name,
            text=request.summary,
            score=sentiment_score,
            model="kobert"
        )

        # RAG 요약 저장
        save_rag_summary(
            db=db,
            region_name=request.region_name,
            topic=request.topic,
            summary=request.summary
        )

        # Gap Score 전체 업데이트
        update_all_gap_scores(db)

        print(f"[analysis_router] {request.region_name} 지역의 분석 완료")
        return {
            "status": "success",
            "region": request.region_name,
            "sentiment_score": sentiment_score,
            "topic": request.topic,
            "summary": request.summary,
            "updated_at": datetime.utcnow()
        }

    except Exception as e:
        db.rollback()
        print(f"[analysis_router] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}


# 실행 확인용
if __name__ == "__main__":
    print("[analysis_router.py] 모듈이 정상적으로 로드되었습니다.")
