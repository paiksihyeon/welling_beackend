from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from app.utils.database import get_db
from app.utils.models import RegionData
from app.services.gap_calculator import update_all_gap_scores

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/region-summary/")
def get_region_summary(db: Session = Depends(get_db)):
    try:
        regions = db.query(RegionData).all()
        result = [
            {
                "region_name": region.region_name,
                "policy_score": region.policy_avg_score,
                "sentiment_score": region.sentiment_avg_score,
                "gap_score": region.gap_score,
                "updated_at": region.updated_at,
            }
            for region in regions
        ]
        print("[analytics_router] 지역 요약 데이터 반환 완료")
        return {"count": len(result), "data": result}

    except Exception as e:
        print(f"[analytics_router] 오류 발생: {e}")
        return {"error": str(e)}


@router.post("/update-gap/")
def update_gap_scores(db: Session = Depends(get_db)):
    """
    Gap Score를 자동 계산하여 DB에 업데이트하는 엔드포인트
    """
    try:
        update_all_gap_scores(db)
        print("[analytics_router] Gap Score 자동 업데이트 완료")
        return {"status": "success"}

    except Exception as e:
        print(f"[analytics_router] Gap Score 업데이트 중 오류: {e}")
        return {"status": "error", "message": str(e)}
