from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.utils.models import RegionData, RagSummary
from app.utils.schemas import RegionResponse, RegionDetailResponse

router = APIRouter()

"""
region_router.py
지역 데이터 조회 관련 API 라우터
- /api/regions/ (전체 조회)
- /api/regions/{region_name}/ (상세 조회)
"""

@router.get("/regions/", response_model=list[RegionResponse])
def get_all_regions(db: Session = Depends(get_db)):
    regions = db.query(RegionData).all()
    return regions


@router.get("/regions/{region_name}/", response_model=RegionDetailResponse)
def get_region_detail(region_name: str, db: Session = Depends(get_db)):
    region = db.query(RegionData).filter(RegionData.region_name == region_name).first()
    if not region:
        raise HTTPException(status_code=404, detail="해당 지역을 찾을 수 없습니다.")
    summaries = db.query(RagSummary).filter(RagSummary.region_id == region.id).all()

    return {
        "region_name": region.region_name,
        "policy_score": region.policy_score,
        "sentiment_score": region.sentiment_score,
        "gap_score": region.gap_score,
        "summaries": summaries
    }


print("[region_router.py] 지역 데이터 라우터가 성공적으로 로드되었습니다.")
