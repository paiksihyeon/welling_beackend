# app/routers/region_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.utils.models import RegionData, RagSummary
from app.utils.schemas import RegionResponse, RegionDetailResponse

router = APIRouter()

@router.get("/regions/", response_model=list[RegionResponse])
def get_all_regions(db: Session = Depends(get_db)):
    return db.query(RegionData).all()

@router.get("/regions/{region_name}/", response_model=RegionDetailResponse)
def get_region_detail(region_name: str, db: Session = Depends(get_db)):
    region = db.query(RegionData).filter(RegionData.region_name == region_name).first()
    if not region:
        raise HTTPException(status_code=404, detail="해당 지역을 찾을 수 없습니다.")
    summaries = db.query(RagSummary).filter(RagSummary.region_id == region.id).all()

    return {
        "id": region.id,
        "region_name": region.region_name,

        "policy_avg_score": region.policy_avg_score,
        "transport_infra_policy_score": region.transport_infra_policy_score,
        "labor_economy_policy_score": region.labor_economy_policy_score,
        "healthcare_policy_score": region.healthcare_policy_score,
        "policy_efficiency_score": region.policy_efficiency_score,
        "housing_environment_policy_score": region.housing_environment_policy_score,

        "sentiment_avg_score": region.sentiment_avg_score,
        "sentiment_transport_infra_score": region.sentiment_transport_infra_score,
        "sentiment_labor_economy_score": region.sentiment_labor_economy_score,
        "sentiment_healthcare_score": region.sentiment_healthcare_score,
        "sentiment_policy_efficiency_score": region.sentiment_policy_efficiency_score,
        "sentiment_housing_environment_score": region.sentiment_housing_environment_score,

        "gap_score": region.gap_score,
        "updated_at": region.updated_at,
        "summaries": summaries,
    }
