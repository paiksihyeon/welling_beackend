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


@router.get("/regions/{region_name}/top-gaps/")
def get_top_gap_topics(region_name: str, db: Session = Depends(get_db)):
    """
    특정 지역의 주제별 gap을 계산하여 상위 3개 주제 반환
    """
    region = db.query(RegionData).filter(RegionData.region_name == region_name).first()
    if not region:
        raise HTTPException(status_code=404, detail="해당 지역을 찾을 수 없습니다.")

    # 주제별 gap 계산
    topics = [
        {
            "topic": "교통인프라",
            "topic_en": "transport_infra",
            "policy_score": region.transport_infra_policy_score,
            "sentiment_score": region.sentiment_transport_infra_score,
            "gap": abs(region.transport_infra_policy_score - region.sentiment_transport_infra_score)
        },
        {
            "topic": "노동경제",
            "topic_en": "labor_economy",
            "policy_score": region.labor_economy_policy_score,
            "sentiment_score": region.sentiment_labor_economy_score,
            "gap": abs(region.labor_economy_policy_score - region.sentiment_labor_economy_score)
        },
        {
            "topic": "의료",
            "topic_en": "healthcare",
            "policy_score": region.healthcare_policy_score,
            "sentiment_score": region.sentiment_healthcare_score,
            "gap": abs(region.healthcare_policy_score - region.sentiment_healthcare_score)
        },
        {
            "topic": "정책효율성",
            "topic_en": "policy_efficiency",
            "policy_score": region.policy_efficiency_score,
            "sentiment_score": region.sentiment_policy_efficiency_score,
            "gap": abs(region.policy_efficiency_score - region.sentiment_policy_efficiency_score)
        },
        {
            "topic": "주거환경",
            "topic_en": "housing_environment",
            "policy_score": region.housing_environment_policy_score,
            "sentiment_score": region.sentiment_housing_environment_score,
            "gap": abs(region.housing_environment_policy_score - region.sentiment_housing_environment_score)
        }
    ]

    # gap 기준 내림차순 정렬 후 상위 3개 선택
    top_topics = sorted(topics, key=lambda x: x["gap"], reverse=True)[:3]

    return {
        "region_name": region.region_name,
        "top_gap_topics": top_topics
    }
