# app/utils/schemas.py
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional, List

class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class RagSummaryResponse(ConfiguredBaseModel):
    id: int
    region_id: int | None = None
    topic: str
    summary: str
    proposal_list: Optional[str] = None
    created_at: datetime

class RegionResponse(ConfiguredBaseModel):
    id: int
    region_name: str

    policy_avg_score: float
    transport_infra_policy_score: float
    labor_economy_policy_score: float
    healthcare_policy_score: float
    policy_efficiency_score: float
    housing_environment_policy_score: float

    sentiment_avg_score: float
    sentiment_transport_infra_score: float
    sentiment_labor_economy_score: float
    sentiment_healthcare_score: float
    sentiment_policy_efficiency_score: float
    sentiment_housing_environment_score: float

    gap_score: float
    updated_at: datetime

class RegionDetailResponse(RegionResponse):
    summaries: Optional[List[RagSummaryResponse]] = []
