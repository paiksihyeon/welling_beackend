from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

"""
이 파일은 요청(Request)과 응답(Response)에 사용되는 데이터 스키마를 정의합니다.
models.py에 새로 추가된 세부 심리지수와 proposal_list를 반영하였습니다.
"""

# ------------------------------------------------------
# 공통 베이스 클래스
# ------------------------------------------------------

class ConfiguredBaseModel(BaseModel):
    class Config:
        orm_mode = True


# ------------------------------------------------------
# RegionData 스키마
# ------------------------------------------------------

class RegionBase(ConfiguredBaseModel):
    region_name: str
    policy_score: float
    sentiment_score: float
    gap_score: float

    # ✅ 세부 심리지수 5개
    infra_sentiment: float
    housing_sentiment: float
    health_sentiment: float
    economy_sentiment: float
    policy_efficiency: float


class RegionCreate(RegionBase):
    """새로운 지역 데이터 등록 시 사용"""
    pass


class RegionResponse(RegionBase):
    """응답용: region_id 및 갱신 시간 포함"""
    id: int
    updated_at: datetime


# ------------------------------------------------------
# SentimentAnalysisLog 스키마
# ------------------------------------------------------

class SentimentLogBase(ConfiguredBaseModel):
    text: str
    sentiment_score: float
    model: Optional[str] = None


class SentimentLogCreate(SentimentLogBase):
    region_name: str


class SentimentLogResponse(SentimentLogBase):
    id: int
    region_id: int
    created_at: datetime


# ------------------------------------------------------
# RagSummary 스키마
# ------------------------------------------------------

class RagSummaryBase(ConfiguredBaseModel):
    topic: str
    summary: str
    # ✅ 정책 제안 리스트 추가
    proposal_list: Optional[str] = None


class RagSummaryCreate(RagSummaryBase):
    region_name: str


class RagSummaryResponse(RagSummaryBase):
    id: int
    region_id: int
    created_at: datetime


# ------------------------------------------------------
# API 응답 구조체 (리스트 및 조합형)
# ------------------------------------------------------

class RegionListResponse(ConfiguredBaseModel):
    regions: List[RegionResponse]


class RegionDetailResponse(ConfiguredBaseModel):
    region_name: str
    policy_score: float
    sentiment_score: float
    gap_score: float
    infra_sentiment: float
    housing_sentiment: float
    health_sentiment: float
    economy_sentiment: float
    policy_efficiency: float
    summaries: Optional[List[RagSummaryResponse]] = []


# ------------------------------------------------------
# 실행 확인용 출력문
# ------------------------------------------------------

print("[schemas.py] Pydantic 스키마가 성공적으로 로드되었습니다.")
