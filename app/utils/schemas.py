from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

"""
이 파일은 요청(Request)과 응답(Response)에 사용되는 데이터 스키마를 정의합니다.
테이블 구조가 바뀌더라도, API 입출력 형식을 쉽게 조정할 수 있도록 분리되어 있습니다.
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
    summaries: Optional[List[RagSummaryResponse]] = []


# ------------------------------------------------------
# 실행 확인용 출력문
# ------------------------------------------------------

print("[schemas.py] Pydantic 스키마가 성공적으로 로드되었습니다.")
