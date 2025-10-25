# app/utils/models.py
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.utils.database import Base

class RegionData(Base):
    __tablename__ = "region_data"

    id = Column(Integer, primary_key=True, index=True)
    region_name = Column(String, unique=True, nullable=False)  # CSV: region

    # 정책 평균 및 세부 정책 점수
    policy_avg_score = Column(Float, nullable=False, default=0.0)
    transport_infra_policy_score = Column(Float, nullable=False, default=0.0)
    labor_economy_policy_score = Column(Float, nullable=False, default=0.0)
    healthcare_policy_score = Column(Float, nullable=False, default=0.0)
    policy_efficiency_score = Column(Float, nullable=False, default=0.0)
    housing_environment_policy_score = Column(Float, nullable=False, default=0.0)

    # 여론 평균 및 세부 여론 점수
    sentiment_avg_score = Column(Float, nullable=False, default=0.0)
    sentiment_transport_infra_score = Column(Float, nullable=False, default=0.0)
    sentiment_labor_economy_score = Column(Float, nullable=False, default=0.0)
    sentiment_healthcare_score = Column(Float, nullable=False, default=0.0)
    sentiment_policy_efficiency_score = Column(Float, nullable=False, default=0.0)
    sentiment_housing_environment_score = Column(Float, nullable=False, default=0.0)

    # 괴리 점수
    gap_score = Column(Float, nullable=False, default=0.0)

    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # 아래는 기존과 동일하게 두세요. 예시로 RagSummary 관계가 있다면 유지.
    summaries = relationship("RagSummary", backref="region", primaryjoin="RegionData.id==RagSummary.region_id")


class SentimentAnalysisLog(Base):
    __tablename__ = "sentiment_analysis_log"
    id = Column(Integer, primary_key=True, index=True)
    region = Column(String, nullable=False)      # 지역명 ("서울", "강원" 등)
    topic = Column(String, nullable=False)       # 주제 ("주거환경", "노동경제" 등)
    text = Column(Text, nullable=False)          # 시민 의견
    label = Column(Integer, nullable=False)      # +1: 긍정, -1: 부정

class RagSummary(Base):
    __tablename__ = "rag_summary"
    id = Column(Integer, primary_key=True, index=True)
    region_id = Column(Integer, ForeignKey("region_data.id"), nullable=True)
    topic = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    proposal_list = Column(Text, nullable=True)
    embedding = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

class RagPolicy(Base):
    """RAG 정책 테이블"""
    __tablename__ = "rag_policy"

    id = Column(Integer, primary_key=True, index=True)
    region = Column(String, nullable=False)
    policy = Column(String, nullable=False)