from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

"""
이 파일은 데이터베이스 테이블 구조(ORM 모델)를 정의합니다.
정책 제안 리스트(proposal_list) 칼럼이 RAG 테이블에 추가되었습니다.
"""

# 1. 지역별 데이터 테이블 (핵심)
class RegionData(Base):
    __tablename__ = "region_data"

    id = Column(Integer, primary_key=True, index=True)
    region_name = Column(String, unique=True, nullable=False)

    # 정책/심리/불균형 지표
    policy_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    gap_score = Column(Float, default=0.0)

    # ✅ 세부 심리지수 5개
    infra_sentiment = Column(Float, default=0.0)      # 교통/인프라
    housing_sentiment = Column(Float, default=0.0)    # 주거/환경
    health_sentiment = Column(Float, default=0.0)     # 의료/보건
    economy_sentiment = Column(Float, default=0.0)    # 노동/경제
    policy_efficiency = Column(Float, default=0.0)    # 정책효능감

    updated_at = Column(DateTime, default=datetime.utcnow)

    # 관계 정의
    rag_summaries = relationship("RagSummary", back_populates="region")
    sentiment_logs = relationship("SentimentAnalysisLog", back_populates="region")

    def __repr__(self):
        return (
            f"<RegionData(id={self.id}, region='{self.region_name}', "
            f"policy={self.policy_score}, sentiment={self.sentiment_score}, gap={self.gap_score})>"
        )


# 2. 감정 분석 로그 테이블
class SentimentAnalysisLog(Base):
    __tablename__ = "sentiment_log"

    id = Column(Integer, primary_key=True, index=True)
    region_id = Column(Integer, ForeignKey("region_data.id"))
    text = Column(String)
    sentiment_score = Column(Float)
    model = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    region = relationship("RegionData", back_populates="sentiment_logs")

    def __repr__(self):
        return f"<SentimentLog(region_id={self.region_id}, score={self.sentiment_score}, model={self.model})>"


# 3. 정책 요약 (RAG 결과) 테이블
class RagSummary(Base):
    __tablename__ = "rag_summary"

    id = Column(Integer, primary_key=True, index=True)
    region_id = Column(Integer, ForeignKey("region_data.id"))
    topic = Column(String)
    summary = Column(String)

    # ✅ 정책 제안 리스트 (AI 모델이 생성한 정책 개선안 저장)
    proposal_list = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    region = relationship("RegionData", back_populates="rag_summaries")

    def __repr__(self):
        return (
            f"<RagSummary(region_id={self.region_id}, topic='{self.topic}', "
            f"proposal_list='{self.proposal_list}')>"
        )


# 실행 확인용 (이 파일이 import될 때 한 번만 실행됨)
print("[models.py] 모델 클래스가 성공적으로 로드되었습니다.")
