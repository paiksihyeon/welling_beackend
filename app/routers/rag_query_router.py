# app/routers/rag_query_router.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session
from datetime import datetime
import os

from app.utils.database import get_db
from app.utils.models import RagSummary
from app.services.vector_store_service import reindex_all_embeddings, search_relevant_policies

# OpenAI (chat) 호출
from openai import OpenAI

router = APIRouter(prefix="/rag", tags=["RAG-Query"])

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

class ReindexRequest(BaseModel):
    limit: Optional[int] = None
    force: bool = False

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    region_name: Optional[str] = None

class RetrievedItem(BaseModel):
    id: int
    region_id: int
    topic: Optional[str] = None
    summary: Optional[str] = None
    proposal_list: Optional[str] = None
    created_at: Optional[datetime] = None
    score: Optional[float] = None  # 클라이언트 표시에 유용 (옵션)

@router.post("/reindex-embeddings")
def reindex_embeddings(req: ReindexRequest, db: Session = Depends(get_db)):
    """
    rag_summary의 embedding을 일괄 생성/갱신
    """
    updated = reindex_all_embeddings(db, limit=req.limit, force=req.force)
    return {"status": "success", "updated": updated, "timestamp": datetime.utcnow()}

@router.post("/query")
def rag_query(req: QueryRequest, db: Session = Depends(get_db)):
    """
    KoELECTRA로 질의 임베딩 → DB에서 유사 요약 상위 K개 검색 → ChatGPT로 최종 답변 생성
    """
    if not _openai_client:
        return {"status": "error", "message": "OPENAI_API_KEY가 설정되지 않았습니다."}

    # 검색
    retrieved = search_relevant_policies(
        db=db, query_text=req.query, top_k=req.top_k, region_name=req.region_name
    )

    if not retrieved:
        return {"status": "success", "answer": "관련 정책을 찾지 못했습니다.", "contexts": []}

    # 컨텍스트 구성
    contexts = []
    for r in retrieved:
        contexts.append(
            {
                "id": r.id,
                "region_id": r.region_id,
                "topic": r.topic,
                "summary": r.summary,
                "proposal_list": r.proposal_list,
                "created_at": r.created_at,
            }
        )
    context_text = "\n\n".join(
        [f"[{i+1}] Topic: {c['topic']}\nSummary: {c['summary']}\nProposals: {c.get('proposal_list') or ''}"
         for i, c in enumerate(contexts)]
    )

    prompt = (
        "당신은 대한민국 정책 분석 전문가입니다. 아래 컨텍스트를 바탕으로 사용자 질문에 근거 있는 답변을 한국어로 명확하게 작성하세요.\n\n"
        f"[컨텍스트]\n{context_text}\n\n"
        f"[질문]\n{req.query}\n\n"
        "가능하면 컨텍스트에서 근거 문장을 간단히 인용해 주세요."
    )

    resp = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 신뢰할 수 있는 정책 분석가입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )

    answer = resp.choices[0].message.content.strip()
    return {
        "status": "success",
        "answer": answer,
        "contexts": contexts,
        "timestamp": datetime.utcnow(),
    }
