# app/services/vector_store_service.py

import json
import math
import numpy as np
import torch
from typing import List, Optional
from sqlalchemy.orm import Session
from transformers import ElectraTokenizer, ElectraModel

from app.utils.models import RagSummary, RegionData


# =========================================================
# 1. KoELECTRA 모델 로드 및 임베딩 유틸
# =========================================================

_tokenizer = None
_model = None

def _load_model():
    """
    KoELECTRA 모델 및 토크나이저를 lazy loading으로 불러옴.
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        _model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        _model.eval()
    return _tokenizer, _model


def embed_text_koelectra(text: str, max_length: int = 256) -> List[float]:
    """
    입력 텍스트를 KoELECTRA CLS 벡터로 임베딩하여 리스트(float)로 반환.
    """
    tokenizer, model = _load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS 토큰 벡터 추출 [batch, seq, hidden] -> [hidden]
    vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().tolist()
    return vec


def dumps_embedding(vec: List[float]) -> str:
    """
    벡터를 JSON 문자열로 직렬화.
    """
    return json.dumps(vec, ensure_ascii=False)


def loads_embedding(s: str) -> List[float]:
    """
    JSON 문자열을 벡터(List[float])로 역직렬화.
    """
    return json.loads(s)


# =========================================================
# 2. RAG 임베딩 관리 및 검색 서비스
# =========================================================

def ensure_embedding_for_row(db: Session, row: RagSummary, force: bool = False) -> bool:
    """
    RagSummary 행에 embedding이 없으면 생성해서 저장.
    force=True면 항상 재계산.
    반환: 새로 생성/갱신했는지 여부(bool)
    """
    if row.embedding and not force:
        return False
    vec = embed_text_koelectra(row.summary or "")
    row.embedding = dumps_embedding(vec)
    db.add(row)
    return True


def reindex_all_embeddings(db: Session, limit: Optional[int] = None, force: bool = False) -> int:
    """
    rag_summary 전체를 순회하며 임베딩 생성/갱신.
    limit 지정 시 상위 N개만 수행.
    반환: 생성/갱신한 행의 개수(int)
    """
    q = db.query(RagSummary).order_by(RagSummary.id.asc())
    if limit:
        q = q.limit(limit)
    rows = q.all()
    updated = 0
    for r in rows:
        if ensure_embedding_for_row(db, r, force=force):
            updated += 1
    db.commit()
    return updated


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    코사인 유사도 계산 함수.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def search_relevant_policies(
    db: Session,
    query_text: str,
    top_k: int = 3,
    region_name: Optional[str] = None,
) -> List[RagSummary]:
    """
    질의 텍스트를 KoELECTRA로 임베딩 → DB 내 정책 임베딩과 코사인 유사도 계산 → 상위 K개 반환.
    """
    # 질의 임베딩 생성
    q_vec = np.array(embed_text_koelectra(query_text), dtype=np.float32)

    # 후보 집합 조회 (region_name 지정 시 해당 지역만)
    if region_name:
        region = db.query(RegionData).filter(RegionData.region_name == region_name).first()
        if not region:
            return []
        candidates = db.query(RagSummary).filter(RagSummary.region_id == region.id).all()
    else:
        candidates = db.query(RagSummary).all()

    scored = []
    for c in candidates:
        if not c.embedding:
            # 임베딩이 없으면 생성 후 저장
            ensure_embedding_for_row(db, c, force=False)
            db.commit()
        if not c.embedding:
            continue

        emb = np.array(loads_embedding(c.embedding), dtype=np.float32)
        sim = cosine_similarity(q_vec, emb)
        scored.append((c, sim))

    # 유사도 기준 내림차순 정렬 후 상위 top_k 반환
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:top_k]]
