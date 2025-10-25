# app/scripts/import_all_files.py

import os
import json
from datetime import datetime, UTC
from sqlalchemy.orm import Session
from app.utils.database import SessionLocal
from app.utils.models import RegionData, RagSummary

"""
import_all_files.py
-------------------
files 폴더 내의 정책 텍스트 및 벡터 파일을 DB에 삽입합니다.
(기존 import_policy_corpus.py 기능 통합 완료)
"""

# 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FILES_DIR = os.path.join(BASE_DIR, "files")

# DB 세션 생성
db: Session = SessionLocal()
print(f"[import_all_files] ✅ DB 연결 성공 ({FILES_DIR})")

# =========================================================
# 1️⃣ 정책 원문 파일 (policy_corpus.txt)
# =========================================================
corpus_path = os.path.join(FILES_DIR, "policy_corpus.txt")

if os.path.exists(corpus_path):
    added, skipped = 0, 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "-" not in line:
                continue

            # 예시 포맷: "서울-역세권 청년안심주택: 설명..."
            region_name, rest = line.split("-", 1)
            topic, summary = rest.split(":", 1) if ":" in rest else (rest, "")
            region_name, topic, summary = region_name.strip(), topic.strip(), summary.strip()

            region = db.query(RegionData).filter(RegionData.region_name == region_name).first()
            if not region:
                print(f"[import_all_files] ⚠️ 지역 '{region_name}' 없음 → 건너뜀")
                skipped += 1
                continue

            exists = db.query(RagSummary).filter(
                RagSummary.region_id == region.id,
                RagSummary.topic == topic
            ).first()
            if exists:
                skipped += 1
                continue

            new_row = RagSummary(
                region_id=region.id,
                topic=topic,
                summary=summary,
                created_at=datetime.now(UTC)
            )
            db.add(new_row)
            added += 1

    db.commit()
    print(f"[import_all_files] 📝 정책 원문 {added}건 추가, {skipped}건 스킵 완료")

else:
    print(f"[import_all_files] ⚠️ 파일 없음: {corpus_path}")

# =========================================================
# 2️⃣ 정책 벡터 파일 (policy_vectors.json)
# =========================================================
vectors_path = os.path.join(FILES_DIR, "policy_vectors.json")

if os.path.exists(vectors_path):
    with open(vectors_path, "r", encoding="utf-8") as f:
        vectors = json.load(f)

    if isinstance(vectors, list):
        print(f"[import_all_files] 🧩 벡터 파일 로드 완료 ({len(vectors)}개 정책)")
        added_vec, skipped_vec = 0, 0

        for item in vectors:
            policy_name = item.get("policy_name", "").strip()
            description = item.get("description", "").strip()
            region_name = item.get("region_name", "서울").strip()
            vector = item.get("vector", [])

            region = db.query(RegionData).filter(RegionData.region_name == region_name).first()
            if not region:
                print(f"[import_all_files] ⚠️ '{region_name}' 지역 없음 → 스킵")
                skipped_vec += 1
                continue

            exists = db.query(RagSummary).filter(
                RagSummary.region_id == region.id,
                RagSummary.topic == policy_name
            ).first()
            if exists:
                skipped_vec += 1
                continue

            summary = RagSummary(
                region_id=region.id,
                topic=policy_name,
                summary=description,
                embedding=json.dumps(vector),
                created_at=datetime.now(UTC)
            )
            db.add(summary)
            added_vec += 1

        db.commit()
        print(f"[import_all_files] ✅ 벡터 데이터 {added_vec}건 추가, {skipped_vec}건 스킵 완료")

    else:
        print("[import_all_files] ⚠️ 벡터 파일 형식이 list가 아닙니다.")
else:
    print(f"[import_all_files] ⚠️ 파일 없음: {vectors_path}")

# =========================================================
# 종료
# =========================================================
db.close()
print("[import_all_files] ✅ 모든 데이터가 DB에 반영되었습니다.")
