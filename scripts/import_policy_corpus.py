import os
from sqlalchemy.orm import Session
from app.utils.database import SessionLocal
from app.utils.models import RegionData, RagSummary

# 파일 경로
FILE_DIR = os.path.join(os.path.dirname(__file__), "..", "file")
CORPUS_PATH = os.path.join(FILE_DIR, "policy_corpus.txt")

def import_policy_corpus():
    db: Session = SessionLocal()
    added, skipped = 0, 0

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "-" not in line:
                continue

            # 예: "서울-역세권 청년안심주택: 설명..."
            region_name, rest = line.split("-", 1)
            topic, summary = rest.split(":", 1) if ":" in rest else (rest, "")
            region_name, topic, summary = region_name.strip(), topic.strip(), summary.strip()

            region = db.query(RegionData).filter(RegionData.region_name == region_name).first()
            if not region:
                print(f"[WARN] 지역 '{region_name}' 없음, 건너뜀")
                skipped += 1
                continue

            exists = db.query(RagSummary).filter(
                RagSummary.region_id == region.id,
                RagSummary.topic == topic
            ).first()

            if exists:
                skipped += 1
                continue

            new_row = RagSummary(region_id=region.id, topic=topic, summary=summary)
            db.add(new_row)
            added += 1

    db.commit()
    db.close()
    print(f"[import_policy_corpus] 완료: {added}건 추가, {skipped}건 스킵")

if __name__ == "__main__":
    import_policy_corpus()
