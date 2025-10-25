import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

# ✅ 프로젝트 루트 기준 절대경로 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app/
ROOT_DIR = os.path.dirname(BASE_DIR)  # welling_backend/
DB_PATH = os.path.join(ROOT_DIR, "region_data.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

# ✅ 로그 출력
print(f"[database.py] SQLite 연결 경로: {DB_PATH}")

# SQLAlchemy 세션/엔진 설정
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI 의존성 주입용 세션"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_engine():
    """테이블 생성용 엔진 반환"""
    return engine


if __name__ == "__main__":
    from app.utils.models import Base
    print("[database.py] region_data.db 테이블 생성 중...")
    Base.metadata.create_all(bind=engine)
    print("[database.py] 테이블 생성 완료 ✅")
