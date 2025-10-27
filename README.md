# Welling Backend API

AI 기반 복지정책 불균형 지도 Welling 백엔드 API

## 프로젝트 개요

Welling Backend는 지역별 정책 점수와 여론 점수를 분석하여 정책 불균형을 시각화하고, AI 기반 정책 개선 방향을 제안하는 FastAPI 기반 백엔드 서버입니다.

## 주요 기능

- 지역별 정책/여론 점수 관리
- Gap Score(정책-여론 괴리도) 자동 계산
- 여론 분석 (Sentiment Analysis)
- RAG 기반 정책 추천
- Cross-Region 정책 개선 액션 제안

## 기술 스택

- **Framework**: FastAPI
- **Database**: SQLite (SQLAlchemy ORM)
- **AI/ML**: OpenAI API (GPT-4)
- **Vector Search**: Custom JSON-based vector similarity
- **Language**: Python 3.10+

---

## 최근 수정 사항 (2025-10-27)

### 문제 상황

프론트엔드에서 다음과 같은 오류 발생:
```
'RegionData' object has no attribute 'policy_score'
```

### 원인 분석

데이터베이스 모델은 세분화된 점수 체계를 사용하지만, 여러 서비스 로직에서는 이전 버전의 `policy_score`, `sentiment_score` 필드를 참조하고 있었음.

**실제 DB 모델 필드:**
- `policy_avg_score` (정책 평균 점수)
- `transport_infra_policy_score` (교통 인프라 정책 점수)
- `labor_economy_policy_score` (노동 경제 정책 점수)
- `healthcare_policy_score` (의료 정책 점수)
- `policy_efficiency_score` (정책 효율성 점수)
- `housing_environment_policy_score` (주거 환경 정책 점수)
- `sentiment_avg_score` (여론 평균 점수)
- `sentiment_transport_infra_score` (교통 인프라 여론 점수)
- `sentiment_labor_economy_score` (노동 경제 여론 점수)
- `sentiment_healthcare_score` (의료 여론 점수)
- `sentiment_policy_efficiency_score` (정책 효율성 여론 점수)
- `sentiment_housing_environment_score` (주거 환경 여론 점수)
- `gap_score` (정책-여론 괴리 점수)

### 수정된 파일 목록

#### 1. [app/routers/analytics_router.py](app/routers/analytics_router.py)
```python
# Before
"policy_score": region.policy_score,
"sentiment_score": region.sentiment_score,

# After
"policy_score": region.policy_avg_score,
"sentiment_score": region.sentiment_avg_score,
```

#### 2. [app/services/gap_calculator.py](app/services/gap_calculator.py)
```python
# Before
region.gap_score = calculate_gap(region.policy_score, region.sentiment_score)

# After
region.gap_score = calculate_gap(region.policy_avg_score, region.sentiment_avg_score)
```

#### 3. [app/services/sentiment_service.py](app/services/sentiment_service.py)
```python
# Before
region = RegionData(
    region_name=region_name,
    policy_score=0.0,
    sentiment_score=0.0,
    gap_score=0.0,
)
region.sentiment_score = round(score, 2)
region.gap_score = calculate_gap(region.policy_score, region.sentiment_score)

# After
region = RegionData(
    region_name=region_name,
    policy_avg_score=0.0,
    sentiment_avg_score=0.0,
    gap_score=0.0,
)
region.sentiment_avg_score = round(score, 2)
region.gap_score = calculate_gap(region.policy_avg_score, region.sentiment_avg_score)
```

#### 4. [app/services/rag_service.py](app/services/rag_service.py)
```python
# Before
region = RegionData(
    region_name=region_name,
    policy_score=0.0,
    sentiment_score=0.0,
    gap_score=0.0,
)

# After
region = RegionData(
    region_name=region_name,
    policy_avg_score=0.0,
    sentiment_avg_score=0.0,
    gap_score=0.0,
)
```

#### 5. [main.py](main.py) - 테스트 엔드포인트
```python
# Before
region = RegionResponse(
    id=1,
    region_name="서울",
    policy_score=82.5,
    sentiment_score=32.1,
    gap_score=50.4,
    updated_at=datetime.utcnow(),
)

# After
region = RegionResponse(
    id=1,
    region_name="서울",
    policy_avg_score=82.5,
    transport_infra_policy_score=75.0,
    labor_economy_policy_score=80.0,
    healthcare_policy_score=85.0,
    policy_efficiency_score=78.0,
    housing_environment_policy_score=88.0,
    sentiment_avg_score=32.1,
    sentiment_transport_infra_score=30.0,
    sentiment_labor_economy_score=28.0,
    sentiment_healthcare_score=35.0,
    sentiment_policy_efficiency_score=33.0,
    sentiment_housing_environment_score=34.5,
    gap_score=50.4,
    updated_at=datetime.utcnow(),
)
```

### 수정 명령어 (재현 방법)

```bash
# policy_score -> policy_avg_score 변경
sed -i 's/\.policy_score/.policy_avg_score/g' app/routers/analytics_router.py app/services/gap_calculator.py app/services/sentiment_service.py app/services/rag_service.py main.py

# sentiment_score -> sentiment_avg_score 변경
sed -i 's/\.sentiment_score/.sentiment_avg_score/g' app/routers/analytics_router.py app/services/gap_calculator.py app/services/sentiment_service.py

# policy_score= -> policy_avg_score= 변경 (할당문)
sed -i 's/policy_score=/policy_avg_score=/g' app/services/sentiment_service.py app/services/rag_service.py

# sentiment_score= -> sentiment_avg_score= 변경 (할당문)
sed -i 's/sentiment_score=/sentiment_avg_score=/g' app/services/sentiment_service.py app/services/rag_service.py
```

---

## 설치 및 실행 가이드

### ① GitHub에서 백엔드 클론
```bash
git clone https://github.com/your-repo/welling_backend.git
cd welling_backend
```

### ② 가상환경 + 의존성 설치
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### ③ .env 생성 (API Key 등록)
```bash
# .env 파일 생성
OPENAI_API_KEY=your_openai_api_key_here
```

### ④ DB 초기화 (완전 초기화)
```bash
python -c "import os; from app.utils.database import engine, DB_PATH; from app.utils.models import Base; os.path.exists(DB_PATH) and os.remove(DB_PATH); Base.metadata.create_all(bind=engine); print('✅ DB 초기화 완료')"
```

### ⑤ 초기 데이터 삽입
```bash
python -c "from app.utils.init_data import insert_real_dataset; from app.utils.init_sentiment_data import insert_sentiment_dataset; from app.utils.init_rag_policy import insert_rag_policy_data; insert_real_dataset(); insert_sentiment_dataset(); insert_rag_policy_data(); print('✅ 모든 데이터 삽입 완료')"
```

### ⑥ 서버 실행
```bash
uvicorn main:app --reload
```

### ⑦ Swagger에서 API 테스트
브라우저에서 접속:
```
http://127.0.0.1:8000/docs
```

### ⑧ 프론트엔드와 연동 테스트
주요 API 엔드포인트:
- `GET /api/regions/` - 전체 지역 데이터
- `GET /api/regions/{region_name}/` - 특정 지역 상세 정보
- `GET /api/analysis/diagnosis/{region}` - 지역별 문제 진단
- `GET /api/rag/action/{region}` - 정책 개선 제안
- `GET /api/analytics/region-summary/` - 지역 요약 통계

---

## 원스텝 초기화 (DB 삭제 + 데이터 삽입)

```bash
python -c "import os; from app.utils.database import engine, DB_PATH; from app.utils.models import Base; from app.utils.init_data import insert_real_dataset; from app.utils.init_sentiment_data import insert_sentiment_dataset; from app.utils.init_rag_policy import insert_rag_policy_data; os.path.exists(DB_PATH) and os.remove(DB_PATH); Base.metadata.create_all(bind=engine); print('✅ 테이블 생성'); insert_real_dataset(); insert_sentiment_dataset(); insert_rag_policy_data(); print('🎉 완료!')"
```

---

## 데이터베이스 스키마

### RegionData 테이블
| 필드명 | 타입 | 설명 |
|--------|------|------|
| id | Integer | Primary Key |
| region_name | String | 지역명 (unique) |
| policy_avg_score | Float | 정책 평균 점수 |
| transport_infra_policy_score | Float | 교통 인프라 정책 점수 |
| labor_economy_policy_score | Float | 노동 경제 정책 점수 |
| healthcare_policy_score | Float | 의료 정책 점수 |
| policy_efficiency_score | Float | 정책 효율성 점수 |
| housing_environment_policy_score | Float | 주거 환경 정책 점수 |
| sentiment_avg_score | Float | 여론 평균 점수 |
| sentiment_transport_infra_score | Float | 교통 인프라 여론 점수 |
| sentiment_labor_economy_score | Float | 노동 경제 여론 점수 |
| sentiment_healthcare_score | Float | 의료 여론 점수 |
| sentiment_policy_efficiency_score | Float | 정책 효율성 여론 점수 |
| sentiment_housing_environment_score | Float | 주거 환경 여론 점수 |
| gap_score | Float | 정책-여론 괴리 점수 |
| updated_at | DateTime | 최종 업데이트 시각 |

### SentimentAnalysisLog 테이블
| 필드명 | 타입 | 설명 |
|--------|------|------|
| id | Integer | Primary Key |
| region | String | 지역명 |
| topic | String | 주제 |
| text | Text | 시민 의견 |
| label | Integer | 감정 레이블 (+1: 긍정, -1: 부정) |

### RagSummary 테이블
| 필드명 | 타입 | 설명 |
|--------|------|------|
| id | Integer | Primary Key |
| region_id | Integer | RegionData FK (nullable) |
| topic | String | 주제 |
| summary | Text | 요약 내용 |
| proposal_list | Text | 제안 목록 |
| embedding | Text | 벡터 임베딩 (JSON) |
| created_at | DateTime | 생성 시각 |

### RagPolicy 테이블
| 필드명 | 타입 | 설명 |
|--------|------|------|
| id | Integer | Primary Key |
| region | String | 지역명 |
| policy | String | 정책 내용 |

---

## API 엔드포인트

### Region 관련
- `GET /api/regions/` - 전체 지역 목록 조회
- `GET /api/regions/{region_name}/` - 특정 지역 상세 정보
- `GET /api/regions/{region_name}/top-gaps/` - 특정 지역의 주제별 gap 상위 3개 조회 ⭐ NEW

### Analysis 관련
- `GET /api/analysis/diagnosis/{region}` - 지역별 여론 기반 문제 진단 (AI)

### RAG 관련
- `GET /api/rag/action/{region}` - 지역별 정책 개선 방향 제안 (Cross-Region RAG)

### Analytics 관련
- `GET /api/analytics/region-summary/` - 전체 지역 요약 통계
- `POST /api/analytics/update-gap/` - Gap Score 일괄 업데이트

### Health Check
- `GET /api/health/` - 서버 상태 확인

---

## API 사용 예시

### 특정 지역의 상위 3개 Gap 주제 조회

**요청**:
```bash
GET /api/regions/서울/top-gaps/
```

**응답 예시**:
```json
{
  "region_name": "서울",
  "top_gap_topics": [
    {
      "topic": "주거환경",
      "topic_en": "housing_environment",
      "policy_score": 88.0,
      "sentiment_score": 34.5,
      "gap": 53.5
    },
    {
      "topic": "노동경제",
      "topic_en": "labor_economy",
      "policy_score": 80.0,
      "sentiment_score": 28.0,
      "gap": 52.0
    },
    {
      "topic": "교통인프라",
      "topic_en": "transport_infra",
      "policy_score": 75.0,
      "sentiment_score": 30.0,
      "gap": 45.0
    }
  ]
}
```

**설명**:
- 각 주제별(교통인프라, 노동경제, 의료, 정책효율성, 주거환경)로 정책 점수와 여론 점수의 차이(gap)를 계산
- gap이 큰 순서대로 정렬하여 상위 3개 주제 반환
- 프론트엔드에서 해당 지역의 가장 시급한 정책 개선 영역을 시각화할 때 사용

---

## 트러블슈팅

### 1. 'RegionData' object has no attribute 'policy_score'
**원인**: 모델 스키마 불일치
**해결**: 위의 수정 명령어 실행 또는 파일 수정

### 2. DB 파일이 없다는 오류
**해결**:
```bash
python -m app.utils.database
```

### 3. CSV 파일을 찾을 수 없음
**확인**: `app/files/` 디렉토리에 다음 파일들이 있는지 확인
- `Welling_Master_dataset.csv`
- `sentiment_dataset.csv`
- `Rag_Policy_dataset.csv`

### 4. OpenAI API 오류
**확인**: `.env` 파일에 `OPENAI_API_KEY`가 올바르게 설정되었는지 확인

---

## 개발 환경

- Python: 3.10+
- FastAPI: 최신 버전
- SQLAlchemy: 2.0+
- OpenAI: 최신 SDK
- Uvicorn: ASGI 서버

---
