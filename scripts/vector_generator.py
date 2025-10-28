"""
Welling Vector Generator (E5 기반)
-----------------------------------
✅ 역할:
- 정책 문서 → policy_vectors.json 생성
- 지역별 여론 데이터 → {region}_vectors.json 생성
- 기존 app/files 폴더 구조에 맞춤형으로 동작

✅ 요구사항:
- app/files/policy_corpus.txt   → 정책 원문 텍스트 파일
- app/files/sentiment_dataset.csv → 지역별 여론 텍스트 (선택사항)
- app/files 폴더에 17개 시도명 데이터셋 존재
"""

import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --------------------------
# 기본 설정
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(BASE_DIR, "app", "files")
MODEL_NAME = "intfloat/multilingual-e5-base"

# SentenceTransformer 모델 로드
model = SentenceTransformer(MODEL_NAME)


# --------------------------
# 유틸 함수
# --------------------------
def normalize(vec: np.ndarray):
    """L2 정규화"""
    return vec / (np.linalg.norm(vec) + 1e-12)


def chunk_text(text, max_tokens=350):
    """긴 문장을 일정 길이로 분할"""
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])


def embed_text(text: str, prefix="passage: "):
    """문서 또는 문장 임베딩"""
    if not text or not text.strip():
        return None
    chunks = list(chunk_text(text))
    inputs = [prefix + c for c in chunks]
    embs = model.encode(inputs, normalize_embeddings=True)
    vec = np.mean(embs, axis=0)
    return normalize(vec).tolist()


# --------------------------
# 1️⃣ 정책 문서 벡터화
# --------------------------
def generate_policy_vectors():
    input_path = os.path.join(FILES_DIR, "policy_corpus.txt")
    output_path = os.path.join(FILES_DIR, "policy_vectors.json")

    if not os.path.exists(input_path):
        print(f"⚠️ 정책 문서 파일이 없습니다: {input_path}")
        return

    print(f"[vector_generator] 정책 문서 임베딩 중... ({MODEL_NAME})")
    out = {}

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    for line in tqdm(lines):
        # "정책명: 내용" 형태로 되어 있다고 가정
        if ":" in line:
            name, text = line.split(":", 1)
        else:
            name, text = f"정책_{len(out)+1}", line

        v = embed_text(text, prefix="passage: ")
        if v is not None:
            out[name.strip()] = v

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ 정책 벡터 저장 완료: {output_path}")


# --------------------------
# 2️⃣ 지역별 여론 벡터화
# --------------------------
def generate_region_vectors():
    """기존 17개 지역 *_vectors.json 자동 생성"""
    regions = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
    ]

    for region in regions:
        src_path = os.path.join(FILES_DIR, f"{region}_vectors.json")
        dst_path = os.path.join(FILES_DIR, f"{region}_vectors_e5.json")

        if not os.path.exists(src_path):
            print(f"⚠️ {region}_vectors.json 파일이 없습니다.")
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            region_data = json.load(f)

        print(f"[vector_generator] {region} 여론 벡터 재생성 중...")
        topic_dict = {}

        # 형태에 따라 자동 인식 (list 또는 dict)
        if isinstance(region_data, list):
            iterable = region_data
        elif isinstance(region_data, dict):
            iterable = [{"topic": t, "text": d.get("text", "")} for t, d in region_data.items()]
        else:
            print(f"⚠️ {region} 데이터 구조 인식 실패")
            continue

        for item in tqdm(iterable):
            topic = item.get("topic")
            text = item.get("text")
            if not topic or not text:
                continue
            v = embed_text(text, prefix="query: ")
            if v is not None:
                topic_dict.setdefault(topic, []).append(v)

        topic_avg = {
            topic: {
                "vector": np.mean(vectors, axis=0).tolist(),
                "sample_count": len(vectors)
            }
            for topic, vectors in topic_dict.items()
        }

        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(topic_avg, f, ensure_ascii=False, indent=2)

        print(f"✅ {region}_vectors_e5.json 저장 완료 ({len(topic_avg)}개 토픽)")


# --------------------------
# 실행 진입점
# --------------------------
if __name__ == "__main__":
    print("🚀 Welling Vector Generator 시작")

    # 정책 벡터 생성
    generate_policy_vectors()

    # 지역별 여론 벡터 생성
    generate_region_vectors()

    print("🎉 모든 벡터 생성이 완료되었습니다.")
