"""
Welling Vector Generator (CSV → E5 벡터 변환)
-----------------------------------
✅ 역할:
- 정책 문서 → policy_vectors.json 생성
- 지역별 CSV 파일 → {region}_vectors_e5.json 생성
- CSV는 app/files 폴더에 "{지역명}.csv" 형태로 존재해야 함

✅ CSV 요구사항:
- 컬럼: topic, text
  예시:
  topic,text
  주거환경,부산의 전세가격이 너무 높아요
  노동경제,일자리 찾기가 너무 힘들어요
"""

import os
import json
import numpy as np
import pandas as pd
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


def embed_text(text: str, prefix="query: "):
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
# 2️⃣ 지역별 CSV → 벡터 변환
# --------------------------
def generate_region_vectors_from_csv():
    """CSV 파일을 직접 임베딩하여 *_vectors_e5.json 생성"""
    csv_files = [f for f in os.listdir(FILES_DIR) if f.endswith(".csv")]

    for csv_file in csv_files:
        region_name = csv_file.replace(".csv", "")
        csv_path = os.path.join(FILES_DIR, csv_file)
        dst_path = os.path.join(FILES_DIR, f"{region_name}_vectors_e5.json")

        print(f"[vector_generator] {region_name}.csv → {region_name}_vectors_e5.json 변환 중...")

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="cp949")

        if "topic" not in df.columns or "text" not in df.columns:
            print(f"⚠️ {region_name}.csv 파일에 'topic' 또는 'text' 컬럼이 없습니다. 건너뜁니다.")
            continue

        topic_dict = {}
        for _, row in tqdm(df.iterrows(), total=len(df)):
            topic = str(row["topic"]).strip()
            text = str(row["text"]).strip()
            if not topic or not text:
                continue
            v = embed_text(text)
            if v is not None:
                topic_dict.setdefault(topic, []).append(v)

        if not topic_dict:
            print(f"⚠️ {region_name}.csv에서 유효한 데이터가 없습니다.")
            continue

        topic_avg = {
            topic: {
                "vector": np.mean(vectors, axis=0).tolist(),
                "sample_count": len(vectors)
            }
            for topic, vectors in topic_dict.items()
        }

        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(topic_avg, f, ensure_ascii=False, indent=2)

        print(f"✅ {region_name}_vectors_e5.json 저장 완료 ({len(topic_avg)}개 주제)")


# --------------------------
# 실행 진입점
# --------------------------
if __name__ == "__main__":
    print("🚀 Welling Vector Generator 시작")

    # 정책 벡터 생성
    generate_policy_vectors()

    # 지역 CSV 파일 기반 벡터 생성
    generate_region_vectors_from_csv()

    print("🎉 모든 벡터 생성이 완료되었습니다.")
