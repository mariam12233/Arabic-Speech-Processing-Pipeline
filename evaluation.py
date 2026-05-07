# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load data
sum_df = pd.read_csv("outputs/search_metadata.csv")

# Load model
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

# Build index
embeddings = embedder.encode(
    ["passage: " + t for t in sum_df["transcript"].tolist()],
    normalize_embeddings=True
)
embeddings = np.array(embeddings).astype("float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Test queries with known expected files
test_queries = [
    ("ألمانيا وهامبورغ",     "-SuGpbd7KMI.wav"),
    ("كندا واستراليا",        "-kudo6VQZwE.wav"),
    ("صحة وغذاء وخضروات",    "1yPxazSDrwk.wav"),
    ("برمجة وتصميم",         "6dX8TbyW9Cg.wav"),
    ("رمضان وأغاني",         "-sVkr0x8Rrg.wav"),
]

print("=" * 60)
print("   EVALUATION RESULTS — Precision@K")
print("=" * 60)

correct_at_1 = 0
correct_at_3 = 0

for query, expected_file in test_queries:
    query_vec = embedder.encode(
        ["query: " + query],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(query_vec, 3)
    retrieved_files = [sum_df.iloc[idx]["filename"] for idx in indices[0]]

    hit_at_1 = retrieved_files[0] == expected_file
    hit_at_3 = expected_file in retrieved_files

    if hit_at_1:
        correct_at_1 += 1
    if hit_at_3:
        correct_at_3 += 1

    print(f"\nQuery   : {query}")
    print(f"Expected: {expected_file}")
    print(f"Got     : {retrieved_files[0]}")
    print(f"P@1     : {'✅ Correct' if hit_at_1 else '❌ Wrong'}")
    print(f"P@3     : {'✅ In top 3' if hit_at_3 else '❌ Not found'}")

print("\n" + "=" * 60)
print(f"Precision@1 : {correct_at_1}/{len(test_queries)} = {correct_at_1/len(test_queries)*100:.0f}%")
print(f"Precision@3 : {correct_at_3}/{len(test_queries)} = {correct_at_3/len(test_queries)*100:.0f}%")
print("=" * 60)