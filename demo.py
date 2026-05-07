# -*- coding: utf-8 -*-
"""
Arabic Audio Search Engine — Demo
Loads saved results, no internet needed, starts in 30 seconds
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

print("=" * 60)
print("   ARABIC AUDIO SEARCH ENGINE — DEMO")
print("=" * 60)

# Load saved results
print("\nLoading saved data...")
sum_df = pd.read_csv("outputs/search_metadata.csv")
print(f"Loaded {len(sum_df)} documents.")

# Load embedding model
print("Loading embedding model (first run downloads ~1.1GB)...")
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

# Build index with normalization + passage prefix
print("Building search index...")
embeddings = embedder.encode(
    ["passage: " + t for t in sum_df["transcript"].tolist()],
    show_progress_bar=True,
    normalize_embeddings=True
)
embeddings = np.array(embeddings).astype("float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"Ready! {index.ntotal} audio files indexed.\n")

print("=" * 60)
print("   Type any Arabic query to search your audio files")
print("   Type 'exit' to quit")
print("=" * 60)

while True:
    query = input("\n🔍 Enter your query in Arabic: ").strip()

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if not query:
        continue

    query_vec = embedder.encode(
        ["query: " + query],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(query_vec, 3)

    print(f"\nTop 3 results for: '{query}'")
    print("-" * 50)
    for rank, idx in enumerate(indices[0]):
        score = distances[0][rank]
        if score > 0.6:
            relevance = "✅ Relevant"
        elif score > 0.4:
            relevance = "🟡 Possible match"
        else:
            relevance = "⚠️ Weak match"

        print(f"\n  Rank {rank+1}: {relevance}")
        print(f"  📁 File    : {sum_df.iloc[idx]['filename']}")
        print(f"  📝 Summary : {sum_df.iloc[idx]['summary']}")
        print(f"  📊 Score   : {score:.4f}  (1.0 = perfect match)")