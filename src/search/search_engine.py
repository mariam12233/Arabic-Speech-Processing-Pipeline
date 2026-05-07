# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

# Load summarization results
df = pd.read_csv("outputs/summarization_results.csv")
print(f"Loaded {len(df)} documents\n")

# Load multilingual embedding model
print("Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded.\n")

# Encode all transcripts into vectors
print("Encoding transcripts...")
transcripts = df["transcript"].tolist()
embeddings = embedder.encode(transcripts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
print("Building search index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"Index built with {index.ntotal} vectors.\n")

# Search function
def search(query, top_k=3):
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    print(f"\nQuery: {query}")
    print("=" * 50)
    for rank, idx in enumerate(indices[0]):
        print(f"Result {rank+1}:")
        print(f"  File    : {df.iloc[idx]['filename']}")
        print(f"  Summary : {df.iloc[idx]['summary']}")
        print(f"  Score   : {distances[0][rank]:.4f}")
        print()

# Test queries
search("مدينة ألمانية وميناء")
search("برمجة وكلاس")
search("كندا واستراليا")
search("دين وإسلام")

# Save index and metadata
faiss.write_index(index, "outputs/search_index.faiss")
df.to_csv("outputs/search_metadata.csv", index=False, encoding="utf-8-sig")
print("Index saved.")