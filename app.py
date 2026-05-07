import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# Load Data
try:
    sum_df = pd.read_csv("outputs/search_metadata.csv")
    embedder = SentenceTransformer("intfloat/multilingual-e5-base")
    
    if os.path.exists("outputs/search_index.faiss"):
        index = faiss.read_index("outputs/search_index.faiss")
    else:
        embeddings = embedder.encode(
            ["passage: " + t for t in sum_df["transcript"].tolist()],
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
    IS_READY = True
except Exception as e:
    IS_READY = False
    ERROR_MSG = str(e)

def search(query):
    if not IS_READY:
        return f"Error loading index: {ERROR_MSG}", None
    
    query_vec = embedder.encode(
        ["query: " + query],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(query_vec, 3)
    
    results = []
    for rank, idx in enumerate(indices[0]):
        score = distances[0][rank]
        row = sum_df.iloc[idx]
        
        relevance = "✅ Relevant" if score > 0.6 else "🟡 Possible match" if score > 0.4 else "⚠️ Weak match"
        
        results.append([
            rank + 1,
            row['filename'],
            relevance,
            f"{score:.4f}",
            row['transcript'],
            row['summary']
        ])
    
    df_results = pd.DataFrame(results, columns=["Rank", "Filename", "Relevance", "Score", "Transcript", "Summary"])
    return df_results

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎧 Arabic Audio Search Engine - Demo Interface")
    gr.Markdown("Search through indexed Arabic audio transcripts and summaries using semantic search.")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="🔍 Enter your query in Arabic", placeholder="e.g. ألمانيا وهامبورغ")
            search_btn = gr.Button("Search", variant="primary")
        
    with gr.Row():
        results_output = gr.Dataframe(
            headers=["Rank", "Filename", "Relevance", "Score", "Transcript", "Summary"],
            interactive=False,
            wrap=True
        )
        
    search_btn.click(fn=search, inputs=query_input, outputs=results_output)
    query_input.submit(fn=search, inputs=query_input, outputs=results_output)

if __name__ == "__main__":
    demo.launch()
