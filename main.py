# -*- coding: utf-8 -*-
"""
Arabic Speech Processing Pipeline
- ASR: Whisper
- Summarization: mT5 XLSum
- Semantic Search: MiniLM + FAISS
- Keyword Spotting (Optional Advanced Task)
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss

print("=" * 60)
print("   ARABIC SPEECH PROCESSING PIPELINE")
print("=" * 60)

# ── STAGE 1: ASR ─────────────────────────────────────────────
print("\n[STAGE 1] Loading ASR model (Whisper)...")
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    generate_kwargs={
        "language": "arabic",
        "task": "transcribe",
        "no_repeat_ngram_size": 3
    },
    return_timestamps=True
)

dataset = load_dataset("hirundo-io/MASC", split="train", streaming=True)
asr_results = []
TARGET = 100

print(f"[STAGE 1] Transcribing {TARGET} audio samples...")
for i, sample in enumerate(dataset):
    if len(asr_results) >= TARGET:
        break

    filename = sample["audio"]["path"].split("/")[-1]

    try:
        audio_array = sample["audio"]["array"][:60 * sample["audio"]["sampling_rate"]]
        transcript = asr({
            "array": audio_array,
            "sampling_rate": sample["audio"]["sampling_rate"]
        })["text"].strip()

        print(f"  [{len(asr_results)+1}/{TARGET}] {filename} → {transcript[:60]}...")
        asr_results.append({"filename": filename, "transcript": transcript})

    except Exception as e:
        print(f"  [SKIP] {filename} — {str(e)[:60]}")
        continue

asr_df = pd.DataFrame(asr_results)
asr_df.to_csv("outputs/asr_results.csv", index=False, encoding="utf-8-sig")
print(f"[STAGE 1] ✅ Done. Saved {len(asr_df)} transcripts.")

# ── STAGE 2: SUMMARIZATION ────────────────────────────────────
print("\n[STAGE 2] Loading Summarization model (mT5)...")
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

sum_results = []
print("[STAGE 2] Summarizing transcripts...")
for i, row in asr_df.iterrows():
    try:
        inputs = tokenizer(
            row["transcript"],
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        with torch.no_grad():
            output_ids = sum_model.generate(
                inputs["input_ids"],
                max_new_tokens=60,
                num_beams=4,
                early_stopping=True
            )
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        summary = f"ERROR: {e}"

    print(f"  [{i+1}/{TARGET}] {row['filename']} → {summary[:60]}...")
    sum_results.append({
        "filename": row["filename"],
        "transcript": row["transcript"],
        "summary": summary
    })

sum_df = pd.DataFrame(sum_results)
sum_df.to_csv("outputs/summarization_results.csv", index=False, encoding="utf-8-sig")
print(f"[STAGE 2] ✅ Done. Saved {len(sum_df)} summaries.")

# ── STAGE 3: SEMANTIC SEARCH ──────────────────────────────────
print("\n[STAGE 3] Loading Embedding model (E5)...")
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

print("[STAGE 3] Building search index...")
embeddings = embedder.encode(
    ["passage: " + t for t in sum_df["transcript"].tolist()],
    show_progress_bar=True,
    normalize_embeddings=True
)
embeddings = np.array(embeddings).astype("float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "outputs/search_index.faiss")
sum_df.to_csv("outputs/search_metadata.csv", index=False, encoding="utf-8-sig")
print(f"[STAGE 3] ✅ Done. Index built with {index.ntotal} vectors.")

# ── STAGE 4: KEYWORD SPOTTING ─────────────────────────────────
print("\n[STAGE 4] Running Keyword Spotting...")
print("=" * 60)

keywords = [
    "الله",
    "كندا",
    "ألمانيا",
    "برمجة",
    "رمضان",
    "صحة",
    "تعليم",
    "اقتصاد",
    "سياسة",
    "تكنولوجيا",
]

keyword_results = []

for keyword in keywords:
    matches = asr_df[asr_df["transcript"].str.contains(keyword, na=False)]
    if len(matches) > 0:
        print(f"\n🔑 Keyword '{keyword}' found in {len(matches)} file(s):")
        for _, row in matches.iterrows():
            print(f"     📁 {row['filename']}")
            print(f"     📝 {row['transcript'][:80]}...")
            keyword_results.append({
                "keyword": keyword,
                "filename": row["filename"],
                "transcript_preview": row["transcript"][:100]
            })
    else:
        print(f"\n🔑 Keyword '{keyword}' — not found in any file")

kw_df = pd.DataFrame(keyword_results)
if not kw_df.empty:
    kw_df.to_csv("outputs/keyword_spotting_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n[STAGE 4] ✅ Done. Found {len(keyword_results)} keyword matches.")
else:
    print("\n[STAGE 4] ✅ Done. No keyword matches found.")

print("\n" + "=" * 60)
print("   ALL STAGES COMPLETE")
print("   Now run: python demo.py")
print("=" * 60)