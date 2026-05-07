# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import jiwer
from rouge_score import rouge_scorer


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

# ==========================================
# ASR & Summarization Evaluation
# ==========================================
print("\n" + "=" * 60)
print("   EVALUATION RESULTS — ASR & Summarization")
print("=" * 60)

# Since we don't have the original dataset ground truths locally, 
# here is a mock ground truth dictionary for demonstration of WER and ROUGE.
mock_ground_truth = {
    "-SuGpbd7KMI.wav": {
        "text": "تشتهر مدينة هامبورغ الألمانية بظاهرتين يسموها ظاهرة الأسطورة الأولى هي أسواق السمك",
        "summary": "تشتهر مدينة هامبورغ الألمانية بظاهرتين يسموها ظاهرة الأسطورة"
    },
    "-kudo6VQZwE.wav": {
        "text": "لا يوجد مجال للشك أن كندا وأستراليا تعتبران من أفضل دول العالم عندما يتعلق الأمر سواء بالتعليم أو العيش",
        "summary": "كندا وأستراليا تعتبران من أفضل دول العالم"
    },
    "1yPxazSDrwk.wav": {
        "text": "يعد الخيار من الأطعمة المفيدة لصحة الجسم وسلامته نظرا لاحتوائه على العديد من العناصر الغذائية",
        "summary": "يعد الخيار من الأطعمة المفيدة لصحة الجسم وسلامته"
    }
}

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

wer_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

print("Evaluating mock ground truth samples...\n")
for filename, gt in mock_ground_truth.items():
    pred_row = sum_df[sum_df['filename'] == filename]
    if len(pred_row) > 0:
        pred_transcript = pred_row.iloc[0]['transcript']
        pred_summary = pred_row.iloc[0]['summary']
        
        # Word Error Rate (WER)
        wer = jiwer.wer(gt['text'], pred_transcript)
        wer_scores.append(wer)
        
        # ROUGE Score
        rouge_scores = scorer.score(gt['summary'], pred_summary)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        
        print(f"📁 File: {filename}")
        print(f"  WER    : {wer:.2%}")
        print(f"  ROUGE-1: {rouge_scores['rouge1'].fmeasure:.2f}")
        print(f"  ROUGE-2: {rouge_scores['rouge2'].fmeasure:.2f}")
        print(f"  ROUGE-L: {rouge_scores['rougeL'].fmeasure:.2f}\n")

if wer_scores:
    print("-" * 60)
    print(f"Average WER    : {np.mean(wer_scores):.2%}")
    print(f"Average ROUGE-1: {np.mean(rouge1_scores):.2f}")
    print(f"Average ROUGE-2: {np.mean(rouge2_scores):.2f}")
    print(f"Average ROUGE-L: {np.mean(rougeL_scores):.2f}")
print("=" * 60)