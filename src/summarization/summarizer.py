from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch

# Load ASR results
df = pd.read_csv("outputs/asr_results.csv")
print(f"Loaded {len(df)} transcripts\n")

# Load model directly (bypasses broken pipeline)
print("Loading summarization model (first run downloads ~1.2GB)...")
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Model loaded.\n")

results = []

for i, row in df.iterrows():
    filename = row["filename"]
    transcript = str(row["transcript"])

    print(f"Summarizing sample {i}: {filename}")
    print(f"  Original : {transcript[:80]}...")

    try:
        inputs = tokenizer(
            transcript,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=60,
                num_beams=4,
                early_stopping=True
            )
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        summary = f"ERROR: {e}"

    print(f"  Summary  : {summary}\n")

    results.append({
        "filename": filename,
        "transcript": transcript,
        "summary": summary
    })

# Save
out_df = pd.DataFrame(results)
out_df.to_csv("outputs/summarization_results.csv", index=False, encoding="utf-8-sig")
print(f"Done! Saved to outputs/summarization_results.csv")