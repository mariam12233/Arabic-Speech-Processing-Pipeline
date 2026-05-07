from datasets import load_dataset
import pandas as pd
from transformers import pipeline

# Load Whisper
print("Loading Whisper model...")
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
print("Model loaded.")

# Stream dataset
dataset = load_dataset("hirundo-io/MASC", split="train", streaming=True)

results = []

for i, sample in enumerate(dataset):
    print(f"\nProcessing sample {i}...")

    filename = sample["audio"]["path"].split("/")[-1]
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    duration = len(audio_array) / sampling_rate
    print(f"  File: {filename} | Duration: {duration:.2f}s")

    # Trim to first 30 seconds
    audio_trimmed = audio_array[:30 * sampling_rate]

    # Run ASR
    print("  Transcribing...")
    output = asr({"array": audio_trimmed, "sampling_rate": sampling_rate})
    transcript = output["text"].strip()

    print(f"  Transcript: {transcript}")

    results.append({
        "filename": filename,
        "duration": round(duration, 2),
        "transcript": transcript
    })

    if i >= 9:  # process 10 samples
        break

# Save
df = pd.DataFrame(results)
df.to_csv("outputs/asr_results.csv", index=False, encoding="utf-8-sig")
print(f"\nDone! Saved {len(df)} transcripts to outputs/asr_results.csv")
print(df[["filename", "transcript"]])