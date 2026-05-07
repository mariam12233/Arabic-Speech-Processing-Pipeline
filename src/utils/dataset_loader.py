"""
src/utils/dataset_loader.py
============================
Updated version — handles MASC dataset correctly
including long audio segmentation and metadata.
"""

import os
import json
import itertools
import numpy as np
import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

DATASET_NAME = "hirundo-io/MASC"
TARGET_SR    = 16000
SAVE_DIR     = "./data/masc/samples"

# MASC has very long audio files (full recordings)
# We split them into chunks of this size for ASR
MAX_CHUNK_DURATION_SEC = 30   # seconds — good for Whisper


# ─────────────────────────────────────────────────────────────
# STEP 1: Investigate Dataset Structure
# ─────────────────────────────────────────────────────────────

def investigate_dataset():
    """
    Investigate the actual structure of MASC dataset.
    Run this first to understand what fields are available.
    """

    print("\n" + "="*60)
    print("  INVESTIGATING MASC DATASET STRUCTURE")
    print("="*60)

    # Load without specifying split to see all splits
    try:
        all_splits = load_dataset(
            DATASET_NAME,
            streaming=True,
            trust_remote_code=True
        )
        print(f"\n✅ Available splits: {list(all_splits.keys())}")
        splits_to_check = list(all_splits.keys())

    except Exception as e:
        print(f"[WARN] Could not load all splits: {e}")
        print("Trying individual splits...")
        splits_to_check = ["train", "test", "validation"]

    # Check first sample of each split
    for split_name in splits_to_check:
        print(f"\n{'─'*50}")
        print(f"  Split: {split_name}")
        print(f"{'─'*50}")

        try:
            split_data = load_dataset(
                DATASET_NAME,
                split=split_name,
                streaming=True,
                trust_remote_code=True
            )

            for sample in itertools.islice(split_data, 1):
                print(f"  Keys: {list(sample.keys())}")

                for key, value in sample.items():
                    if key == "audio":
                        audio = value
                        duration = len(audio["array"]) / audio["sampling_rate"]
                        print(f"  audio:")
                        print(f"    sampling_rate : {audio['sampling_rate']} Hz")
                        print(f"    array length  : {len(audio['array'])}")
                        print(f"    duration      : {duration:.2f} seconds")
                        print(f"    dtype         : {np.array(audio['array']).dtype}")
                        print(f"    path          : {audio.get('path', 'N/A')}")
                    else:
                        print(f"  {key:<20}: {repr(value)[:100]}")

        except Exception as e:
            print(f"  [ERROR] Split '{split_name}' failed: {e}")

    print("\n" + "="*60)
    print("  INVESTIGATION COMPLETE")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────
# STEP 2: Load Dataset (Streaming)
# ─────────────────────────────────────────────────────────────

def load_masc_streaming(split="train", num_samples=None):
    """
    Load MASC dataset in streaming mode.
    """

    print(f"\n[INFO] Loading MASC ({split} split, streaming)...")

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
        trust_remote_code=True
    )

    if num_samples is not None:
        dataset = dataset.take(num_samples)
        print(f"[INFO] Taking {num_samples} samples.")

    return dataset


# ─────────────────────────────────────────────────────────────
# STEP 3: Preprocess Audio — Correct Method
# ─────────────────────────────────────────────────────────────

def preprocess_audio(audio_dict, target_sr=TARGET_SR):
    """
    Extract and preprocess audio from HuggingFace sample.
    Uses audio['array'] — NOT audio['path'].
    """

    # ✅ Use decoded array — never use hf:// path
    audio_array = np.array(audio_dict["array"], dtype=np.float32)
    original_sr = audio_dict["sampling_rate"]

    # Resample if needed
    if original_sr != target_sr:
        audio_array = librosa.resample(
            y         = audio_array,
            orig_sr   = original_sr,
            target_sr = target_sr
        )

    # Normalize
    max_val = np.max(np.abs(audio_array))
    if max_val > 0:
        audio_array = audio_array / max_val

    return audio_array.astype(np.float32), target_sr


# ─────────────────────────────────────────────────────────────
# STEP 4: Split Long Audio into Chunks
# ─────────────────────────────────────────────────────────────

def split_audio_into_chunks(audio_array, sr, chunk_duration=MAX_CHUNK_DURATION_SEC):
    """
    Split a long audio array into smaller chunks.
    
    MASC contains full recordings (5-8 minutes).
    Whisper works best with segments under 30 seconds.
    
    Args:
        audio_array    (np.ndarray) : Full audio waveform
        sr             (int)        : Sample rate
        chunk_duration (int)        : Max chunk length in seconds
    
    Returns:
        List of (chunk_array, start_sec, end_sec) tuples
    """

    chunk_size = chunk_duration * sr   # samples per chunk
    total_samples = len(audio_array)
    chunks = []

    start = 0
    while start < total_samples:
        end   = min(start + chunk_size, total_samples)
        chunk = audio_array[start:end]

        start_sec = start / sr
        end_sec   = end   / sr

        chunks.append((chunk, start_sec, end_sec))
        start = end

    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 5: Save Samples Locally (with chunking)
# ─────────────────────────────────────────────────────────────

def save_samples_locally(
    num_recordings = 5,
    save_dir       = SAVE_DIR,
    split          = "train",
    chunk_duration = MAX_CHUNK_DURATION_SEC
):
    """
    Stream MASC recordings, split into chunks, save as .wav files.

    Args:
        num_recordings (int) : Number of full recordings to process
        save_dir       (str) : Where to save .wav files
        split          (str) : Dataset split
        chunk_duration (int) : Max seconds per chunk
    """

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[INFO] Processing {num_recordings} recordings")
    print(f"[INFO] Chunk size : {chunk_duration} seconds")
    print(f"[INFO] Save dir   : {save_dir}\n")

    dataset     = load_masc_streaming(split=split, num_samples=num_recordings)
    saved       = 0
    failed      = 0
    metadata    = []

    for rec_idx, sample in enumerate(
        tqdm(dataset, total=num_recordings, desc="Processing recordings")
    ):
        try:
            # Step A: Preprocess full recording
            audio_array, sr = preprocess_audio(sample["audio"])

            duration = len(audio_array) / sr
            print(f"\n  Recording {rec_idx+1}: {duration:.1f} sec "
                  f"→ splitting into {chunk_duration}s chunks...")

            # Step B: Get transcript if available
            full_transcript = ""
            for key in ["sentence", "text", "transcription", "transcript"]:
                if key in sample and sample[key]:
                    full_transcript = sample[key]
                    break

            # Step C: Split into chunks
            chunks = split_audio_into_chunks(audio_array, sr, chunk_duration)
            print(f"            → {len(chunks)} chunks created")

            # Step D: Save each chunk
            for chunk_idx, (chunk, start_sec, end_sec) in enumerate(chunks):

                filename = f"rec{rec_idx:03d}_chunk{chunk_idx:04d}.wav"
                filepath = os.path.join(save_dir, filename)

                sf.write(filepath, chunk, sr)

                metadata.append({
                    "file"        : filename,
                    "recording"   : rec_idx,
                    "chunk"       : chunk_idx,
                    "start_sec"   : round(start_sec, 2),
                    "end_sec"     : round(end_sec, 2),
                    "duration_sec": round(end_sec - start_sec, 2),
                    "transcript"  : full_transcript,  # full recording transcript
                    "sr"          : sr
                })
                saved += 1

        except Exception as e:
            print(f"\n[WARNING] Recording {rec_idx} failed: {e}")
            failed += 1

    # Save metadata as JSON (easy to load later)
    meta_path = os.path.join(save_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Also save as simple text
    txt_path = os.path.join(save_dir, "transcripts.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(f"{m['file']}|{m['start_sec']}|{m['end_sec']}|{m['transcript']}\n")

    # Summary
    print(f"\n{'='*50}")
    print(f"  SAVE COMPLETE")
    print(f"{'='*50}")
    print(f"  Recordings : {num_recordings}")
    print(f"  Chunks     : {saved}")
    print(f"  Failed     : {failed}")
    print(f"  Saved to   : {save_dir}")
    print(f"  Metadata   : {meta_path}")
    print(f"{'='*50}\n")

    return save_dir, metadata


# ─────────────────────────────────────────────────────────────
# STEP 6: Load Local Samples
# ─────────────────────────────────────────────────────────────

def load_local_samples(save_dir=SAVE_DIR):
    """
    Load previously saved local .wav chunks.
    Use this after running save_samples_locally() once.

    Returns:
        list of dicts with keys:
        "file", "audio", "sr", "transcript", "start_sec", "end_sec"
    """

    meta_path = os.path.join(save_dir, "metadata.json")

    if not os.path.exists(save_dir):
        raise FileNotFoundError(
            f"Folder not found: {save_dir}\n"
            f"Run save_samples_locally() first."
        )

    # Load metadata
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_list = json.load(f)
            metadata = {m["file"]: m for m in meta_list}

    # Load wav files
    wav_files = sorted([f for f in os.listdir(save_dir) if f.endswith(".wav")])
    print(f"[INFO] Loading {len(wav_files)} chunks from {save_dir}...")

    samples = []
    for wav_file in wav_files:
        filepath = os.path.join(save_dir, wav_file)
        try:
            audio_array, sr = librosa.load(filepath, sr=TARGET_SR)
            meta = metadata.get(wav_file, {})

            samples.append({
                "file"      : filepath,
                "audio"     : audio_array,
                "sr"        : sr,
                "transcript": meta.get("transcript", ""),
                "start_sec" : meta.get("start_sec", 0),
                "end_sec"   : meta.get("end_sec", 0),
                "duration"  : meta.get("duration_sec", 0),
                "recording" : meta.get("recording", 0),
                "chunk"     : meta.get("chunk", 0)
            })
        except Exception as e:
            print(f"[WARN] Could not load {wav_file}: {e}")

    print(f"[INFO] Loaded {len(samples)} chunks successfully.\n")
    return samples


# ─────────────────────────────────────────────────────────────
# MAIN — Run all steps
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  MASC Dataset Loader")
    print("="*60)

    # ── STEP 1: Investigate structure first ───────────────────
    investigate_dataset()

    # ── STEP 2: Stream and inspect raw samples ────────────────
    print("[TEST] Inspecting 2 raw samples...\n")
    dataset = load_masc_streaming(split="train", num_samples=2)

    for i, sample in enumerate(dataset):
        audio_array, sr = preprocess_audio(sample["audio"])
        duration = len(audio_array) / sr

        print(f"  Sample {i+1}:")
        print(f"    Keys     : {list(sample.keys())}")
        print(f"    Duration : {duration:.1f} sec")
        print(f"    Shape    : {audio_array.shape}")

        # Show all non-audio fields
        for key in sample:
            if key != "audio":
                print(f"    {key:<12}: {sample[key]}")
        print()

    # ── STEP 3: Save chunked samples locally ─────────────────
    print("[SAVE] Saving 3 recordings as 30-second chunks...\n")
    save_dir, metadata = save_samples_locally(
        num_recordings = 3,
        save_dir       = SAVE_DIR,
        chunk_duration = 30
    )

    # ── STEP 4: Load and verify local samples ─────────────────
    print("[LOAD] Loading saved chunks...\n")
    local_samples = load_local_samples(SAVE_DIR)

    print(f"  Total chunks loaded : {len(local_samples)}")
    print(f"\n  First chunk:")
    s = local_samples[0]
    print(f"    File      : {s['file']}")
    print(f"    Duration  : {s['duration']:.1f} sec")
    print(f"    Shape     : {s['audio'].shape}")
    print(f"    SR        : {s['sr']} Hz")
    print(f"    Transcript: {s['transcript'] or '(empty — check dataset)'}")

    print("\n✅ Dataset loader working correctly!")
    print("   Next step: feed chunks into Whisper for transcription")