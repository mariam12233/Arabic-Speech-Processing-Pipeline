# 🎙️ Arabic Speech Processing Pipeline

A complete Arabic audio understanding system that performs:

* Automatic Speech Recognition (ASR)
* Arabic Text Summarization
* Semantic Audio Search
* Keyword Spotting
* Search Evaluation using Precision@K

The project is implemented as a step-by-step Jupyter Notebook pipeline for experimentation and reproducibility.

---

# 📌 Project Overview

This project builds an end-to-end Arabic speech processing pipeline using state-of-the-art transformer models.

The system:

1. Converts Arabic speech into text using Whisper
2. Summarizes transcripts using mT5
3. Builds semantic embeddings using multilingual E5
4. Performs semantic search over audio content using FAISS
5. Detects predefined Arabic keywords inside transcripts
6. Evaluates retrieval quality using Precision@K

---

# 🧠 Full Pipeline

```text
Arabic Audio Files (MASC Dataset)
                ↓
┌──────────────────────────────────┐
│ Stage 1 — ASR                    │
│ openai/whisper-small             │
│ Audio → Arabic Transcript        │
└──────────────────────────────────┘
                ↓
┌──────────────────────────────────┐
│ Stage 2 — Summarization          │
│ mT5_multilingual_XLSum           │
│ Transcript → Arabic Summary      │
└──────────────────────────────────┘
                ↓
┌──────────────────────────────────┐
│ Stage 3 — Embeddings             │
│ multilingual-e5-base             │
│ Text → Dense Vector              │
└──────────────────────────────────┘
                ↓
┌──────────────────────────────────┐
│ Stage 4 — Semantic Search        │
│ FAISS IndexFlatIP                │
│ Query → Relevant Audio Files     │
└──────────────────────────────────┘
                ↓
┌──────────────────────────────────┐
│ Stage 5 — Keyword Spotting       │
│ Detect predefined Arabic words   │
└──────────────────────────────────┘
                ↓
┌──────────────────────────────────┐
│ Stage 6 — Evaluation             │
│ Precision@1 and Precision@3      │
└──────────────────────────────────┘
```

---

# 📂 Notebook Structure

The notebook is organized into independent stages.

| Cell | Stage            | Description                               |
| ---- | ---------------- | ----------------------------------------- |
| 1    | Imports          | Load libraries and initialize environment |
| 2    | ASR              | Transcribe Arabic audio using Whisper     |
| 3    | Summarization    | Generate Arabic summaries using mT5       |
| 4    | Search Index     | Create semantic embeddings + FAISS index  |
| 5    | Keyword Spotting | Detect important Arabic keywords          |
| 6    | Search Demo      | Run semantic search queries               |
| 7    | Evaluation       | Measure Precision@K                       |

Each cell can be run independently.

---

# 🗃️ Dataset

## MASC — Massive Arabic Speech Corpus

| Property       | Value                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------- |
| Dataset        | MASC                                                                                               |
| Source         | [https://huggingface.co/datasets/hirundo-io/MASC](https://huggingface.co/datasets/hirundo-io/MASC) |
| Language       | Arabic                                                                                             |
| Sampling Rate  | 16 kHz                                                                                             |
| Type           | Multi-dialect speech                                                                               |
| Loading Method | Streaming                                                                                          |

The dataset contains Arabic speech collected from YouTube videos across different topics and dialects.

The notebook streams the dataset directly from HuggingFace without downloading the entire dataset locally.

---

# 🤖 Models Used

## 1️⃣ Automatic Speech Recognition (ASR)

| Property  | Value                  |
| --------- | ---------------------- |
| Model     | `openai/whisper-small` |
| Task      | Speech-to-Text         |
| Language  | Arabic                 |
| Framework | Transformers Pipeline  |

### Configuration

```python
pipeline(
    'automatic-speech-recognition',
    model='openai/whisper-small',
    generate_kwargs={
        'language': 'arabic',
        'task': 'transcribe',
        'no_repeat_ngram_size': 3
    },
    return_timestamps=True
)
```

### Features

* Arabic speech transcription
* Timestamp generation
* Reduced repetitive outputs
* Streaming dataset support

---

## 2️⃣ Arabic Summarization

| Property | Value                                  |
| -------- | -------------------------------------- |
| Model    | `csebuetnlp/mT5_multilingual_XLSum`    |
| Task     | Multilingual Abstractive Summarization |
| Output   | Arabic summaries                       |

The summarization stage converts long transcripts into short and meaningful Arabic summaries.

---

## 3️⃣ Semantic Embedding Model

| Property          | Value                                                        |
| ----------------- | ------------------------------------------------------------ |
| Model             | `intfloat/multilingual-e5-base`                              |
| Vector Database   | FAISS                                                        |
| Similarity Metric | Inner Product (Cosine Similarity with normalized embeddings) |

### Embedding Strategy

```python
'embedding = model.encode(
    ["passage: " + text],
    normalize_embeddings=True
)'
```

### Query Encoding

```python
'query: ' + query
```

The E5 model is optimized for semantic retrieval and multilingual search tasks.

---

# 🔍 Semantic Search System

The system allows users to search audio files using natural Arabic queries.

### Example Queries

```text
ألمانيا وهامبورغ
كندا واستراليا
صحة وغذاء
برمجة وتصميم
دين وإسلام
```

### Search Workflow

```text
User Query
    ↓
E5 Query Embedding
    ↓
FAISS Similarity Search
    ↓
Top Matching Audio Files
```

### Output Example

```text
🔍 Query: ألمانيا وهامبورغ

Top Results:
1. -SuGpbd7KMI.wav
2. another_file.wav
3. another_file.wav
```

---

# 🏷️ Keyword Spotting

The project also supports keyword spotting inside transcripts.

### Example Keywords

```python
[
    'الله',
    'كندا',
    'ألمانيا',
    'برمجة',
    'رمضان',
    'صحة',
    'تعليم',
    'اقتصاد',
    'سياسة',
    'تكنولوجيا'
]
```

The system scans all transcripts and returns:

* Matching files
* Transcript preview
* Keyword occurrences

---

# 📊 Evaluation

## Precision@K

The notebook evaluates retrieval quality using:

* Precision@1
* Precision@3

### Evaluation Workflow

Known correct files are manually mapped to test queries.

Example:

```python
('ألمانيا وهامبورغ', '-SuGpbd7KMI.wav')
```

The system checks whether the correct audio file appears in:

* Rank 1
* Top 3 results

---

# 📁 Outputs

| File                                | Description                 |
| ----------------------------------- | --------------------------- |
| `outputs/asr_results.csv`           | Whisper transcripts         |
| `outputs/summarization_results.csv` | Arabic summaries            |
| `outputs/search_index.faiss`        | FAISS vector index          |
| `outputs/search_metadata.csv`       | Metadata used for retrieval |
| `outputs/keyword_results.csv`       | Keyword spotting results    |

---

# ⚙️ Installation

## Install Dependencies

```bash
pip install transformers datasets torch torchaudio
pip install sentence-transformers faiss-cpu
pip install pandas numpy
```

---

# 🚀 Running the Notebook

Open the notebook:

```bash
jupyter notebook pipeline.ipynb
```

Run the cells sequentially.

Notes:

* ASR stage is the slowest stage
* Summarization may take time depending on hardware
* Search and keyword spotting are fast

---

# 🧪 Technologies Used

| Technology               | Purpose               |
| ------------------------ | --------------------- |
| PyTorch                  | Deep Learning Backend |
| HuggingFace Transformers | Model Inference       |
| SentenceTransformers     | Semantic Embeddings   |
| FAISS                    | Vector Search         |
| Pandas                   | Data Processing       |
| NumPy                    | Numerical Operations  |

---

# 📈 Future Improvements

Potential future enhancements:

* Real-time streaming transcription
* Arabic dialect classification
* Speaker diarization
* Hybrid keyword + semantic retrieval
* Web interface deployment
* GPU optimization
* Audio chunking for long recordings
* Fine-tuning Whisper on Arabic dialects

---

# 👩‍💻 Author

Mariam — AI Student at EJUST University
