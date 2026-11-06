# 🚀 DocPrecision-RAG: Advanced RAG Architecture

> **Comparative RAG Engine:** From Naive RAG baseline to Advanced RAG with Semantic Chunking, Hybrid Search (BM25 + BGE), and Cross-Encoder Reranking.

**DocPrecision-RAG**, sabit karakter sayısına dayalı standart (Naive) RAG yaklaşımlarının ötesine geçerek; metni anlamsal mesafe kopuşlarına göre bölen (**Semantic Chunking**), kelime tabanlı ve vektörel aramayı RRF ile birleştiren (**Hybrid Search**), ve bulunan sonuçları Cross-Encoder ile süzen (**Reranking**) uçtan uca modüler bir RAG mimarisidir.

---

## 🏛️ System Architecture

```text
                               [ Raw PDF Document ]
                                        │
                                        ▼
                              [ PyMuPDF Document Reader ]
                                        │
                                        ▼
                           [ Custom Semantic Chunker ]
                  (Dynamic Percentile-based Cosine Distance Thresholding)
                                        │
                                        ▼
                            [ Hybrid Search Index ]
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
   [ BM25 Retriever ]                                       [ BGE Dense Retriever ]
 (Exact Keyword Matching)                                 (BAAI/bge-small-en-v1.5)
           │                                                         │
           └────────────────────────────┬────────────────────────────┘
                                        ▼
                       [ Reciprocal Rank Fusion (RRF) ]
                            (Fused Candidate Pool)
                                        │
                                        ▼
                          [ Cross-Encoder Reranker ]
                          (BAAI/bge-reranker-base)
                                        │
                                        ▼
                           [ Strict Prompt Engineering ]
                                        │
                                        ▼
                             [ Local LLM (Ollama) ]
                               (Llama 3.2 / Mistral)
```

---

## 🔑 Key Features & Technical Highlights

- **Semantic Chunking (`semantic_chunker.py`):** Replaces naive fixed-size chunking with an adaptive sentence-distance analyzer using `BAAI/bge-small-en-v1.5`. Breaks text at semantic transition points using dynamic percentile thresholding ($85^{\text{th}}$ percentile).

- **Hybrid Retrieval Engine (`hybrid_retriever.py`):** Combines lexical search (**BM25**) for exact terms (such as model IDs or error codes) and semantic search (**Dense BGE Embeddings**) for contextual queries. Merges rankings using **Reciprocal Rank Fusion (RRF)** ($k=60$).

- **Cross-Encoder Reranking (`reranker.py`):** Employs `BAAI/bge-reranker-base` to evaluate full query-document attention scores, discarding low-relevance noise before feeding context to the LLM.

- **Strict Hallucination Prevention (`advanced_rag.py`):** Uses strict prompt rules to constrain the LLM strictly to retrieved context, returning clean `"information not found"` fallbacks when evidence is missing.

---

## 🛠️ Project Structure

```text
DocPrecision-RAG/
├── data/
│   └── manual.pdf             # Sample technical PDF document
├── semantic_chunker.py        # Cosine distance percentile chunking engine
├── hybrid_retriever.py        # BM25 + Dense Vector + RRF Retriever module
├── reranker.py                # Cross-Encoder Reranking module
├── advanced_rag.py            # Complete end-to-end RAG Pipeline orchestrator
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites

Ensure you have **Python 3.10+** and **Ollama** installed locally.

```bash
# Pull your preferred local LLM
ollama pull llama3.2
```

### 2. Installation

```bash
# Clone repository
git clone https://github.com/ssevvalyavuz/DocPrecision-RAG.git
cd DocPrecision-RAG

# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Advanced RAG Pipeline

```bash
python advanced_rag.py
```

---

## 📊 RAG Pipeline

The Advanced RAG pipeline consists of the following stages:

### 1. Document Loading

PDF documents are loaded using **PyMuPDF**, allowing the system to extract raw textual content from the source document.

### 2. Semantic Chunking

Instead of splitting documents using a fixed character or token size, the system analyzes the semantic distance between consecutive sentences.

Sentence embeddings are generated using:

```text
BAAI/bge-small-en-v1.5
```

Cosine distances are then calculated between neighboring sentence embeddings.

A dynamic percentile-based threshold is used to identify semantic transition points:

```text
Threshold = 85th Percentile of Sentence Distances
```

This allows the chunker to adapt to the structure of the document rather than relying on a fixed chunk size.

### 3. Hybrid Retrieval

The retrieval layer combines two complementary approaches:

#### BM25 Retrieval

BM25 provides lexical retrieval and is particularly useful for:

- Exact keywords
- Model names
- Error codes
- Product IDs
- Technical terminology
- Specific identifiers

#### Dense Retrieval

Dense retrieval uses BGE embeddings to identify semantically similar content even when the exact query terms are not present.

The embedding model used is:

```text
BAAI/bge-small-en-v1.5
```

### 4. Reciprocal Rank Fusion

BM25 and dense retrieval rankings are combined using **Reciprocal Rank Fusion (RRF)**.

The RRF parameter is:

```text
k = 60
```

RRF allows the system to combine different retrieval strategies into a unified candidate ranking.

### 5. Cross-Encoder Reranking

The candidates returned by the hybrid retriever are passed through a Cross-Encoder.

The model used is:

```text
BAAI/bge-reranker-base
```

Unlike embedding-based retrieval, the Cross-Encoder evaluates the query and document together, allowing more precise relevance scoring.

Low-relevance candidates are removed before the context is passed to the LLM.

### 6. Strict Prompt Engineering

The final prompt instructs the LLM to answer using only the retrieved context.

The system is designed to avoid unsupported answers and hallucinations.

When the retrieved context does not contain enough information, the system returns an explicit fallback such as:

```text
Information not found in the provided context.
```

### 7. Local LLM Generation

The final response is generated locally through **Ollama**.

Supported local models can include:

```text
Llama 3.2
Mistral
```

---

## 🧠 Why Advanced RAG?

A traditional Naive RAG pipeline commonly follows this structure:

```text
PDF
 ↓
Fixed-size Chunking
 ↓
Vector Embeddings
 ↓
Vector Search
 ↓
LLM
```

Although this architecture is simple, it can suffer from several limitations:

- Fixed-size chunks may split related concepts.
- Exact keywords may not be retrieved effectively through semantic search alone.
- Semantically similar but weakly relevant chunks may enter the context.
- Irrelevant retrieved content can reduce answer quality.
- Larger context windows can increase noise.
- The LLM may generate unsupported information when retrieval is insufficient.

DocPrecision-RAG addresses these problems by introducing multiple retrieval and ranking stages:

```text
PDF
 ↓
Semantic Chunking
 ↓
 ┌─────────────────┐
 │ BM25 Retrieval  │
 └────────┬────────┘
          │
          ├─────── RRF ───────┐
          │                    │
 ┌────────▼────────┐           │
 │ Dense Retrieval │───────────┘
 └─────────────────┘
          ↓
 Cross-Encoder Reranking
          ↓
 Strict Prompt
          ↓
 Local LLM
```

---

## ⚡ Naive RAG vs Advanced RAG

| Feature | Naive RAG | DocPrecision-RAG |
|---|---|---|
| Document Loading | PDF Reader | PyMuPDF |
| Chunking | Fixed-size | Semantic Chunking |
| Embeddings | Dense | BGE Dense Embeddings |
| Keyword Search | ❌ | BM25 |
| Semantic Search | ✅ | ✅ |
| Hybrid Retrieval | ❌ | ✅ |
| Rank Fusion | ❌ | RRF |
| Reranking | ❌ | Cross-Encoder |
| Hallucination Control | Basic | Strict Prompting |
| LLM | Local / API | Local Ollama |
| Architecture | Simple | Modular Multi-Stage Pipeline |

---

## 📈 Retrieval Strategy

The system uses a multi-stage retrieval architecture:

```text
User Query
    │
    ▼
┌─────────────────────┐
│   Query Processing  │
└──────────┬──────────┘
           │
           ▼
     ┌─────────────┐
     │   BM25      │
     │   Search    │
     └──────┬──────┘
            │
            │
     ┌──────▼──────┐
     │ Dense BGE   │
     │   Search    │
     └──────┬──────┘
            │
            ▼
┌────────────────────────┐
│ Reciprocal Rank Fusion │
│          RRF           │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  Cross-Encoder         │
│  Reranking             │
└────────────┬───────────┘
             │
             ▼
      Top Relevant Chunks
             │
             ▼
     Strict LLM Prompt
             │
             ▼
        Local LLM
             │
             ▼
      Final Answer
```

---

## 🔬 Semantic Chunking

Semantic Chunking is used to identify meaningful boundaries between different topics or concepts.

Instead of:

```text
Every 500 characters → New Chunk
```

the system uses:

```text
Sentence
   ↓
Embedding
   ↓
Cosine Distance
   ↓
Distance Distribution
   ↓
85th Percentile Threshold
   ↓
Semantic Boundary
```

This approach allows chunks to adapt to the semantic structure of the source document.

### Benefits

- Better preservation of contextual relationships
- Reduced topic fragmentation
- More meaningful retrieval units
- Less dependency on arbitrary chunk sizes
- Improved retrieval precision

---

## 🔎 Hybrid Search

Hybrid Search combines lexical and semantic retrieval.

### BM25

BM25 is effective when the query contains exact terms:

```text
"ERR_CONNECTION_REFUSED"
"Model X100"
"API_KEY"
"SQLSTATE 42P01"
```

### Dense Retrieval

Dense retrieval is useful for conceptual queries:

```text
"What causes the application to lose database connectivity?"
```

Even if the document does not contain the exact wording of the query, the embedding model can identify semantically related content.

### Combined Approach

```text
                 User Query
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      BM25 Search          BGE Search
          │                     │
          └──────────┬──────────┘
                     ▼
                    RRF
                     │
                     ▼
            Unified Ranking
```

---

## 🔗 Reciprocal Rank Fusion (RRF)

RRF combines rankings from different retrieval systems.

The general scoring formula is:

$$
RRF(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}
$$

Where:

- $d$ = document
- $R$ = retrieval systems
- $rank_r(d)$ = rank of document $d$ in retriever $r$
- $k$ = ranking constant

This project uses:

$$
k = 60
$$

The approach allows BM25 and Dense Retrieval to contribute to the final ranking without requiring their raw scores to be directly comparable.

---

## 🎯 Cross-Encoder Reranking

Initial retrieval is optimized for speed and recall.

The Cross-Encoder is then used for more precise relevance estimation.

```text
1000+ Document Chunks
        │
        ▼
  Initial Retrieval
        │
        ▼
   Top Candidates
        │
        ▼
 Cross-Encoder
        │
        ▼
High-Quality Context
        │
        ▼
       LLM
```

The Cross-Encoder model used in this project is:

```text
BAAI/bge-reranker-base
```

This second-stage ranking reduces the amount of irrelevant context passed to the language model.

---

## 🛡️ Hallucination Prevention

The system uses strict prompt engineering to limit the LLM's answer to retrieved evidence.

The intended behavior is:

```text
Retrieved Context
       │
       ▼
Does the context contain
the required information?
       │
   ┌───┴───┐
   │       │
  YES      NO
   │       │
   ▼       ▼
 Answer   Fallback
 from     response
 context
```

The model should not invent information that is not supported by the retrieved context.

Example fallback:

```text
Information not found in the provided context.
```

---

## 🧩 Project Structure

```text
DocPrecision-RAG/
│
├── data/
│   └── manual.pdf
│
├── semantic_chunker.py
│   └── Semantic chunking engine
│
├── hybrid_retriever.py
│   └── BM25 + Dense Retrieval + RRF
│
├── reranker.py
│   └── Cross-Encoder reranking
│
├── advanced_rag.py
│   └── End-to-end RAG pipeline
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Core development language |
| PyMuPDF | PDF document processing |
| BAAI/bge-small-en-v1.5 | Dense sentence/document embeddings |
| BM25 | Lexical retrieval |
| RRF | Retrieval rank fusion |
| BAAI/bge-reranker-base | Cross-Encoder reranking |
| Ollama | Local LLM inference |
| Llama 3.2 | Local language model |
| Mistral | Alternative local language model |

---

## 📦 Installation

### Requirements

- Python 3.10+
- Ollama
- Git
- Sufficient RAM/CPU/GPU resources for local embedding and reranking models

### Clone the Repository

```bash
git clone https://github.com/ssevvalyavuz/DocPrecision-RAG.git
cd DocPrecision-RAG
```

### Create Virtual Environment

```bash
python3 -m venv .venv
```

### Activate Virtual Environment

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```powershell
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Ollama Setup

Install Ollama and pull the desired local model.

For Llama 3.2:

```bash
ollama pull llama3.2
```

Alternatively, another compatible local model such as Mistral can be used:

```bash
ollama pull mistral
```

Make sure the Ollama service is running before starting the RAG pipeline.

---

## ▶️ Running the Project

Run the complete Advanced RAG pipeline with:

```bash
python advanced_rag.py
```

The pipeline performs the following operations:

1. Loads the PDF document.
2. Extracts text using PyMuPDF.
3. Splits the document using Semantic Chunking.
4. Creates dense embeddings.
5. Builds BM25 and Dense Retrieval indexes.
6. Retrieves candidates using both methods.
7. Combines rankings using RRF.
8. Reranks candidates using a Cross-Encoder.
9. Builds a strict context-aware prompt.
10. Sends the prompt to the local Ollama LLM.
11. Returns the final answer.

---

## 🔄 End-to-End Workflow

```text
                    PDF
                     │
                     ▼
              PyMuPDF Reader
                     │
                     ▼
             Sentence Extraction
                     │
                     ▼
            Semantic Chunking
                     │
                     ▼
             ┌───────┴───────┐
             ▼               ▼
           BM25          BGE Embeddings
             │               │
             ▼               ▼
        Lexical Rank      Dense Rank
             │               │
             └───────┬───────┘
                     ▼
                    RRF
                     │
                     ▼
             Candidate Chunks
                     │
                     ▼
          Cross-Encoder Reranker
                     │
                     ▼
              Top Context
                     │
                     ▼
             Strict Prompt
                     │
                     ▼
               Ollama LLM
                     │
                     ▼
               Final Answer
```

---

## 🧪 Example Query Flow

Suppose the user asks:

```text
Why does the application return a database connection error?
```

The system processes the query through both retrieval mechanisms.

### BM25

Searches for exact lexical matches such as:

```text
database
connection
error
connection refused
timeout
```

### Dense Retrieval

Searches for semantically related passages even if the exact phrase is not present.

### RRF

Combines the rankings from both methods.

### Cross-Encoder

Re-evaluates the query and candidate chunks together and selects the most relevant evidence.

### LLM

Receives only the selected context and generates an answer constrained by the retrieved evidence.

---

## 📊 Evaluation Strategy

The architecture can be evaluated by comparing retrieval and generation quality at different stages.

### Baseline

```text
Naive Chunking
      ↓
Dense Retrieval
      ↓
LLM
```

### Advanced

```text
Semantic Chunking
      ↓
BM25 + Dense Retrieval
      ↓
RRF
      ↓
Cross-Encoder Reranking
      ↓
Strict Prompt
      ↓
LLM
```

Potential evaluation metrics include:

- Retrieval Precision
- Retrieval Recall
- Recall@K
- Precision@K
- MRR
- NDCG
- Answer Faithfulness
- Context Relevance
- Answer Relevance
- Hallucination Rate

---

## 📌 Future Improvements

Potential extensions for the project include:

- [ ] Add automated RAG evaluation
- [ ] Add Recall@K and Precision@K benchmarks
- [ ] Add MRR / NDCG evaluation
- [ ] Compare multiple embedding models
- [ ] Compare multiple reranking models
- [ ] Add query expansion
- [ ] Add metadata filtering
- [ ] Add document-level citations
- [ ] Add conversational memory
- [ ] Add multi-document retrieval
- [ ] Add FastAPI backend
- [ ] Add web interface
- [ ] Add Docker support
- [ ] Add experiment tracking
- [ ] Add latency and token-cost benchmarking
- [ ] Add Naive RAG vs Advanced RAG automated comparison

---

## 🧭 Roadmap

### Phase 1 — Core RAG

- [x] PDF ingestion
- [x] Semantic chunking
- [x] Dense retrieval
- [x] BM25 retrieval
- [x] RRF fusion
- [x] Cross-Encoder reranking
- [x] Local LLM integration

### Phase 2 — Evaluation

- [ ] Naive RAG baseline
- [ ] Retrieval benchmark
- [ ] Reranking benchmark
- [ ] Hallucination evaluation
- [ ] Latency comparison

### Phase 3 — Productionization

- [ ] API layer
- [ ] Web UI
- [ ] Dockerization
- [ ] Logging
- [ ] Monitoring
- [ ] Configuration management

### Phase 4 — Advanced Retrieval

- [ ] Query rewriting
- [ ] Multi-query retrieval
- [ ] Parent-document retrieval
- [ ] Metadata-aware retrieval
- [ ] Multi-document reasoning

---

## 🎓 Key Concepts Demonstrated

This project demonstrates practical implementation of several modern RAG concepts:

- **Retrieval-Augmented Generation**
- **Semantic Chunking**
- **Dense Vector Retrieval**
- **Lexical Retrieval**
- **BM25**
- **Hybrid Search**
- **Reciprocal Rank Fusion**
- **Cross-Encoder Reranking**
- **Prompt Engineering**
- **Hallucination Mitigation**
- **Local LLM Inference**
- **Modular RAG Architecture**

---

## 👩‍💻 Author

**Şevval Yavuz**

Computer Engineering

GitHub: [@ssevvalyavuz](https://github.com/ssevvalyavuz)

---

## 📄 License

This project is intended for educational, research, and portfolio purposes.