# Hebrew RAG Chatbot for Psychology & Mental Health  
**LangGraph · Pinecone · BGE‑M3 · OpenAI · Streamlit**

An intelligent, interactive **Hebrew** chatbot focused on **psychology and mental health**.  
The knowledge base is collected from the professional portal **Betipulnet**, so the model is grounded in reliable therapeutic content.  
The solution is built on an advanced **RAG (Retrieval‑Augmented Generation)** architecture, orchestrated with **LangGraph** (State + Memory), and includes a full **Streamlit** chat UI with **streaming** for a natural experience.

> ⚠️ **Safety & guardrails:** This system does **not** provide medical diagnoses or personal treatment recommendations. It is intended for general information only, using strict system prompts to limit sensitive content and reduce hallucinations.

---

## Table of Contents
- [Overview](#overview)
- [Product Thinking & Strategic Decisions](#product-thinking)
- [System Architecture](#architecture)
- [Tech Stack](#tech)
- [Advanced Features](#features)
- [Getting Started](#getting-started)
- [Important Notes](#notes)

---

<a id="overview"></a>
## Overview
This project was designed from day one as an **enterprise‑ready** solution, with emphasis on:
- **Full Hebrew support** (including RTL UI and Hebrew knowledge ingestion).
- **Modularity** and a clean architecture (Data → Retrieval → Generation → UI).
- **Transparency** via citations (chunks + similarity score).
- **Handling classic RAG limitations** (loss of context in follow‑up questions, hallucination sensitivity, etc.).

---

<a id="product-thinking"></a>
## Product Thinking & Strategic Decisions
This was not built as a “tech exercise in a vacuum,” but with a product mindset.  
After mapping typical enterprise customer profiles (government bodies, ministries of health, banks, and large public institutions), a key strategic decision was made: **build the entire system end‑to‑end around Hebrew**.

- **Hebrew NLP challenge:** Most LLMs were trained primarily on English corpora. To achieve enterprise‑grade retrieval quality in Hebrew, the embedding model **BAAI/bge‑m3** was chosen for its strong multilingual representation capabilities.
- **Reliability & safety:** The psychology/mental‑health domain is a good demonstration of handling sensitive information. The system enforces strict **system prompts** that prevent unsafe answers (e.g., diagnoses or medical advice).

---

<a id="architecture"></a>
## System Architecture
The system follows a modular design split into three main layers: **Data**, **RAG Engine**, and **UI**.

### 1) Data Ingestion & Preprocessing
- Extract text from PDFs and perform basic cleaning (deduplication, whitespace/line normalization).
- **Semantic‑ish chunking** (in `src/preprocess.py`):  
  Instead of “blind” fixed‑size cutting, the algorithm detects paragraph/sentence boundaries and keeps a **1–2 sentence overlap** between chunks to preserve context.

### 2) Vector DB & Indexing
- Each chunk is embedded and uploaded to **Pinecone**.
- Each vector includes **rich metadata** (document name, topic, identifiers, etc.) to enable traceability, filtering, and source display.

### 3) LangGraph Agent (Core RAG Engine)
The flow is managed by a **StateGraph** with two main nodes:

- **Retrieve**  
  Receives the user query. If there is chat history, a fast LLM rewrites the question into a **standalone query** based on context.  
  The system also supports **HyDE (Hypothetical Document Embeddings)** — generating a hypothetical document to improve semantic retrieval.

- **Generate**  
  Consumes retrieved content (only above a minimal **similarity threshold**) and produces a Hebrew answer under the system prompt, while blocking hallucinations and sensitive medical content.

#### Flow Diagram (Mermaid)
```mermaid
flowchart LR
  U[User] --> S[Streamlit UI]
  S --> LG[LangGraph Agent]
  LG --> R[Retrieve Node]

  R -->|Rewrite question| Mmini[GPT-4o-mini]
  R -->|HyDE optional| Mmini

  Mmini --> Q[Standalone / HyDE text]
  Q --> E[Embed query (BGE-M3)]
  E -->|Vector search| P[Pinecone]

  P --> C[Relevant Chunks + Metadata]
  C --> G[Generate Node]
  Q --> G

  G --> Mo[GPT-4o]
  Mo --> S
  S -->|Citations| U
```

---

<a id="tech"></a>
## 🛠️ Tech Stack
- **LangGraph & LangChain** – state management, agentic workflow orchestration, and memory via an **SQLite checkpointer**.
- **Pinecone (Vector DB)** – cloud vector search (cosine similarity).
- **BAAI/bge‑m3** (HuggingFace / SentenceTransformers) – high‑quality multilingual embeddings, especially strong for Hebrew retrieval.
- **OpenAI (GPT‑4o & GPT‑4o‑mini)**
  - `gpt-4o` for **generation** (high‑quality Hebrew answers)
  - `gpt-4o-mini` for fast/cheaper tasks (standalone query rewriting, HyDE)
- **Streamlit** – chat UI with CSS customization for RTL and streaming responses.
- **Pydantic & pydantic-settings** – type‑safe configuration management from `.env` with validation.

---

<a id="features"></a>
## ✨ Advanced Features (Extra Points)
- **Session memory with a checkpointer:** Persist conversation state per user (**thread ID**) to support continuous dialogue.
- **Contextualize Query:** Rewrite follow‑up questions into a clear standalone query before vector search.
- **Streaming responses:** Token‑by‑token UI streaming to reduce perceived latency.
- **Citations / transparency:** Display the source chunks + similarity score inside a UI expander.

---

<a id="getting-started"></a>
## 🚀 Getting Started

### 1) Prerequisites
- Python **3.9+** (recommended 3.10+)
- An active **OpenAI** account (for LLM calls)
- An active **Pinecone** account (for the vector database)

### 2) Environment Setup
Clone the repo and install dependencies:

```bash
git clone <your-repo-link>
cd <your-repo-folder>

python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3) Environment Variables (.env)
Create a `.env` file in the project root and add:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=therapy-rag
```

> Additional settings can be configured in `src/config.py` (e.g., embedding model, thresholds, etc.).

### 4) Data Processing & Indexing
> 💡 To save you time, the original PDFs and processed chunks (JSONL) are already included under the `data` folder in the repository.

**(Optional) Re-run preprocessing:**
```bash
python -m src.preprocess
```

**Index to Pinecone:**
```bash
python -m src.index_embeddings
```

### 5) Run the App
Two usage options are provided:

**Option A — Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**Option B — CLI for quick tests/debugging:**
```bash
python -m src.cli_chat --thread-id user_1
```

---

<a id="notes"></a>
## Important Notes
- **RTL in the UI:** RTL is applied via custom CSS. If you see mixed directionality, make sure the CSS is loaded and that no component forces LTR.
- **Hebrew retrieval quality:** The embedding model (`bge-m3`) is critical for accurate retrieval over Hebrew content.
- **Safety:** Even with guardrails, this is not a substitute for professional care. In urgent situations, consult a qualified professional.

---

### Credit
The knowledge base relies on content from the **Betipulnet** portal.
