# Architecture

## System Overview

The AI Ticket Router is a multi-stage AI agent that processes customer support messages through a LangGraph state machine. Each message flows through classification, knowledge base search, confidence evaluation, and either auto-response generation or human routing.

```
┌──────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│   FastAPI     │────▶│ LangGraph│────▶│ ChromaDB │────▶│  SQLite  │
│   REST API    │     │  Agent   │     │  (RAG)   │     │ (History)│
└──────────────┘     └──────────┘     └──────────┘     └──────────┘
       │                   │
       │              ┌────┴────┐
       │              │  Groq   │
       │              │  (LLM)  │
       │              └─────────┘
  ┌────┴────┐
  │ Frontend│
  │  (HTML) │
  └─────────┘
```

---

## LangGraph Agent Flow

The agent is a directed graph with 5 nodes and one conditional branch:

```
START
  │
  ▼
classify_intent ──────────────────────────────────────┐
  │                                                    │
  │  Calls Groq LLM to extract:                       │
  │  • intent (billing/technical/shipping/etc.)        │
  │  • sentiment (positive/neutral/negative/angry)     │
  │  • urgency (low/medium/high/critical)              │
  │                                                    │
  ▼                                                    │
search_kb                                              │
  │                                                    │
  │  Queries ChromaDB with message text                │  State flows
  │  Filters by intent category                        │  through each
  │  Returns top 5 results with similarity scores      │  node as a
  │                                                    │  TypedDict
  ▼                                                    │
evaluate_confidence                                    │
  │                                                    │
  │  If no results or low similarity → route to human  │
  │  Otherwise, asks LLM to evaluate if KB articles    │
  │  actually answer the question                      │
  │                                                    │
  ├── confidence >= 0.75 ──▶ generate_response         │
  │                            │                       │
  │                            │  Uses KB context to   │
  │                            │  draft a professional │
  │                            │  customer response    │
  │                            │                       │
  │                            ▼                       │
  │                           END (auto_response)      │
  │                                                    │
  └── confidence < 0.75 ──▶ route_to_human             │
                              │                        │
                              │  Maps intent → team    │
                              │  Calculates priority   │
                              │  Generates summary     │
                              │                        │
                              ▼                        │
                             END (routing packet) ─────┘
```

### State Definition

All nodes read from and write to a shared `TicketState` TypedDict:

```python
class TicketState(TypedDict):
    raw_message: str              # Input message
    customer_id: Optional[str]
    channel: Optional[str]        # web, email, chat, whatsapp
    intent: Optional[str]         # Classified intent
    sentiment: Optional[str]      # Classified sentiment
    urgency: Optional[str]        # Classified urgency
    kb_results: Optional[list]    # Retrieved documents
    kb_confidence: Optional[float]# 0.0 - 1.0
    action: Optional[str]        # auto_respond or route_to_human
    auto_response: Optional[str]  # Generated response
    routing: Optional[dict]       # Team, priority, summary
    agent_trace: Optional[list]   # Step-by-step log
```

### Routing Logic

The `route_to_human` node assigns teams and calculates priority:

| Intent | Team |
|--------|------|
| billing | billing_team |
| technical | engineering |
| shipping | logistics |
| complaint | escalation |
| refund | billing_team |
| general | general_support |

Priority escalation rules:
- Critical urgency → always critical priority
- Angry sentiment + medium/high urgency → critical priority
- Angry sentiment + low urgency → high priority

---

## RAG Pipeline

### Document Ingestion

```
Markdown files → DocumentChunker → OpenAI Embeddings → ChromaDB
```

1. **Chunking** (`src/rag/chunker.py`): Splits documents into ~512 token chunks with 50 token overlap. Preserves section headers as metadata. Supports markdown, plain text, and PDF.

2. **Embedding** (`src/rag/embeddings.py`): Uses OpenAI `text-embedding-3-small` with batch processing and in-memory caching (MD5 hash keys).

3. **Storage** (`src/rag/vectorstore.py`): ChromaDB with `PersistentClient` for local dev or `HttpClient` for Docker. Documents stored with category metadata for filtered search.

### Retrieval

```
Query → ChromaDB similarity search (filtered by intent) → Top 5 results
```

The search node filters by the classified intent category, so a billing question only searches billing-tagged documents. This improves precision over searching the entire corpus.

---

## API Layer

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/tickets/process` | Process a ticket through the agent |
| GET | `/api/v1/tickets/history` | Paginated ticket history |
| POST | `/api/v1/knowledge-base/ingest` | Upload documents |
| GET | `/api/v1/knowledge-base/search` | Direct KB search |
| GET | `/api/v1/knowledge-base/stats` | KB statistics |
| GET | `/health` | Health check |

### Request Flow

```
HTTP Request
  → RequestLoggingMiddleware (assigns request ID, logs timing)
    → FastAPI route handler
      → Dependency injection (DB session, vectorstore)
        → LangGraph agent invocation
          → SQLite persistence
            → JSON response
```

### Dependency Injection

FastAPI's `Depends()` provides:
- `get_db_session()`: Async SQLAlchemy session from the connection pool
- `get_vectorstore()`: ChromaDB instance from `app.state`

---

## Database

SQLite with async access via `aiosqlite` and SQLAlchemy 2.0 async engine.

### Tables

**tickets**: Stores every processed ticket with classification results, routing decisions, and timing metrics. Supports filtering by intent and pagination.

**kb_documents**: Tracks ingested knowledge base documents with category and chunk count.

---

## Deployment Architecture

```
Internet → Cloudflare Zero Trust Tunnel → localhost:8000
                                              │
                                    ┌─────────┴─────────┐
                                    │   Docker Compose   │
                                    │                    │
                                    │  ┌──────────────┐  │
                                    │  │  FastAPI App  │  │
                                    │  │  (port 8000)  │  │
                                    │  └──────┬───────┘  │
                                    │         │          │
                                    │  ┌──────┴───────┐  │
                                    │  │   ChromaDB   │  │
                                    │  │  (port 8000) │  │
                                    │  └──────────────┘  │
                                    │                    │
                                    │  Volumes:          │
                                    │  • app_data (SQLite)│
                                    │  • chroma_data     │
                                    └────────────────────┘
```

Both services bind to `127.0.0.1` only — no direct internet exposure. Cloudflare handles TLS termination and access control.

---

## Production Considerations

If scaling beyond a single VPS, these components would change:

| Current | Production | Why |
|---------|-----------|-----|
| SQLite | PostgreSQL | Concurrent writes, connection pooling |
| ChromaDB (self-hosted) | Pinecone or Weaviate (managed) | Scalability, availability |
| Groq (free tier) | Anthropic Claude or GPT-4 | Higher rate limits, SLAs |
| Direct API calls | Task queue (Celery/Redis) | Async processing, retries |
| Single server | Kubernetes | Horizontal scaling, health management |
| File-based seed | Admin UI + API | Dynamic KB management |
| In-memory embedding cache | Redis | Shared cache across instances |

The current architecture is intentionally simple for a portfolio demo on a single VPS. The abstractions (dependency injection, async I/O, Pydantic models) make migration to production infrastructure straightforward.

---

## Evaluation & Scaling Strategy

### Model and Routing Evaluation

To avoid treating the agent as a black box, the project includes a small labeled dataset and an evaluation script:

- `evaluation/dataset.jsonl`: Hand-crafted examples with expected `intent` and `action` (auto_respond vs route_to_human).
- `scripts/run_evaluation.py`: Async runner that:
  - Initializes the real LangGraph pipeline (including ChromaDB and LLM).
  - Executes each example end-to-end.
  - Reports intent accuracy, routing accuracy, and per-class precision/recall.

Run locally with:

```bash
make eval
```

This is intentionally lightweight, but it establishes the pattern for:

- Adding more labeled data from production tickets over time.
- Tracking metrics over git commits or deployments.
- Comparing different confidence thresholds or prompt versions.

### Scaling the System

The path from single VPS to production cluster focuses on three dimensions:

- **Throughput**
  - Move from SQLite to PostgreSQL with connection pooling for concurrent writes.
  - Replace in-process calls with a task queue (e.g., Celery + Redis) for high-volume ticket processing.
  - Horizontally scale FastAPI instances behind a load balancer or Kubernetes.

- **Reliability**
  - Use a managed vector database (Pinecone/Weaviate) with replication instead of self-hosted ChromaDB.
  - Add circuit breakers and timeouts around LLM and vectorstore calls.
  - Persist evaluation metrics and key decision signals (intent, action, confidence) for offline analysis.

- **Model and Prompt Evolution**
  - Version prompts and thresholds explicitly (e.g., `PROMPT_VERSION`, `CONFIDENCE_THRESHOLD` history).
  - Store prompt/version metadata alongside each processed ticket to enable backtesting.
  - Experiment with different models (Groq, Anthropic, OpenAI) and compare evaluation metrics using the same dataset.

These steps turn the current demo into a system that can be measured, iterated on, and safely scaled as ticket volume and model complexity grow.
