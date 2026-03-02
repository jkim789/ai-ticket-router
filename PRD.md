# PRD: AI Ticket Router

## Project Overview

An AI-powered customer support ticket routing agent that classifies incoming messages, searches a knowledge base via RAG, and either auto-responds or routes to the appropriate human team with full context. Built as a portfolio project demonstrating production-grade AI engineering skills.

**Live Demo Target:** `https://tickets.cognurix.com`

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent Framework | LangGraph | Stateful multi-step agent orchestration |
| API Layer | FastAPI | REST API with async support + Swagger docs |
| Vector Database | ChromaDB | Embedding storage and similarity search |
| LLM Provider | Groq (Llama 3.3 70B) | Intent classification, response generation |
| Embeddings | OpenAI `text-embedding-3-small` | Document and query embeddings |
| Database | SQLite | Ticket history, routing logs, analytics |
| Containerization | Docker + Docker Compose | Multi-service orchestration |
| Deployment | Hostinger VPS + Cloudflare Zero Trust Tunnel | Edge deployment with Zero Trust security |
| Testing | pytest + pytest-asyncio | Unit and integration tests |
| CI/CD | GitHub Actions | Automated testing on push |

---

## Project Structure

```
intelligent-ticket-router/
├── PRD.md                       # This file
├── README.md                    # Public-facing documentation
├── Makefile                     # Common commands (build, run, test, deploy)
├── docker-compose.yml           # Multi-service orchestration
├── Dockerfile                   # App container
├── .env.example                 # Environment variable template
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata
├── src/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Settings and environment config (Pydantic BaseSettings)
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph state machine definition
│   │   ├── state.py             # TypedDict state definitions
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── classify.py      # Intent classification node
│   │   │   ├── search_kb.py     # Knowledge base RAG search node
│   │   │   ├── evaluate.py      # Confidence evaluation node
│   │   │   ├── respond.py       # Auto-response generation node
│   │   │   └── route.py         # Human routing node
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Embedding generation
│   │   ├── chunker.py           # Document chunking strategies
│   │   ├── vectorstore.py       # ChromaDB operations
│   │   └── retriever.py         # Retrieval pipeline with reranking
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── tickets.py       # Ticket processing endpoints
│   │   │   ├── knowledge_base.py # KB management endpoints
│   │   │   └── health.py        # Health check endpoint
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── requests.py      # Pydantic request models
│   │   │   └── responses.py     # Pydantic response models
│   │   └── dependencies.py      # FastAPI dependency injection
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py          # SQLite connection and session management
│   │   └── models.py            # SQLAlchemy models for tickets and routing logs
│   └── frontend/
│       └── index.html           # Single-page demo UI
├── data/
│   ├── seed/
│   │   └── knowledge_base/      # Seed FAQ/support documents (markdown files)
│   └── chroma/                  # ChromaDB persistent storage (gitignored)
├── scripts/
│   └── seed_kb.py               # Script to load seed documents into ChromaDB
└── tests/
    ├── __init__.py
    ├── conftest.py              # Shared fixtures
    ├── unit/
    │   ├── test_classify.py     # Intent classification tests
    │   ├── test_search_kb.py    # RAG search tests
    │   ├── test_evaluate.py     # Confidence evaluation tests
    │   ├── test_respond.py      # Response generation tests
    │   └── test_route.py        # Routing decision tests
    └── integration/
        ├── test_api.py          # FastAPI endpoint tests
        └── test_agent_flow.py   # End-to-end agent flow tests
```

---

## Architecture

### LangGraph State Machine

```
                    ┌─────────────────┐
                    │   START          │
                    │   (Raw Message)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ classify_intent  │
                    │                  │
                    │ Classifies into: │
                    │ - billing        │
                    │ - technical      │
                    │ - shipping       │
                    │ - general        │
                    │ - complaint      │
                    │ - refund         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ search_kb        │
                    │                  │
                    │ RAG retrieval    │
                    │ from ChromaDB    │
                    │ with metadata    │
                    │ filtering by     │
                    │ intent category  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ evaluate         │
                    │ _confidence      │
                    │                  │
                    │ Scores relevance │
                    │ of KB results    │
                    │ (0.0 - 1.0)     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                  │
              confidence >= 0.75  confidence < 0.75
                    │                  │
                    ▼                  ▼
           ┌───────────────┐  ┌───────────────┐
           │ generate      │  │ route_to      │
           │ _response     │  │ _human        │
           │               │  │               │
           │ Draft auto-   │  │ Create handoff│
           │ response using│  │ packet with:  │
           │ KB context    │  │ - Intent      │
           │               │  │ - Summary     │
           │               │  │ - KB docs     │
           │               │  │ - Suggested   │
           │               │  │   team        │
           │               │  │ - Priority    │
           └───────┬───────┘  └───────┬───────┘
                   │                  │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │      END         │
                   │  (Structured     │
                   │   Response)      │
                   └─────────────────┘
```

### Agent State Definition

```python
from typing import TypedDict, Optional, Literal
from datetime import datetime

class TicketState(TypedDict):
    # Input
    raw_message: str
    customer_id: Optional[str]
    channel: Optional[str]  # "web", "email", "whatsapp", "chat"
    timestamp: datetime

    # Classification
    intent: Optional[str]  # "billing", "technical", "shipping", "general", "complaint", "refund"
    sentiment: Optional[str]  # "positive", "neutral", "negative", "angry"
    urgency: Optional[str]  # "low", "medium", "high", "critical"

    # RAG Results
    kb_results: Optional[list[dict]]  # Retrieved documents with scores
    kb_confidence: Optional[float]  # 0.0 - 1.0

    # Output
    action: Optional[str]  # "auto_respond" or "route_to_human"
    auto_response: Optional[str]  # Generated response if auto-responding
    routing: Optional[dict]  # {"team": str, "priority": str, "summary": str, "context": list}

    # Metadata
    processing_time_ms: Optional[float]
    model_used: Optional[str]
    agent_trace: Optional[list[str]]  # Step-by-step trace for debugging/demo
```

---

## Phase Breakdown

### Phase 1: Project Scaffolding & Configuration

**Goal:** Set up the project structure, dependencies, Docker configuration, and basic FastAPI app.

**Tasks:**
1. Create the full directory structure as defined above
2. Create `requirements.txt` with all dependencies:
   - `fastapi>=0.115.0`
   - `uvicorn[standard]>=0.30.0`
   - `langgraph>=0.2.0`
   - `langchain-core>=0.3.0`
   - `chromadb>=0.5.0`
   - `groq>=0.12.0`
   - `openai>=1.50.0` (for embeddings)
   - `pydantic>=2.9.0`
   - `pydantic-settings>=2.5.0`
   - `sqlalchemy>=2.0.0`
   - `aiosqlite>=0.20.0`
   - `python-multipart>=0.0.9`
   - `python-dotenv>=1.0.0`
   - `httpx>=0.27.0`
   - `pytest>=8.0.0`
   - `pytest-asyncio>=0.24.0`
3. Create `config.py` using Pydantic BaseSettings for environment management
4. Create `main.py` with FastAPI app, CORS middleware, router includes, and lifespan events
5. Create `Dockerfile` (Python 3.11-slim base, non-root user, multi-stage build)
6. Create `docker-compose.yml` with app service and ChromaDB service
7. Create `.env.example` with all required environment variables
8. Create `Makefile` with targets: `build`, `run`, `dev`, `test`, `seed`, `deploy`
**Acceptance Criteria:**
- `make dev` starts the FastAPI app on port 8000
- `http://localhost:8000/docs` shows Swagger UI
- `http://localhost:8000/health` returns `{"status": "healthy"}`
- Docker build completes without errors

---

### Phase 2: LangGraph Agent Core

**Goal:** Build the LangGraph state machine with all nodes and conditional routing.

**Tasks:**
1. Define `TicketState` TypedDict in `state.py` as specified above
2. Implement `classify_intent` node in `nodes/classify.py`:
   - Uses LLM to classify intent and extract sentiment/urgency
   - Prompt should return structured JSON with intent, sentiment, urgency
   - Add result to agent_trace
3. Implement `search_kb` node in `nodes/search_kb.py`:
   - Queries ChromaDB with the raw message
   - Filters by intent category metadata
   - Returns top 5 results with similarity scores
   - Add result to agent_trace
4. Implement `evaluate_confidence` node in `nodes/evaluate.py`:
   - Scores the relevance of KB results (average similarity, content overlap)
   - Uses LLM to evaluate if the KB results actually answer the question
   - Sets `kb_confidence` float and `action` string
   - Add result to agent_trace
5. Implement `generate_response` node in `nodes/respond.py`:
   - Uses LLM with KB context to draft a customer response
   - Includes citations from KB documents
   - Sets `auto_response` in state
   - Add result to agent_trace
6. Implement `route_to_human` node in `nodes/route.py`:
   - Creates structured handoff packet
   - Maps intent to team: billing→billing_team, technical→engineering, shipping→logistics, complaint→escalation, general→general_support
   - Generates context summary for the human agent
   - Sets priority based on sentiment + urgency
   - Add result to agent_trace
7. Assemble the graph in `graph.py`:
   - Add all nodes
   - Add conditional edge after `evaluate_confidence` based on threshold
   - Compile the graph

**Acceptance Criteria:**
- Graph compiles without errors
- Can invoke graph with a test message and get structured output
- Agent trace shows all steps taken
- Conditional routing works (high confidence → auto-respond, low → route)

---

### Phase 3: RAG Pipeline

**Goal:** Build the document ingestion, embedding, and retrieval pipeline with ChromaDB.

**Tasks:**
1. Implement `chunker.py`:
   - Recursive text splitter with 512 token chunks, 50 token overlap
   - Support for markdown, plain text, and basic PDF text extraction
   - Preserves metadata (source file, section headers, category)
2. Implement `embeddings.py`:
   - Wrapper around OpenAI `text-embedding-3-small` (or alternative)
   - Batch embedding generation
   - Caching layer to avoid re-embedding unchanged documents
3. Implement `vectorstore.py`:
   - ChromaDB collection management (create, delete, list)
   - Document upsert with metadata (category, source, timestamp)
   - Similarity search with optional metadata filtering
   - Collection stats (document count, categories)
4. Implement `retriever.py`:
   - High-level retrieval pipeline
   - Query embedding → similarity search → optional reranking
   - Returns structured results with scores and metadata
5. Create seed documents in `data/seed/knowledge_base/`:
   - `billing.md` - Payment methods, refund policy, subscription management, billing disputes
   - `shipping.md` - Shipping times, tracking, international shipping, lost packages
   - `technical.md` - Account setup, password reset, API docs, troubleshooting
   - `returns.md` - Return policy, return process, exchanges, damaged items
   - `general.md` - Business hours, contact info, company policies, FAQ
   - Each file should have 15-20 realistic Q&A pairs
6. Create `scripts/seed_kb.py`:
   - Reads all seed documents
   - Chunks and embeds them
   - Loads into ChromaDB with proper metadata tagging

**Acceptance Criteria:**
- `python scripts/seed_kb.py` loads all documents into ChromaDB
- Similarity search returns relevant results for test queries
- Metadata filtering by category works
- Retrieval pipeline returns scored, structured results

---

### Phase 4: FastAPI Endpoints

**Goal:** Expose the agent and knowledge base through a clean REST API.

**Endpoints:**

#### `POST /api/v1/tickets/process`
Process a customer support ticket through the agent pipeline.

Request:
```json
{
  "message": "I was charged twice for my order #12345 and nobody is responding to my emails",
  "customer_id": "cust_abc123",
  "channel": "email"
}
```

Response:
```json
{
  "ticket_id": "tkt_uuid",
  "action": "route_to_human",
  "classification": {
    "intent": "billing",
    "sentiment": "angry",
    "urgency": "high"
  },
  "routing": {
    "team": "billing_team",
    "priority": "high",
    "summary": "Customer reports duplicate charge on order #12345. Multiple contact attempts without resolution. High urgency due to financial impact and customer frustration.",
    "relevant_kb_articles": [
      {"title": "Billing Disputes", "relevance_score": 0.82},
      {"title": "Refund Policy", "relevance_score": 0.76}
    ]
  },
  "kb_confidence": 0.68,
  "processing_time_ms": 2340,
  "agent_trace": [
    "classify_intent: billing (confidence: 0.95)",
    "search_kb: found 5 results, top score 0.82",
    "evaluate_confidence: 0.68 (below threshold, routing to human)",
    "route_to_human: billing_team, priority high"
  ]
}
```

#### `POST /api/v1/knowledge-base/ingest`
Upload and ingest a document into the knowledge base.

Request: `multipart/form-data` with file upload and category field.

Response:
```json
{
  "status": "ingested",
  "document_id": "doc_uuid",
  "chunks_created": 12,
  "category": "billing"
}
```

#### `GET /api/v1/knowledge-base/search`
Direct similarity search against the knowledge base.

Query params: `q` (search query), `category` (optional filter), `limit` (default 5)

#### `GET /api/v1/knowledge-base/stats`
Return knowledge base statistics.

Response:
```json
{
  "total_documents": 5,
  "total_chunks": 87,
  "categories": {
    "billing": 18,
    "shipping": 15,
    "technical": 22,
    "returns": 16,
    "general": 16
  }
}
```

#### `GET /api/v1/tickets/history`
Return processed ticket history from SQLite.

Query params: `limit` (default 20), `offset` (default 0), `intent` (optional filter)

#### `GET /health`
Health check returning service status and dependency health.

**Tasks:**
1. Create Pydantic request/response models in `api/models/`
2. Implement all route handlers in `api/routes/`
3. Add dependency injection for database sessions and ChromaDB client
4. Add proper error handling with HTTPException
5. Add request logging middleware
6. Configure CORS for the demo frontend

**Acceptance Criteria:**
- All endpoints return proper responses with correct status codes
- Swagger docs at `/docs` are complete and interactive
- Error cases return structured error responses
- Request/response validation works via Pydantic

---

### Phase 5: Database & Ticket History

**Goal:** Persist ticket processing results for history and analytics.

**Tasks:**
1. Define SQLAlchemy models in `db/models.py`:
   - `Ticket`: id, raw_message, customer_id, channel, intent, sentiment, urgency, action, auto_response, routing_team, routing_priority, routing_summary, kb_confidence, processing_time_ms, created_at
   - `KBDocument`: id, filename, category, chunk_count, created_at
2. Create async database session management in `db/database.py`
3. Save every processed ticket to the database
4. Implement the history endpoint with filtering and pagination

**Acceptance Criteria:**
- Tickets are persisted after processing
- History endpoint returns paginated results
- Filtering by intent works

---

### Phase 6: Demo Frontend

**Goal:** Build a clean, single-page demo UI that non-technical recruiters can use.

**Tasks:**
1. Create `src/frontend/index.html` — a single HTML file with embedded CSS and JS
2. **Design requirements:**
   - Clean, modern design (dark or light theme, professional look)
   - Company branding placeholder at top
   - Input section: textarea for customer message, optional customer ID, channel selector dropdown
   - "Process Ticket" button with loading state
   - Results section that appears after processing:
     - Classification badge (intent + sentiment + urgency with color coding)
     - Action taken (auto-respond or route) with visual distinction
     - If auto-responded: show the generated response in a chat-bubble style
     - If routed: show the routing card (team, priority, summary)
     - Expandable "Agent Trace" section showing step-by-step reasoning
     - KB articles used with relevance scores
   - Processing time displayed
   - Sample messages section with 4-5 clickable examples that auto-fill the input
3. **Sample messages to include:**
   - "I was charged twice for my order #12345 and nobody is responding" (billing, angry → route)
   - "How do I reset my password?" (technical, neutral → auto-respond)
   - "Where is my package? It's been 2 weeks" (shipping, frustrated → depends on KB)
   - "I'd like to return a damaged item I received yesterday" (returns, neutral → auto-respond)
   - "What are your business hours?" (general, neutral → auto-respond)
4. Serve the frontend at the root path `/`
5. Use vanilla JavaScript (no framework needed for a single page)

**Acceptance Criteria:**
- Frontend loads at root URL
- Can process tickets and see results
- Sample messages work as click-to-fill
- Mobile responsive
- Looks professional enough for a recruiter demo

---

### Phase 7: Docker & Deployment

**Goal:** Containerize and deploy to Hostinger VPS with Cloudflare Zero Trust.

**Tasks:**
1. Update `Dockerfile`:
   - Python 3.11-slim base
   - Non-root user
   - Copy requirements and install
   - Copy source code
   - Expose port 8000
   - CMD with uvicorn
2. Update `docker-compose.yml`:
   - App service binding to `127.0.0.1:8000`
   - ChromaDB service with persistent volume
   - SQLite volume mount at `./data`
   - Environment file reference
   - Restart policy
   - Health checks
3. Create deployment script or Makefile target for:
   - SSH into VPS
   - Pull latest code
   - Docker compose build and up
   - Run seed script if KB is empty
4. Configure Cloudflare Zero Trust Tunnel to `tickets.cognurix.com`
5. Set up UFW rules (same pattern as sentiment pipeline)

**Acceptance Criteria:**
- `docker compose up -d` starts all services
- Application accessible via Cloudflare tunnel
- Data persists across container restarts
- Health check passes

---

### Phase 8: Testing & CI

**Goal:** Add comprehensive tests and automated CI pipeline.

**Tasks:**
1. Create test fixtures in `conftest.py`:
   - Mock LLM client
   - Test ChromaDB collection (in-memory)
   - Test database session
   - Sample ticket data
2. Unit tests:
   - `test_classify.py`: Test intent classification with mocked LLM responses
   - `test_search_kb.py`: Test retrieval with pre-loaded test documents
   - `test_evaluate.py`: Test confidence scoring logic
   - `test_respond.py`: Test response generation with mocked context
   - `test_route.py`: Test routing logic for each intent → team mapping
3. Integration tests:
   - `test_api.py`: Test all FastAPI endpoints with TestClient
   - `test_agent_flow.py`: Test full agent pipeline end-to-end
4. Create GitHub Actions workflow `.github/workflows/ci.yml`:
   - Trigger on push to main and pull requests
   - Python 3.11 setup
   - Install dependencies
   - Run linting (ruff)
   - Run tests with coverage
   - Report coverage

**Acceptance Criteria:**
- All tests pass
- Coverage > 80%
- CI pipeline runs on push
- Tests use mocked LLM calls (no real API calls in tests)

---

### Phase 9: Documentation & Polish

**Goal:** Create professional documentation that showcases engineering quality.

**Tasks:**
1. Write `README.md` with:
   - Project title and one-line description
   - Architecture diagram (Mermaid)
   - Live demo link and API docs link
   - Tech stack badges
   - Features list
   - Quick start (local development)
   - API documentation summary
   - Project structure overview
   - Design decisions section explaining WHY each technology was chosen
   - Screenshots/GIF of the demo
2. Add inline code documentation:
   - Docstrings on all public functions
   - Type hints everywhere
   - Comments on non-obvious logic
3. Create `ARCHITECTURE.md` with:
   - System design overview
   - LangGraph agent flow explanation
   - RAG pipeline details
   - Scaling considerations
   - What would change in production (message queues, managed vector DB, etc.)
4. Record a demo GIF (30 seconds showing the full flow)

**Acceptance Criteria:**
- README is polished and professional
- All public functions have docstrings
- Architecture doc explains key decisions
- Demo GIF is embedded in README

---

## Quality Standards

These apply to ALL code in the project:

1. **Type hints on every function** — parameters and return types
2. **Docstrings on all public functions** — Google style
3. **Async everywhere** — all I/O operations must be async
4. **Pydantic for all data models** — no raw dicts for API data
5. **Proper error handling** — try/except with specific exceptions, HTTPException for API errors
6. **Logging** — use Python logging module, structured log messages
7. **No hardcoded values** — all config via environment variables through Pydantic BaseSettings
8. **Clean imports** — absolute imports, no wildcard imports
9. **DRY** — extract shared logic into utility functions
10. **Security** — no API keys in code, input validation, rate limiting consideration

---

## Environment Variables

```env
# LLM
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...           # For embeddings

# App
APP_ENV=development             # development | production
APP_PORT=8000
APP_HOST=0.0.0.0
LOG_LEVEL=INFO

# ChromaDB
CHROMA_HOST=chromadb
CHROMA_PORT=8001

# Agent
CONFIDENCE_THRESHOLD=0.75       # Below this → route to human
MAX_KB_RESULTS=5
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=llama-3.3-70b-versatile

# Database
DATABASE_URL=sqlite+aiosqlite:///./data/tickets.db
```

---

## Demo Company Persona

For the seed knowledge base, use the following fictional company:

**Company:** NovaTech Solutions
**Industry:** SaaS platform for small business management
**Products:** Invoicing, inventory management, CRM, online storefront
**Support channels:** Email, live chat, WhatsApp

This gives realistic context for all support categories (billing for subscriptions, technical for platform issues, shipping for storefront orders, etc.)
