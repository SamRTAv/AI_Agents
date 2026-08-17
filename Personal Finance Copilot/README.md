# Personal Finance Copilot

A LangGraph agent that answers personal-finance questions by combining
**retrieval** over curated Indian regulator publications with **tools** over the
user's own expense records — so advice is grounded in real rules *and* real
spending.

Served as a FastAPI application. Swagger UI at `/docs`.

The query the architecture exists for:

> *"I spent ₹17,749 in September. Am I overspending on food, and what should I
> be targeting?"*

Answering it requires the MCP expense tools **and** document retrieval, in that
order, composed into one answer.

---

## Status

| Phase | What | State |
|---|---|---|
| 1 | FastAPI shell, startup lifecycle, SQLite checkpointing, `/chat` | done |
| 2 | LangSmith tracing | done |
| 3 | MCP expense tools | done |
| 4 | RAG over curated corpus | done (228 chunks indexed) |
| 5 | Human-in-the-loop approval + `/resume` | written, needs a live run |
| 6 | SSE streaming (`/chat/stream`) | written, needs a live run |
| 7 | Eval set (50 cases) + runner | written, needs a baseline |
| 8 | Docker + deploy | deferred |
| 9 | Thin UI client | deferred |

**MCP note:** `langchain-mcp-adapters` 0.3.2 fails the stdio handshake on this
stack (`McpError: Connection closed`) while the raw `mcp` SDK completes it and
lists all 4 tools. `app/tools/mcp_client.py` therefore talks to the SDK
directly. Reproduce with `python scripts/diagnose_mcp.py`.

---

## Setup

```powershell
cd "Personal Finance Copilot"

python -m venv .venv
.venv\Scripts\Activate.ps1        # cmd: .venv\Scripts\activate.bat

pip install -r requirements.txt

copy .env.example .env
```

Then fill in `.env`:

| Variable | Where from |
|---|---|
| `GROQ_API_KEY` | console.groq.com/keys (free) |
| `LANGSMITH_API_KEY` | smith.langchain.com → Settings → API Keys (personal access token) |

**VS Code:** `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `.venv`.
Otherwise the editor reports every package as missing even though it runs fine.

### If pip fails on `langchain-community`

It pins `langchain-core<1.0`, which conflicts with the core 1.x that
langgraph 1.x needs. If that happens:

```powershell
pip uninstall langchain-community
pip install langchain-classic
```

`app/tools/vectorstore.py` tries both locations and will tell you which one it
found. Nothing else needs changing.

---

## Build the index (once)

```powershell
python scripts/ingest.py --report    # inspect the corpus, build nothing
python scripts/ingest.py             # build it
```

`--report` prints per-PDF character counts, duplication ratio, table count and a
verdict. Run it every time you add PDFs — it is how you catch a scanned or
defective document *before* it poisons retrieval.

First real run downloads the embedding model (~130 MB for `bge-small`).

Add more PDFs under `data/corpus/<topic>/` and re-run. The folder name becomes
the `topic` metadata the model cites. Biggest coverage gaps right now: **tax**
(Income Tax Dept Taxpayer Information Series) and **investing** (SEBI booklets).

---

## Run

```powershell
uvicorn app.main:app --reload
```

http://127.0.0.1:8000/docs

Check `GET /health` first — `tools_loaded` should be **5**:
`add_expense`, `list_expenses`, `summarize`, `list_categories`,
`search_finance_docs`. Fewer means something failed to load; the startup log
says which, and the app deliberately starts anyway rather than dying.

### Things to try

**Memory** — `POST /chat` twice on the same `thread_id`, then `GET /history/demo-1`.
Restart the server and call it again; it is still there.

**Expense tools**
```json
{"thread_id": "demo-1", "message": "How much did I spend in September 2025, and which category was biggest?"}
```
Expect ₹17,749 with food at 25.1%.

**Retrieval**
```json
{"thread_id": "demo-2", "message": "What is a Business Correspondent in banking?"}
```
The answer should cite the source document.

**The composed query** — the one that justifies the architecture
```json
{"thread_id": "demo-3", "message": "Based on my September 2025 spending, am I overspending on food compared to standard budgeting guidance?"}
```
Watch the LangSmith trace: `summarize` → `search_finance_docs` → answer. You
never scripted that order.

**Approval flow**
```json
{"thread_id": "demo-4", "message": "Add an expense of 450 rupees for an auto ride on 2025-09-28"}
```
Returns `status: "approval_required"`. **Kill the server now**, restart it, then:
```json
POST /resume  {"thread_id": "demo-4", "decision": "yes"}
```
It continues from exactly where it paused — the state was on disk, not in RAM.

**Streaming** (Swagger buffers SSE, so use curl):
```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d "{\"thread_id\":\"demo-5\",\"message\":\"What is compound interest?\"}"
```

---

## Tests — no API key needed

```powershell
python tests/smoke_phase1.py   # graph wiring, memory, thread isolation, persistence
python tests/smoke_phase3.py   # MCP over stdio: discovery, schemas, real tool calls
python tests/smoke_phase4.py   # RAG: index loads, passages carry citations
python tests/smoke_phase5.py   # approval: pause, restart, resume, decline, read-only passthrough
```

Phases 1, 3 and 5 substitute stub models, so they cost nothing and are
deterministic. Phase 4 needs the index built first.

---

## Evaluation

```powershell
python scripts/run_eval.py                  # all 50 cases
python scripts/run_eval.py --kind composed  # one category
python scripts/run_eval.py --limit 5        # quick check
```

Reports per category: tool recall, forbidden-tool avoidance, approval/refusal
behaviour, p50/p95 latency. Written to `evals/last_run.json`, traced in
LangSmith.

**This is the part that makes the project stand out.** Record a baseline, then
change exactly one thing and re-run:

| Knob | Where |
|---|---|
| Chunk size / overlap | `scripts/ingest.py` — `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Retrieval depth `k` | `app/tools/rag.py` — `TOP_K` |
| System prompt | `app/prompts.py` |
| Model / provider | `.env` — `LLM_PROVIDER`, `GROQ_MODEL` |

"Raised composed-query tool recall 62% → 89% by X" is worth more on a resume
than any feature you could add instead.

---

## Layout

```
app/
  main.py       FastAPI, startup lifecycle, /chat /resume /chat/stream /history
  config.py     every env-dependent choice (LLM + embedding provider, paths)
  graph.py      LangGraph assembly
  nodes.py      chat_node, approval_node, routing
  state.py      ChatState
  prompts.py    system prompt — a tuning surface
  schemas.py    request/response models; also the Swagger docs
  tools/
    mcp_client.py   spawns + discovers the expense server
    rag.py          FAISS retriever -> search_finance_docs
    embeddings.py   embedding factory (shared by ingest and serving)
    vectorstore.py  FAISS import shim across langchain packagings
mcp_server/     the expense MCP server (own process, no LangGraph awareness)
data/
  corpus/       source PDFs, by topic
  faiss_index/  built index (regenerate with scripts/ingest.py)
scripts/
  ingest.py     PDFs -> markdown -> chunks -> embeddings -> index
  run_eval.py   the eval harness
evals/
  dataset.json  50 labelled cases
tests/
```

---

## Design notes

**Startup vs. per-request.** Checkpointer, LLM, MCP subprocess and FAISS index
are built once in the `lifespan` handler and stashed on `app.state`. Handlers
only *use* what is loaded. Building any of it per request would re-pay the cost
on every message.

**Three separate memories.** Conversation history in `checkpoints.db` (per
`thread_id`); document knowledge in `data/faiss_index/` (shared, read-only);
expenses in the MCP server's own database. Distinct stores, distinct purposes.

**Why one tool is MCP and one isn't.** `search_finance_docs` closes over an
in-memory FAISS index — behind MCP the server would have to own that index, a
redesign rather than a decorator swap. The expense tools have no such tie and
are independently useful, so they live behind MCP. Point Claude Desktop at
`mcp_server/main.py` over stdio and it works unchanged — same server, different
client, zero code changes.

**Why approval lives in the graph, not the server.** `interrupt()` suspends the
LangGraph run via the checkpointer. `mcp_server/main.py` has no idea LangGraph
exists, so the gate sits in `approval_node` *before* `ToolNode` dispatches.

**Declining still answers every tool call.** Providers reject the next turn if
any `tool_call` lacks a matching `ToolMessage`, so a refusal injects one per
call rather than skipping them.

**Tool errors are content, not exceptions.** A bad date returns
`"Error ... must be in YYYY-MM-DD format (got '01-09-2025')"` as a normal tool
result, so the model reads it and retries instead of crashing the graph. That
makes error wording effectively a prompt — messages state the fix and echo the
bad value.

**Extraction was chosen by measurement.** `pypdf` reproduced a defective page 48
times (267k chars from one page) and flattened tables; `pymupdf4llm`
deduplicates and emits markdown tables, which also enables header-aware
chunking. `scripts/ingest.py --report` reproduces the numbers.

**Provider is one env var.** `LLM_PROVIDER=groq|openai` swaps the chat model
without touching graph code, so the eval set can A/B them. Groq serves no
embeddings, so `EMBEDDING_PROVIDER` is configured separately.

---

## Known gaps

- Corpus is 2 documents in `budgeting/`. Tax, insurance and investing are empty.
- No auth — anyone who can reach the API can read and write expenses.
- Single-user: expenses are one shared database, not per-`thread_id`.
- `MUTATING_TOOLS` in `app/nodes.py` is a hard-coded set; a new write tool must
  be added there or it bypasses approval.
- MCP tools are stateless — the adapter spawns the server per call. Simple and
  robust, but if latency matters, hold a persistent `client.session()` open.
