# Mnemo

> **Organizational memory MCP server** — local RAG over your private project knowledge, exposed to any IDE coding agent via the Model Context Protocol.

Mnemo gives your IDE agent (GitHub Copilot, Claude, Cursor, Continue, …)
access to two parallel knowledge collections:

- **`specs`** — what to build: user stories, ADRs, epics, BDD scenarios
- **`bug_memory`** — what already went wrong: resolved bugs with root cause + fix

The agent stays the same — Mnemo extends its context by serving private
data through standard MCP tools.

```text
┌─────────────────────┐         pull/delta            ┌─────────────────────┐
│  External sources   │ ◀──────────────────────────── │  Ingestion adapter  │
│  (Wiki, ADO, Jira,  │                               │  (pluggable)        │
│   Confluence, …)    │                               └──────────┬──────────┘
└─────────────────────┘                                          │ upsert
                                                                 ▼
                                                      ┌────────────────────┐
                                                      │   RAG store        │
                              MCP tools               │   Chroma | LanceDB │
                       ┌──────────────────────────────│   specs · bugs     │
                       │                              └────────────────────┘
                       ▼
              ┌──────────────────┐
              │  IDE Agent       │
              │  (GHCP, Claude…) │
              └──────────────────┘
```

---

## Why Mnemo

LLM-based coding agents hit a wall when your project has **private
conventions** — specific response shapes, error-handling patterns,
architectural rules that no Internet training set knows. They generate
plausible-but-wrong code that gets rewritten in review.

Mnemo gives them:
- **Private context**: specs, ADRs, decisions — your conventions, not generic ones
- **Institutional memory**: every resolved bug becomes a reusable lesson
- **Up-to-date knowledge**: refreshed on schedule, no model-cutoff staleness

All **local**: privacy preserved, zero per-query cost, offline-capable.

---

## Install

```bash
pip install mnemo                  # from PyPI (when published)
# or, for local development:
pip install -e .                   # from a clone of this repo
```

Optional backends:

```bash
pip install mnemo[lance]           # LanceDB with native hybrid search
pip install mnemo[llamaindex]      # LlamaIndex pipeline alternative
pip install mnemo[full]            # all of the above
```

Requires **Python 3.10+**. Wheels available for Windows, macOS, Linux (x64/ARM64).

---

## Quick start

```bash
# 1. Point Mnemo at your knowledge sources (folders for the lab; APIs in prod)
cp .env.example .env               # then edit MNEMO_SPECS_SOURCE_DIR, MNEMO_BUGS_SOURCE_DIR

# 2. Ingest both axes
mnemo-ingest all

# 3. Register the MCP server in your IDE (example for VS Code below)
mnemo-server                       # stdio-based MCP server
```

### Register in VS Code

Create `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "mnemo": {
      "type": "stdio",
      "command": "mnemo-server"
    }
  }
}
```

GitHub Copilot Chat (Agent mode) picks up the registration automatically
and exposes Mnemo's tools.

### Register in Claude Code

```bash
claude mcp add mnemo -- mnemo-server
```

---

## MCP tools exposed

| Tool | Purpose |
| --- | --- |
| `query_specs(question, k?)` | Semantic search over the specs collection |
| `get_spec(spec_id)` | Retrieve a spec/ADR by identifier (e.g. `"US-102"`) |
| `query_bugs(symptom, k?)` | Semantic search over the bug-memory collection |
| `get_bug(bug_id)` | Retrieve a bug by identifier (e.g. `"BUG-503"`) |
| `mnemo_info()` | Active configuration — handy for diagnostics |

Tools are **narrow and composable** so an agent can chain them during
multi-step reasoning (e.g. spec → related bugs → code generation).

---

## Configuration

All settings via env vars (prefix `MNEMO_`) or `.env` file:

| Variable | Default | Description |
| --- | --- | --- |
| `MNEMO_STORE` | `chroma` | Vector store: `chroma` or `lance` |
| `MNEMO_PIPELINE` | `default` | Pipeline: `default` (hand-rolled) or `llamaindex` |
| `MNEMO_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model identifier |
| `MNEMO_PERSIST_DIR` | `./data` | Local persistence directory |
| `MNEMO_SPECS_COLLECTION` | `specs` | Specs collection name |
| `MNEMO_BUGS_COLLECTION` | `bug_memory` | Bugs collection name |
| `MNEMO_SPECS_SOURCE_DIR` | `./examples/specs` | Specs source folder |
| `MNEMO_BUGS_SOURCE_DIR` | `./examples/bugs` | Bugs source folder |
| `MNEMO_CHUNK_SIZE` | `512` | Chunk size in tokens (cl100k_base) |
| `MNEMO_CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `MNEMO_TOP_K` | `5` | Default number of retrieved chunks |

---

## Adapting to your sources

The default ingestion adapters read from the file system — convenient for
local development and demos. **In production replace the adapter only.**

In [`src/mnemo/ingestion/specs_loader.py`](src/mnemo/ingestion/specs_loader.py)
and [`src/mnemo/ingestion/bugs_loader.py`](src/mnemo/ingestion/bugs_loader.py)
look for the `PRODUCTION NOTE` comment and swap `iter_*` with a call to:

- **Azure DevOps**: Wiki REST API (specs), Work Items REST API (bugs, filter `state=Resolved AND type=Bug`)
- **Confluence**: REST API for pages under a given space
- **Jira**: REST API for issues with resolution
- **GitHub**: Wiki / docs repo (specs), Issues API (bugs, filter `state=closed,label=bug`)
- **Notion / Linear / Sentry / Bugsnag**: respective REST/GraphQL APIs

The rest of the pipeline (chunking, embedding, store, MCP server)
**does not change**.

---

## Architecture & patterns

Mnemo is intentionally small — the source is ~600 LOC. Patterns worth
borrowing:

- **PEP 544 Protocols** for `VectorStore`, `Embedder`, `RagPipeline`,
  `SupportsHybridSearch` — third-party adapters conform structurally,
  no forced inheritance
- **Factory + lazy imports** for optional dependencies (`mnemo[lance]`,
  `mnemo[llamaindex]`): unused extras cost nothing
- **Runtime capability detection**: `isinstance(store, SupportsHybridSearch)`
  switches the pipeline to hybrid search automatically when LanceDB is selected
- **Multi-collection** behind a single shared embedder
- **Adapter pattern** for ingestion sources — the only place you change
  when moving from lab to production
- **pydantic-settings** for typed environment configuration

---

## Repo structure

```text
mnemo/
├── src/mnemo/
│   ├── core/                # Protocols + data models (PEP 544)
│   ├── ingestion/           # Pluggable source adapters
│   ├── embedders/           # FastEmbed adapter
│   ├── stores/              # Chroma + LanceDB adapters
│   ├── pipelines/           # DefaultPipeline + LlamaIndexPipeline
│   ├── factory.py           # build_system() → MnemoSystem
│   ├── mcp_server.py        # FastMCP server with 5 tools
│   ├── cli.py               # mnemo-ingest [specs|bugs|all]
│   ├── chunking.py          # langchain-text-splitters wrapper
│   └── config.py            # pydantic-settings
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Where Mnemo came from

Mnemo was extracted from the closing lab of the **GHCP Advanced course**
as a reusable asset. The lab uses Mnemo to demonstrate two concrete
use cases (spec-aware coding and bug-memory retrieval) on a simulated
"Project Tracker" codebase — see the lab repo for full materials.

---

## License

MIT.
