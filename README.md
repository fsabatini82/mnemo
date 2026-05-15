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
| `query_specs(question, k?, lifecycle?)` | Semantic search over the specs collection, with optional lifecycle filter |
| `get_spec(spec_id)` | Retrieve a spec/ADR by identifier (e.g. `"US-102"`) |
| `query_bugs(symptom, k?)` | Semantic search over the bug-memory collection |
| `get_bug(bug_id)` | Retrieve a bug by identifier (e.g. `"BUG-503"`) |
| `audit_spec(spec_id, deep?)` | Run drift checks against a spec — cheap by default, deep (LLM) when `deep=true` |
| `audit_spec_behavior(spec_id)` | LLM-powered behavior drift audit (deep only) |
| `audit_implemented_specs()` | Run cheap drift checks across all `lifecycle=implemented` specs |
| `mnemo_info()` | Active configuration — useful for diagnostics |

Tools are **narrow and composable** so an agent can chain them during
multi-step reasoning (e.g. spec → related bugs → drift audit → code generation).

---

## CLI tools

Mnemo ships with four executables registered as `[project.scripts]`:

| Command | Purpose |
| --- | --- |
| `mnemo-server` | MCP stdio server (launched by your IDE via `.vscode/mcp.json`) |
| `mnemo-ingest specs\|bugs\|all [--agentic copilot]` | Ingest a knowledge axis into the active project/env |
| `mnemo-admin` | Project registry + spec scaffolding (see below) |
| `mnemo-audit drift [--deep]` | Drift detection between specs and the working tree |

### `mnemo-admin` subcommands

```bash
mnemo-admin list-projects                              # tabulate registered projects + envs
mnemo-admin rename-project <old> <new>                 # rename slug, preserve numeric ID
mnemo-admin drop-project <slug> [--env <env>]          # drop all or one environment
mnemo-admin show-collection-names <slug> [--env <env>] # resolve actual collection names
mnemo-admin new-spec story --id US-205 --title "..."   # scaffold from a canonical template
mnemo-admin new-spec adr   --id ADR-007 --title "..."
mnemo-admin new-spec epic  --id EPIC-12 --title "..."
```

### `mnemo-audit` subcommands

```bash
mnemo-audit drift                                      # all specs, cheap checks
mnemo-audit drift --spec US-102                        # single spec
mnemo-audit drift --lifecycle implemented              # filter by lifecycle
mnemo-audit drift --deep                               # cheap + LLM behavior audit
mnemo-audit drift -o report.json                       # write JSON for CI gating
```

Exit codes: `0` clean (or only low/medium issues), `1` at least one
`high` severity drift, `2` invocation error.

---

## Configuration

All settings via env vars (prefix `MNEMO_`) or `.env` file:

| Variable | Default | Description |
| --- | --- | --- |
| `MNEMO_STORE` | `chroma` | Vector store: `chroma` or `lance` |
| `MNEMO_PIPELINE` | `default` | Pipeline: `default` (hand-rolled) or `llamaindex` |
| `MNEMO_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model identifier |
| `MNEMO_PERSIST_DIR` | `./data` | Local persistence directory |
| `MNEMO_PROJECT` | `demo-project` | Active project slug (lowercase, [a-z0-9-]) |
| `MNEMO_ENVIRONMENT` | `dev` | Environment scope: `dev` \| `col` \| `pre` \| `prd` |
| `MNEMO_SPECS_COLLECTION` | `specs` | Specs collection axis name |
| `MNEMO_BUGS_COLLECTION` | `bug_memory` | Bugs collection axis name |
| `MNEMO_SPECS_SOURCE_DIR` | `./examples/specs` | Specs source folder |
| `MNEMO_BUGS_SOURCE_DIR` | `./examples/bugs` | Bugs source folder |
| `MNEMO_CHUNK_SIZE` | `512` | Chunk size in tokens (cl100k_base) |
| `MNEMO_CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `MNEMO_TOP_K` | `5` | Default number of retrieved chunks |
| `MNEMO_CODE_ROOT` | `.` | Path to the code tree for drift auditing |
| `MNEMO_COPILOT_BIN` | `copilot` | Copilot CLI binary for agentic ingestion + deep audit |
| `MNEMO_COPILOT_ARGS` | `--no-stream` | Extra args to pass to the CLI |
| `MNEMO_COPILOT_STDIN` | `1` | `1` = prompt via stdin, `0` = positional |
| `MNEMO_COPILOT_TIMEOUT` | `180` | Per-call timeout in seconds |
| `MNEMO_AUDIT_FILE_BUDGET` | `6000` | Max chars per file injected into the deep-audit prompt |

The collection name actually used by the store is the tuple
`{project_id}_{env}_{axis}` (e.g. `001_prd_specs`). Project IDs are
assigned automatically and persisted in `<MNEMO_PERSIST_DIR>/projects.json`.

---

## Multi-tenant model (project × environment)

A single Mnemo install serves multiple projects across multiple
lifecycle environments. Each `(project, environment)` pair gets its own
isolated collections — no cross-contamination, but cross-env comparisons
remain trivial because everything lives in the same store.

- **Project slugs** (`alpha`, `demo-project`) are human-facing. The
  registry maps each slug to a stable 3-digit ID (`001`, `002`, …) so
  renaming a project is a one-line JSON edit, not a DB migration.
- **Environment** is constrained to the canonical four:
  `dev | col | pre | prd`. Anything else is rejected at config time.
- **Auto-register**: the first `mnemo-ingest` for a new `(project, env)`
  registers them automatically. `mnemo-server` is read-only — if you
  start it for an unknown project it falls back to a deterministic
  ephemeral ID and logs a clear warning.

```json
// .vscode/mcp.json — point each workspace at its own slice
{
  "servers": {
    "mnemo": {
      "type": "stdio",
      "command": "mnemo-server",
      "args": ["--project", "alpha", "--env", "prd"]
    }
  }
}
```

---

## Lifecycle vocabulary

Every spec carries a `lifecycle` metadata field with a canonical value:

| Value | Meaning |
| --- | --- |
| `proposed` | Drafted, not yet implementation-bound |
| `in-progress` | Actively being built |
| `implemented` | Shipped and present in the codebase |
| `superseded` | Replaced by a newer decision |
| `as-is` | Reverse-engineering notes on legacy code |

Ingestion is **permissive** (non-canonical values are stored as-is with
a warning); filtering is **strict** (the MCP tool rejects unknown
filter values). Pair this with `query_specs(lifecycle="implemented")`
to retrieve only what currently lives in the code, or
`lifecycle="proposed"` for the TO-BE roadmap.

---

## Spec templates

Three canonical templates ship with the package, each with required
sections per kind:

| Kind | Required sections |
| --- | --- |
| `story` | User Story · Acceptance Criteria · Test Scenarios (Happy / Error / Edge) · Acceptance Summary |
| `adr` | Context · Decision · Consequences · Acceptance Summary |
| `epic` | Goals · Constraints · Stories in scope · Acceptance Summary |

The deterministic loader and the Copilot agent both extract these
sections into flat metadata fields (`user_story`, `acceptance_criteria`,
`test_scenarios_happy/error/edge`, `acceptance_summary`, `context`,
`decision`, `consequences`, `goals`, `constraints`, `stories_in_scope`).
A `template_compliance` field is computed per spec — `full` /
`partial` / `non-compliant` / `n/a` — and surfaces as a drift signal.

Scaffold a new spec from a canonical template with
`mnemo-admin new-spec <kind> --id X --title Y`.

---

## Drift detection

The audit engine compares the spec corpus to the actual code on disk.
The cheap path runs in milliseconds with deterministic checks; the
deep path uses one LLM call per implemented spec.

| Signal | Detected by | Severity |
| --- | --- | --- |
| **Status** drift | Cheap | spec is `implemented` but declared `related_files` are missing from disk |
| **Coverage over** | Cheap | a declared file exists but never mentions the spec ID |
| **Coverage under** | Cheap | code files mention the spec ID but aren't listed in `related_files` |
| **Template** drift | Cheap | spec is `partial` or `non-compliant` against its canonical template |
| **Behavior** drift | Deep (LLM via Copilot CLI) | the code visibly takes a different branch than the spec describes |

Run via CLI for batch / CI:

```bash
mnemo-audit drift --lifecycle implemented -o report.json
# Exit code: 0 = clean, 1 = high drift, 2 = invocation error
```

Or via MCP tools from the IDE agent during refactoring sessions:

```text
@mnemo audit_spec("US-102")              # cheap only
@mnemo audit_spec("US-102", deep=true)   # cheap + LLM behavior check
@mnemo audit_implemented_specs()         # batch cheap audit
```

Severity hierarchy: `high` > `medium` > `low` > `none`. The reported
severity is the worst across all issues for the spec.

---

## Agentic ingestion (Copilot CLI, opt-in)

Beyond the deterministic file-parsing loaders, Mnemo can drive an LLM
agent via the GitHub Copilot CLI to extract structured metadata,
classify items as indexable vs noise, and normalize cross-references:

```bash
mnemo-ingest specs --agentic copilot      # per-spec LLM extraction
mnemo-ingest bugs  --agentic copilot      # per-bug normalization + classification
mnemo-ingest all   --agentic copilot
```

Prerequisites:

- Active GitHub Copilot subscription
- `gh auth login` completed (Copilot CLI uses cached OAuth token)
- A working `copilot` or `gh copilot` binary on PATH (override via
  `MNEMO_COPILOT_BIN`)

The deep drift audit (`audit_spec_behavior`, `audit_spec(..., deep=true)`,
`mnemo-audit drift --deep`) uses the same runner.

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
