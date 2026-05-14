# Mnemo — Overview

> Story, architecture, technical choices, and a practical guide to the
> "MCP server as a reusable asset" pattern. Companion document to the
> [README.md](README.md), which focuses on the quickstart.

---

## Table of contents

1. [The problem it solves](#1-the-problem-it-solves)
2. [The idea in one sentence](#2-the-idea-in-one-sentence)
3. [Architecture](#3-architecture)
4. [Anatomy of an MCP server](#4-anatomy-of-an-mcp-server)
5. [Tech stack and rationale](#5-tech-stack-and-rationale)
6. [Use cases](#6-use-cases)
7. [Building your own MCP server](#7-building-your-own-mcp-server)
8. [Patterns and principles adopted](#8-patterns-and-principles-adopted)
9. [Limits and when *not* to use it](#9-limits-and-when-not-to-use-it)
10. [Resources and next steps](#10-resources-and-next-steps)

---

## 1. The problem it solves

IDE coding agents (GitHub Copilot, Claude Code, Cursor, …) have become
powerful tools, but they hit a wall when the project has **private
conventions**:

- response shape decided by an internal ADR
- error-handling patterns specific to the team
- layered architectural rules no Internet training set knows
- bugs the team has already resolved that no newcomer has memory of

An agent without private context produces **plausible but disconnected
code**, which then gets rewritten in code review. Worse: bugs fixed
months ago get reintroduced by people who never saw them, because
institutional knowledge doesn't make it into new code.

The knowledge that matters — specs, ADRs, decisions, post-mortems —
lives in places the agent can't reach: Azure DevOps Wiki, Confluence,
Jira, separate documentation repos. The wall is sharp: knowledge in
house, agent inside the IDE, no bridge between the two.

Mnemo is the bridge.

---

## 2. The idea in one sentence

> *"Give the IDE agent organizational memory, accessed through an open
> standard, maintained by specialized CLI agents."*

Three pieces:

1. **Local RAG** — knowledge lives in a local vector store (Chroma or
   LanceDB). Private, on-disk, no SaaS.
2. **MCP server** — exposes the knowledge to the agent via typed tools
   over the Model Context Protocol.
3. **Agentic ingestion CLIs** — pull from the canonical source
   (Wiki/ADO/Jira/Confluence), normalize, index. Scheduled,
   autonomous.

The IDE agent **stays the same**. Mnemo extends its context by
serving private data through standard tools.

---

## 3. Architecture

```text
   ┌─────────────────────┐                    ┌──────────────────┐
   │  Azure DevOps Wiki  │                    │   Coding Agent   │
   │  Confluence / Jira  │                    │   (GHCP, Claude) │
   │  GitHub Issues      │                    └────────┬─────────┘
   └──────────┬──────────┘                             │ MCP tools
              │ pull/delta                             │
              ▼                                        ▼
   ┌─────────────────────┐    write       ┌──────────────────────┐
   │  Agentic CLI        │ ─────────────▶ │   RAG store (local)  │
   │  (scheduled)        │                │   Chroma / LanceDB   │
   │  • plan             │    read        │                      │
   │  • extract          │ ◀───────────── │   ┌──────────────┐   │
   │  • classify         │                │   │ specs        │   │
   │  • cross-ref        │                │   ├──────────────┤   │
   └─────────────────────┘                │   │ bug_memory   │   │
                                          │   └──────────────┘   │
                                          └──────────┬───────────┘
                                                     │ MCP tools
                                          ┌──────────┴───────────┐
                                          │   MCP Server         │
                                          │   (mnemo, stdio)     │
                                          └──────────────────────┘
```

### Three processes, fully decoupled

| Component | Role | When it runs |
| --- | --- | --- |
| **Ingestion CLI** | writes to the RAG store, reading from canonical sources | scheduled (cron / Task Scheduler / GitHub Actions cron) |
| **RAG store** | persistent, local, two separate collections | persistent on disk |
| **MCP server** | reads from the store, exposes tools to the IDE agent | spawned on demand by the IDE client |

They communicate **only through the store**. No synchronous call
between ingestion and consumer. It's a pubsub-style architecture:
ingestion publishes, the store acts as a persistent broker, the MCP
server is a lazy subscriber.

### Two parallel knowledge axes

The same pipeline is instantiated twice, behind a single shared
embedder:

- **`specs`** — what needs to be built: user stories, ADRs, BDDs, API
  contracts. Ingestion adapter: `specs_loader.py`.
- **`bug_memory`** — what already went wrong: resolved bugs with
  symptom, root cause, fix, files touched. Ingestion adapter:
  `bugs_loader.py`.

The IDE agent accesses them via two families of MCP tools:
`query_specs`/`get_spec` and `query_bugs`/`get_bug`.

Same pattern, two collections, two adapters. Adding a third axis
(e.g. `runbooks`, `compliance`, `post_mortems`) requires **only a new
ingestion adapter** — the server and the IDE don't change.

---

## 4. Anatomy of an MCP server

### What MCP is

The **Model Context Protocol** is the open standard championed by
Anthropic for exposing tools, resources, and prompts to AI clients
(models or agents). Conceptually: a USB-C for AI tools. You build
once, your server is usable by any client that speaks MCP — today
that's Copilot, Claude Code, Cursor, Continue, Zed, and more.

### The four elements of an MCP server

1. **Transport** — how it talks to the client. Two options:
   - `stdio` (default, local): the server is launched as a subprocess
     by the client, communicating via stdin/stdout in JSON-RPC. No
     network, no auth, no CORS.
   - `SSE` / `HTTP`: the server runs on a port; clients connect over
     the network. Necessary when the server lives on a different host
     or serves multiple clients.
2. **Tools** — functions the agent can call. Typed (input/output
   JSON Schema derived from Python type hints). Tools are the main
   extension mechanism.
3. **Resources** — data the agent can read on demand (files, records,
   snapshots). More passive than tools.
4. **Prompts** — predefined prompt templates the agent can invoke with
   parameters.

Mnemo uses exclusively **tools**. Resources and prompts are great in
other scenarios but for "expose a RAG" the tool model is the most
natural fit.

### Session lifecycle

```text
Client                        MCP Server (Mnemo)
   │                                  │
   │── spawn(stdio) ─────────────────▶│
   │                                  │  build_system(load_settings())
   │                                  │  └─ FastEmbedEmbedder()  ← load ONNX
   │                                  │  └─ ChromaStore() × 2    ← open ./data/
   │                                  │
   │── initialize ───────────────────▶│
   │◀── server_capabilities ──────────│
   │                                  │
   │── tools/list ───────────────────▶│
   │◀── [query_specs, get_spec, ...]──│
   │                                  │
   │── tools/call(query_specs) ──────▶│
   │                                  │  embedder.embed(question)
   │                                  │  store.search(vec, k=5)
   │◀── {hits: [...]} ────────────────│
   │                                  │
   │── ...more tool calls...─────────▶│
   │                                  │
   │── shutdown ─────────────────────▶│
   │                                  │  process exit
```

Mnemo's first `initialize` costs **~30 seconds** (ONNX model load,
initialization of the two Chroma clients). Subsequent `tools/call`
requests cost milliseconds.

### The `@mcp.tool()` decorator (real example from Mnemo)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mnemo")

@mcp.tool()
def query_specs(question: str, k: int | None = None) -> dict[str, Any]:
    """Semantic search across the project specs.

    Use this when implementing a feature: it surfaces the acceptance
    criteria, related ADRs, and any cross-references the spec carries.

    Args:
        question: Natural-language question or spec identifier.
        k: Optional override for the number of chunks to return.
    """
    result = _system.specs.query(question, k=k or _settings.top_k)
    return {"axis": "specs", "hits": [...]}
```

FastMCP introspects the type hints and auto-generates the JSON Schema
the agent sees. The docstring becomes the `description` visible to
the model. No other boilerplate.

---

## 5. Tech stack and rationale

### Vector store

| Backend | When |
| --- | --- |
| **Chroma** (default) | Flattest onboarding, embedded, persistent. Up to ~1M vectors. |
| **LanceDB** (optional) | Native hybrid search (vector + BM25), Rust-fast, modern Lance format. Scales further. |

Switching is one env variable. The `RagPipeline` detects the
`SupportsHybridSearch` capability at runtime via a PEP 544 Protocol
and changes paths without any change at the call site.

### Embeddings

**fastembed** (Qdrant) — ONNX-optimized embedding models, no PyTorch
(avoids 2+ GB of dependencies), instant cold start, vectors identical
to sentence-transformers within quantization tolerances.

### Chunking

**langchain-text-splitters** (standalone package, *not* full
LangChain). `RecursiveCharacterTextSplitter.from_tiktoken_encoder()`
with semantic separators (markdown headings → paragraphs → lines →
spaces). Token-aware (tiktoken `cl100k_base`).

### MCP layer

**MCP Python SDK + FastMCP** — Anthropic's official SDK. FastMCP is
the ergonomic front-end: define a tool with a decorator, and the
framework handles JSON-RPC, schema, transport, and error handling.

### Configuration

**pydantic-settings** — typed configuration from env / `.env`. Free
validation (ranges, `Literal` for enums, typed defaults).
Comprehensible errors when users supply invalid values.

### Interfaces

**PEP 544 Protocols** — structural subtyping for `VectorStore`,
`Embedder`, `RagPipeline`, `SupportsHybridSearch`. Third-party
adapters conform without inheriting from one of our ABCs.

Component-by-component deep dive (with explicit "why we picked this")
lives in
[docs/TECH-STACK-DEEP-DIVE.md of the lab repo](../session-6-rag-mcp-lab/docs/TECH-STACK-DEEP-DIVE.md)
if you have the workspace alongside.

---

## 6. Use cases

### Spec-aware coding

**Scenario:** a new dev implements a user story (`US-102`). Without
Mnemo, GHCP generates plausible code disconnected from the team's
Sentinel pattern, silently introducing the regression `BUG-502` had
already solved.

**With Mnemo:** GHCP calls `query_specs("US-102")`, discovers the
related ADR-002, defensively runs `query_bugs("filter assignee null")`,
finds BUG-502, and produces code that cites the sources in comments.

### Bug memory retrieval

**Scenario:** new bug report; symptom familiar to whoever was on the
team six months ago, unknown to everyone else. Without Mnemo: hours
of debugging. With Mnemo: `query_bugs("kanban revert drag drop")`
returns `BUG-503` with full root cause and files_touched. The fix is
applied in the same consolidated shape.

### Performance regression triage

**Scenario:** P99 in production breaches SLO on an endpoint. Mnemo
recovers the equivalent historical bug, *and* cross-references with
the current code: "does the migration for that fix exist on this
branch?". If yes → investigate bloat. If no → generate the missing
migration. **This is real agentic reasoning**: bug memory × code
state.

### Generalizations

The same skeleton scales to:

- **Compliance & policy** — GDPR rules, audit, security
- **OnCall runbooks** — operational procedures recovered during an incident
- **Aggregated post-mortems** — knowledge from resolved incidents
- **Dependency intelligence** — changelogs + breaking changes of your deps
- **Customer feedback** — recurring patterns in support tickets

In all of these, only the **ingestion adapter** changes. MCP server,
embedder, store, MCP tools: unchanged.

---

## 7. Building your own MCP server

### When it's worth it

It's worth it when you have private knowledge the AI cannot otherwise
have access to:

- Voluminous or fragmented internal documentation across multiple systems
- Team's technical history (bugs, post-mortems, design decisions)
- Data schemas, API contracts, domain-specific rules
- Internal tools (ticketing, monitoring) the agent could pilot

It's not worth it when:

- The data is already accessible via standard tools (e.g. files in the
  repo, public GitHub Issues)
- The knowledge changes minute-to-minute and you can't maintain an
  ingestion pipeline
- The agent can get the same value with a single initial prompt

### The minimum skeleton (40 lines)

```python
# my_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")


@mcp.tool()
def hello(name: str) -> dict:
    """Greets the user.

    Args:
        name: who to greet
    """
    return {"greeting": f"Hello, {name}!"}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
```

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "my-server": {
      "type": "stdio",
      "command": "python",
      "args": ["my_server.py"]
    }
  }
}
```

Reload VS Code, and Copilot Chat in Agent mode sees the `hello` tool.
It works.

### From prototype to production: the checklist

1. **Typed and described tools**: typed signatures, clear docstrings
   (they're the "manual" the model reads).
2. **Lazy init**: don't load models/connections at module-import.
   Use an explicit `main()` and a factory.
3. **TTY-guard**: if the user launches the server from a terminal
   (stdin = TTY), exit with a friendly message instead of crashing on
   the first newline parsed as "JSON-RPC".
4. **Typed configuration**: pydantic-settings with a dedicated prefix.
5. **Narrow & composable tools**: 5 focused tools beat 1 monolithic
   tool with 20 parameters. The agent reasons better over tight tools
   it can compose, than over a mega-tool it has to configure.
6. **Stable schema**: tool names and signatures are an API. Breaking
   changes → consider a version bump and migration.
7. **Tests**: the server is a process. Test the logic as a library
   (mocking tool calls) — no need to spin up the actual stdio loop.

### Common pitfalls

- **Module-level side effects**: importing the module must *not* load
  models or open connections. Move that into `main()` or an explicit
  factory.
- **Tools returning long text**: the agent injects everything into the
  context window. Return structured JSON with focused fields, not
  multi-thousand-line markdown blobs.
- **Implicit auth**: stdio has no auth. If you need to serve a team
  over the network (SSE), handle auth explicitly — don't assume MCP
  does it.
- **Tools with invisible side effects**: if a tool writes to disk or
  calls external services, declare it in the `description`. Agents
  read the docstring.
- **Confusing tools and resources**: use resources for static data the
  agent reads "like a file"; use tools for operations with parameters
  or effects. Mixing them confuses the client.

---

## 8. Patterns and principles adopted

Patterns worth absorbing from the Mnemo codebase:

### PEP 544 Protocols (structural subtyping)

```python
@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks, embeddings) -> None: ...
    def search(self, embedding, *, k=5) -> list[Hit]: ...
```

Third-party adapters conform without forced inheritance.
`isinstance(x, VectorStore)` works at runtime thanks to
`@runtime_checkable`. More modern than ABC.

### Factory with lazy imports

```python
def _build_store(settings, *, dimension, collection) -> VectorStore:
    if settings.store == "chroma":
        from mnemo.stores.chroma_store import ChromaStore
        return ChromaStore(...)
    if settings.store == "lance":
        try:
            from mnemo.stores.lance_store import LanceStore
        except ImportError as exc:
            raise RuntimeError(
                'LanceDB requested but extras not installed. '
                'Run: pip install "mnemo[lance]"'
            ) from exc
        return LanceStore(...)
```

Optional dependencies cost nothing at import time. Actionable errors
when extras are missing.

### Runtime capability detection

```python
class SupportsHybridSearch(Protocol):
    def hybrid_search(self, embedding, query_text, *, k=5) -> list[Hit]: ...

# In the pipeline:
if isinstance(self._store, SupportsHybridSearch):
    hits = self._store.hybrid_search(embedding, question, k=k)
else:
    hits = self._store.search(embedding, k=k)
```

The same client code automatically picks a different path based on
what the backend supports. No explicit switch statement over backend
names.

### Adapter pattern for sources

All of Mnemo's "production-readiness" lives in *two comments*:

```python
# specs_loader.py
def iter_spec_files(source_dir: Path) -> Iterator[Path]:
    """PRODUCTION NOTE: replace with ADO Wiki / Confluence / Jira API."""
    yield from sorted(source_dir.rglob("*.md"))
```

The lab uses the file system; production replaces *only* this
function. Everything else (chunking, embedding, store, MCP) stays
unchanged.

### pydantic-settings with `Literal` for enums

```python
StoreName = Literal["chroma", "lance"]
class Settings(BaseSettings):
    store: StoreName = "chroma"
    chunk_size: int = Field(default=512, gt=0)
```

Free validation, comprehensible errors, IDE autocomplete. Replaces
the classic `os.environ.get(...)` + manual cast pattern.

---

## 9. Limits and when *not* to use it

Mnemo is designed for a specific range. Honest about what it **does
not** do:

- **It's not an agentic orchestrator**. It doesn't run autonomous
  loops pursuing goals. It performs retrieval on demand from the IDE
  agent.
- **It does not scale beyond ~10⁶ vectors per collection** on the
  default Chroma backend. For larger datasets: switch to embedded
  LanceDB, or move to a dedicated vector server (Qdrant, Weaviate,
  Pinecone).
- **It does not handle spec versioning**. It treats the latest
  ingestion as truth. Multi-version support is a natural extension
  (add a `version` metadata field) but not implemented.
- **It does not garbage-collect stale records automatically**.
  Ingestion upserts by ID; deletions at source are not propagated
  until you add an explicit cleanup policy (mark-and-sweep) in the
  adapter.
- **It does not automatically filter sensitive data**. If your specs
  contain credentials or PII, your ingestion adapter has to scrub
  them. Mnemo is not a Data Loss Prevention solution.
- **It is not a substitute for code review**. It is context, not
  validation. The agent can still produce wrong code, just with
  richer citations.

For each of these limits, extension patterns are well-trodden; none
is structural to the design.

---

## 10. Resources and next steps

### MCP

- Official spec: <https://modelcontextprotocol.io>
- Python SDK: <https://github.com/modelcontextprotocol/python-sdk>
- Anthropic reference servers: <https://github.com/modelcontextprotocol/servers>

### Mnemo

- Quickstart: [README.md](README.md)
- Source code: [`src/mnemo/`](src/mnemo/)
- Example corpus: [`examples/`](examples/)
- Test suite: [`tests/`](tests/) (run with `pytest`, ~98% coverage)

### Building something new on top of Mnemo

Forking is overkill for almost all cases. The recommended pattern:

1. **Add a new knowledge axis**: write a new adapter at
   `src/mnemo/ingestion/<my_axis>_loader.py`, register the collection
   in the `factory`, expose two tools in `mcp_server`.
2. **Change the source of an existing axis**: touch only the
   `iter_*_files` function of the adapter. ADO, Confluence, Jira,
   GitHub Issues: every other part of the code is untouched.
3. **Add a new store backend**: implement the `VectorStore` Protocol
   (and optionally `SupportsHybridSearch`), add a branch in the
   `_build_store` factory. No inheritance, no changes to the
   pipeline.

Mnemo is intentionally **small** (~600 LOC). If you find yourself
multiplying adapters or rewriting the pipeline, you're probably
building something different — and that's fine, but forking makes
sense then.
