# `local/` — running Open Executive with no Anthropic API key

Everything here exists to make Open Executive work on a **local model only**
(Ollama), with **no Anthropic key and no metered API of any kind**. Upstream
supports local models, but two capabilities quietly stop working, and this
directory restores one of them.

Branch: `local-only-no-api-key`, forked from upstream `f8922fc`.

## What breaks on a no-API-key deployment, and why

`providers/registry.py` gives local models `_LOCAL_FEATURE_SPEC`:

```python
supports_cache_control=False   # no prompt caching
supports_thinking=False        # SPECIALIST_EFFORT becomes a no-op
supports_web_search=False      # <-- the one that hurts
supports_tool_use=True         # <-- the one that saves us
```

So `ENABLE_CACHING` and `SPECIALIST_EFFORT` do nothing, and the Executive has no
web access: `providers/feature_gate.py:_strip_web_search_tools` removes any tool
whose `type` starts with `web_search_`/`web_fetch_` before the request leaves the
process, because a local OpenAI-compatible server cannot execute an
Anthropic-hosted server tool.

**But it only strips tools of that shape.** Client tools carrying an
`input_schema` pass through untouched, and MCP tools arrive in exactly that
shape. That is the whole basis of the workaround below.

## 1. Web search via MCP (`oe_ddg_search.py`)

A keyless DuckDuckGo MCP server exposing `search` and `news`, paired with the
official `mcp-server-fetch` for reading pages. Single file, PEP 723 header, so
`uv run --script` resolves its own dependencies — nothing to install or
maintain a venv for.

Copy `mcp_servers.json.example` to `company/mcp_servers.json`, fix the absolute
paths, restart the API. **The presence of that file is what enables MCP**
(`MCP_ENABLED` is inferred from it), so there is no extra env var to set.

Use absolute command paths. The MCP stdio client gives child processes only a
fixed env allowlist, so `~/.local/bin` is not reliably on their `PATH`.

**Why our own server for search but the official package for fetch:** fetching
is the hard half — `mcp-server-fetch` brings `readabilipy` + `markdownify` +
`protego`, and without real article extraction the model receives raw HTML that
destroys a local context window. Search is the easy half, and the popular
third-party DuckDuckGo MCP server hand-scrapes DDG with BeautifulSoup rather
than using `ddgs`, the maintained library that exists to absorb exactly that
breakage. So: buy the hard half, own the easy half.

**The date problem.** Open Executive never puts today's date in the system
prompt. A local model's training data ends well before now, so asked about
recent events it concludes the date is in the *future* and refuses to search at
all — no tool call, no answer. There is no upstream hook for this, so the date
is injected at the tool boundary instead: into the tool **descriptions** (read
when the model decides whether to call) and into every **result**. See the
`_DATE_NOTE` comment in `oe_ddg_search.py`.

## 2. `run_executive_research` removed from the Executive's tools

One-line change in `orchestrator/executive.py` (`_ALL_SKILL_TOOLS`), fully
commented at the site.

The research fan-out builds its specialists' tools in
`monitoring/research/specialist_research.py` as `[EMIT_RESEARCH_FINDINGS_TOOL]`
plus `build_web_search_tool()` — which is `None` here. **MCP tools are never
added**, because MCP dispatch lives only in `orchestrator/executive.py`. So the
fan-out has nothing to search with, while the Executive still prefers it for
research-shaped questions. Measured: 5 minutes across 7 specialists, nothing
returned.

Forbidding it in the system prompt was tried first and did not hold — it worked
for "what happened at X" and failed for "research X for me". Removing the tool
is structural and does not depend on phrasing.

Worth knowing: the fan-out's selling point is 7 specialists in parallel, but on a
single-GPU box they all queue behind one resident model, so even a fixed
fan-out would be minutes per turn.

## What is still missing

- **No parallel research fan-out.** Ask the angles as separate questions
  instead; each returns in ~90s rather than one 10-minute turn.
- **No prompt caching, no extended thinking.** Inherent to the local path.
- **The model does not always `fetch`.** It sometimes answers from search
  snippets alone, which produces a well-grounded report with no citations. The
  prompt asks it to fetch before asserting; it complies for narrow questions
  more reliably than for broad ones.

## Restoring upstream behaviour

Put `*RESEARCH_TOOLS,` back in `_ALL_SKILL_TOOLS` and delete
`company/mcp_servers.json`. Nothing else here modifies upstream files.
