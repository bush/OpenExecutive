#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp>=1.29,<2", "ddgs>=9.14"]
# ///
"""Keyless DuckDuckGo search as an MCP stdio server, for Open Executive.

Why this exists: Open Executive's built-in `web_search` is Anthropic's
server-side tool (`type: "web_search_20250305"`), and `providers/feature_gate.py`
strips any tool whose type starts with `web_search_`/`web_fetch_` before the
request reaches a local model -- the local server cannot execute it. Client-side
tools carrying an `input_schema` are NOT stripped, and MCP tools arrive in
exactly that shape, so this restores search for the local-only setup.

Deliberately thin: results are pure pointers -- title and URL, no page text (see
INCLUDE_SNIPPETS). The model spends its context on pages it chose to read via
the `fetch` tool (mcp-server-fetch) rather than on search boilerplate. That
two-tier split is what research_eval measured gemma4-26b handling cleanly: 3/3
facts, 0 fabrications, every deep source fetched, in 6 tool calls.

Run: uv run --script oe_ddg_search.py   (deps resolve from the header above)
"""

from __future__ import annotations

from datetime import date

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("oe-ddg-search")

# Snippets are pointers, not content. Long snippets crowd out the pages the
# model actually chose to read -- context is the binding constraint locally.
SNIPPET_CHARS = 300
MAX_RESULTS_CAP = 15

# Starve the snippets, so that reading a page is the only way to learn anything.
#
# Measured across three runs of the same broad research question, the local
# model searched, read the snippets, and wrote a confident report WITHOUT ever
# calling `fetch` -- so every run produced real, verifiable facts with zero
# citations, and no way to tell a sourced claim from a recalled one. Giving it
# more time did not change this (449s -> 177s -> 83s, still no fetch), and nor
# did instructing it to fetch, in both the system prompt and these tool
# descriptions. It is a follow-through failure, not a budget or wording one.
#
# With bodies withheld, a result is a title and a URL: enough to choose what to
# read, not enough to answer from. The model cannot route around a fact that is
# simply absent, which is why this is a tool-shaped fix rather than another
# instruction. `news` still carries source + date, which are selection metadata
# rather than content -- the model needs them to judge recency.
#
# Set back to True to restore snippets (and accept the uncited reports).
INCLUDE_SNIPPETS = False

# Why the date is stated everywhere below: a local model's training data ends
# well before today, so asked about "September 2026" it concluded the date was
# in the FUTURE and refused to search at all -- no tool call, no answer. Open
# Executive never puts the current date in the system prompt (checked), so the
# tool boundary is the only place we own that can correct it. It appears in
# both the tool DESCRIPTIONS (which the model reads when deciding whether to
# call) and in every RESULT (so it survives into the synthesis context).
#
# Computed at import, i.e. per server start. The gateway starts this as a
# subprocess when the API boots, so it is fresh on every restart; a process
# left running for days would drift by a day or two, which is harmless for
# "is this the future?" but is the reason not to quote it as authoritative.
TODAY = date.today().isoformat()

_DATE_NOTE = (
    f"TODAY'S DATE IS {TODAY}. Your training data ends before this, so events "
    f"you believe are in the future have already happened. Never refuse a "
    f"search because a date looks future-dated -- search and see."
)


def _clamp(n: int) -> int:
    return max(1, min(int(n), MAX_RESULTS_CAP))


def _snippet(text: object) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= SNIPPET_CHARS else s[: SNIPPET_CHARS - 1] + "…"


def _render(rows: list[dict], url_key: str, extra: tuple[str, ...] = ()) -> str:
    """Numbered, compact, URL-forward. Numbering lets the model say
    'fetch result 3' to itself and keeps the list scannable."""
    if not rows:
        return f"{_DATE_NOTE}\n\nNo results. Try different or broader search terms."
    out = []
    for i, r in enumerate(rows, 1):
        url = r.get(url_key) or ""
        line = [f"[{i}] {r.get('title') or '(untitled)'}", f"    {url}"]
        meta = " | ".join(str(r[k]) for k in extra if r.get(k))
        if meta:
            line.append(f"    ({meta})")
        if INCLUDE_SNIPPETS:
            body = _snippet(r.get("body"))
            if body:
                line.append(f"    {body}")
        out.append("\n".join(line))

    closing = (
        "No page contents are included above -- only titles and URLs. You cannot "
        "answer from this list. Call `fetch` on each URL you intend to rely on, "
        "read it, and cite the URLs you fetched."
        if not INCLUDE_SNIPPETS
        else "These are snippets only. To use any of this, call `fetch` on the "
        "URL to read the page, and cite the URLs you actually fetched."
    )
    return f"{_DATE_NOTE}\n\n" + "\n\n".join(out) + f"\n\n{closing}"


@mcp.tool(
    description=(
        f"Search the web for pages about a topic. {_DATE_NOTE} "
        "Returns titles and URLs ONLY -- no page text. You cannot answer from these "
        "results: they tell you what exists and where. Choose the relevant URLs and "
        "read them with the `fetch` tool, then cite what you fetched. "
        "Args: query (plain keywords beat a full question), "
        "max_results (1-15, default 8)."
    )
)
def search(query: str, max_results: int = 8) -> str:
    rows = DDGS().text(query, max_results=_clamp(max_results))
    return _render(rows, url_key="href")


@mcp.tool(
    description=(
        f"Search recent news. {_DATE_NOTE} "
        "Prefer this over `search` when the question is about current or recent "
        "events, because each result carries a publication date you can check for "
        "recency. Returns headlines, URLs, sources and dates -- but NO story text. "
        "Call `fetch` on a URL to read the story before relying on it. "
        "Args: query (terms describing the event or topic), "
        "max_results (1-15, default 8)."
    )
)
def news(query: str, max_results: int = 8) -> str:
    rows = DDGS().news(query, max_results=_clamp(max_results))
    return _render(rows, url_key="url", extra=("source", "date"))


if __name__ == "__main__":
    mcp.run()
