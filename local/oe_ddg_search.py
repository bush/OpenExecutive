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

import itertools
from datetime import date

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("oe-ddg-search")

# Citation markers, not raw URLs.
#
# Measured: this model cited 0/4 runs when citing meant writing a URL into
# prose, having genuinely read the pages. In research_eval it scored 3/3 on
# citations -- where a source was a short bracket marker like [D1]. So the
# failure looks like FORMAT, not discipline: it will carry a token, it will not
# carry a URL.
#
# Markers are handed out from one monotonic counter per server process rather
# than restarting per call, because a turn usually runs several searches and
# per-call numbering would make [1] mean a different page each time. Numbers
# climbing across a long-lived process is harmless; collisions would not be.
_SOURCE_SEQ = itertools.count(1)

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
    for r in rows:
        url = r.get(url_key) or ""
        marker = f"S{next(_SOURCE_SEQ)}"
        line = [f"[{marker}] {r.get('title') or '(untitled)'}", f"    {url}"]
        meta = " | ".join(str(r[k]) for k in extra if r.get(k))
        if meta:
            line.append(f"    ({meta})")
        if INCLUDE_SNIPPETS:
            body = _snippet(r.get("body"))
            if body:
                line.append(f"    {body}")
        out.append("\n".join(line))

    body_note = (
        "No page contents are included above -- only titles and URLs. You cannot "
        "answer from this list."
        if not INCLUDE_SNIPPETS
        else "The text above is a preview snippet, not the page."
    )
    return (
        f"{_DATE_NOTE}\n\n"
        + "\n\n".join(out)
        + f"\n\n{body_note} Call `fetch` on each URL you intend to rely on.\n\n"
        "CITING: each result above has a marker like [S1]. When a sentence in "
        "your answer uses something you fetched, put that page's marker at the "
        "end of the sentence, e.g. 'Ottawa committed $36B over five years [S4].' "
        "Then finish your answer with a `Sources` section listing every marker "
        "you used and its URL, one per line. Use the markers -- do not write "
        "bare URLs in the body of the report."
    )


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
