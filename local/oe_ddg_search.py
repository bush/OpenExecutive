#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp>=1.29,<2",
#   "ddgs>=9.14",
#   "httpx>=0.27",
#   "readabilipy>=0.2",
#   "markdownify>=0.13",
# ]
# ///
"""Keyless DuckDuckGo search as an MCP stdio server, for Open Executive.

Why this exists: Open Executive's built-in `web_search` is Anthropic's
server-side tool (`type: "web_search_20250305"`), and `providers/feature_gate.py`
strips any tool whose type starts with `web_search_`/`web_fetch_` before the
request reaches a local model -- the local server cannot execute it. Client-side
tools carrying an `input_schema` are NOT stripped, and MCP tools arrive in
exactly that shape, so this restores search for the local-only setup.

Three tools: `search`, `news` and `read`. Search results are pure pointers --
title and URL, no page text (see INCLUDE_SNIPPETS) -- so the model spends its
context on pages it chose to open with `read` rather than on search boilerplate.
That two-tier split is what research_eval measured gemma4-26b handling cleanly:
3/3 facts, 0 fabrications, every deep source fetched, in 6 tool calls.

`read` is also the only source of citation markers; see the `_SOURCE_SEQ`
comment for why that matters, and local/README.md for the measurements behind
every design choice here.

Run: uv run --script oe_ddg_search.py   (deps resolve from the header above)
"""

from __future__ import annotations

import itertools
import warnings
from datetime import date

import httpx
from ddgs import DDGS
from markdownify import markdownify
from mcp.server.fastmcp import FastMCP
from readabilipy import simple_json_from_html_string

warnings.filterwarnings("ignore")

mcp = FastMCP("oe-ddg-search")

# Citation markers, minted ONLY by `read`.
#
# Two measured failures produced this design, and they pull in opposite
# directions:
#
#   snippets withheld, cite URLs      -> 2 pages read, 0 citations
#   snippets withheld, cite [S] marks -> 0 pages read, 9 citations, ALL FALSE
#
# The first: the model reads pages but will not carry a URL into prose. In
# research_eval it scored 3/3 on citations where a source was a short bracket
# marker like [D1], so this is a FORMAT limit, not a discipline one.
#
# The second is the trap. Markers were handed out in the SEARCH listing, so the
# model could emit a citation for a page it had never opened -- and it did,
# citing nine sources it never read. A report that looks sourced and is not is
# worse than one that is visibly unsourced.
#
# So the marker is now minted by `read`, on a successful fetch, and nowhere
# else. There is no way to obtain a citation without having read the page. The
# format the model is good at, gated behind the work it was skipping.
#
# Markers are stable per URL (re-reading a page returns its existing marker) and
# monotonic per server process, so several searches in one turn cannot collide.
_SOURCE_SEQ = itertools.count(1)
_MARKERS: dict[str, str] = {}

# Page text is capped per call; local context is the binding constraint. A
# truncated read reports how to continue rather than silently losing the tail.
READ_CHARS = 8000

# A browser UA. Measured: CBC times out on the MCP default UA and serves fine
# on this one. These are user-initiated reads of specific URLs the person asked
# about -- the same pages they would open themselves -- not crawling.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

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


# Hosts that reliably return no extractable article text, so `read` can mint no
# marker for them and the model ends up with nothing it may cite.
#
# Measured on DuckDuckGo news: "Mark Carney Canada investment summit" returned
# msn.com for 10 of 10 results, while "Canada EU associate member" returned none
# -- so on some queries EVERY result was unreadable and the run produced an
# uncited report despite the model dutifully calling `read`. These are wrappers
# that render their article client-side; the underlying publisher is usually in
# the results too, one position lower, and is readable.
_UNREADABLE_HOSTS = ("msn.com", "news.google.com")

# Over-request, then filter, so dropping wrappers does not shrink the result
# list below what was asked for.
_OVERFETCH = 3


def _clamp(n: int) -> int:
    return max(1, min(int(n), MAX_RESULTS_CAP))


def _drop_unreadable(rows: list[dict], url_key: str, want: int) -> list[dict]:
    return [
        r
        for r in rows
        if not any(h in (r.get(url_key) or "") for h in _UNREADABLE_HOSTS)
    ][:want]


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
        line = [f"{i}. {r.get('title') or '(untitled)'}", f"    {url}"]
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
        "answer from this list, and it gives you nothing you may cite."
        if not INCLUDE_SNIPPETS
        else "The text above is a preview snippet, not the page."
    )
    return (
        f"{_DATE_NOTE}\n\n"
        + "\n\n".join(out)
        + f"\n\n{body_note} Call `read` on each URL you intend to rely on. "
        "`read` returns the page text together with a source marker like [S1], "
        "and that marker is the only citation you can use -- there is no way to "
        "cite a page you have not read."
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
    want = _clamp(max_results)
    rows = DDGS().text(query, max_results=want * _OVERFETCH)
    return _render(_drop_unreadable(rows, "href", want), url_key="href")


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
    want = _clamp(max_results)
    raw = DDGS().news(query, max_results=want * _OVERFETCH)
    rows = _drop_unreadable(raw, "url", want)
    if rows:
        return _render(rows, url_key="url", extra=("source", "date"))

    # Every news hit was an unreadable wrapper -- measured at 10/10 msn.com for
    # some queries. Silently returning them would send the model off to read
    # pages that yield no text and therefore no citation, which is exactly how
    # an uncited report gets produced while the model does everything right.
    # The web index carries the original publishers, so fall back to it.
    web = _drop_unreadable(DDGS().text(query, max_results=want * _OVERFETCH),
                           "href", want)
    if not web:
        return (
            f"{_DATE_NOTE}\n\nEvery result for this query was an aggregator "
            "wrapper (msn.com / news.google.com) with no readable article text, "
            "in both the news and web indexes. Nothing here could be read or "
            "cited. Try different search terms, or name the publisher you want."
        )
    return _render(web, url_key="href") + (
        "\n\n(The news index returned only aggregator wrappers for this query, "
        "so these are web results for the same terms.)"
    )


@mcp.tool(
    description=(
        "Read a web page and get its text, plus the source marker you must cite "
        "it by. This is the ONLY way to obtain page content, and the ONLY way to "
        "obtain a citation: search gives you URLs, `read` gives you what the page "
        "actually says. Returns a marker like [S1] -- put that at the end of any "
        "sentence built on this page, and list the markers you used with their "
        "URLs in a `Sources` section at the end of your answer. Long pages are "
        "truncated and tell you how to continue. "
        "Args: url (from a search result), start_index (default 0; pass the "
        "offset the previous call reported to read further)."
    )
)
def read(url: str, start_index: int = 0) -> str:
    try:
        resp = httpx.get(
            url, follow_redirects=True, timeout=30.0, headers={"User-Agent": _UA}
        )
    except Exception as exc:  # network, DNS, TLS, redirect loops
        return (
            f"Could not reach {url} ({type(exc).__name__}). Nothing was read, so "
            "there is no marker and nothing here may be cited. Try another source."
        )
    if resp.status_code != 200:
        return (
            f"{url} returned HTTP {resp.status_code}. Nothing was read, so there "
            "is no marker and nothing here may be cited. Try another source."
        )

    try:
        parsed = simple_json_from_html_string(resp.text, use_readability=True)
        title = parsed.get("title") or url
        text = markdownify(parsed.get("content") or "").strip()
    except Exception:
        title, text = url, ""

    if not text:
        return (
            f"{url} was reached but no article text could be extracted (it may be "
            "a video, a paywall, or a listing page). Nothing may be cited from "
            "it. Try another source."
        )

    # Mint the marker only now -- the page was genuinely fetched and parsed.
    marker = _MARKERS.get(url)
    if marker is None:
        marker = f"S{next(_SOURCE_SEQ)}"
        _MARKERS[url] = marker

    start = max(0, int(start_index))
    chunk = text[start : start + READ_CHARS]
    remaining = len(text) - (start + len(chunk))
    more = (
        f"\n\n[{remaining} characters not shown. To continue this page, call "
        f"read(url, start_index={start + len(chunk)}).]"
        if remaining > 0
        else ""
    )
    return (
        f"SOURCE MARKER: [{marker}]  -- cite anything you take from this page as "
        f"[{marker}]\nTITLE: {title}\nURL: {url}\n\n{chunk}{more}"
    )


if __name__ == "__main__":
    mcp.run()
