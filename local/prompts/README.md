# Prompt templates

Ready-to-post `/chat` payloads, kept because the exact wording turned out to
matter more than any model or config change during setup. All three ask the
same underlying question, so the differences between them are measurements
rather than opinions.

```bash
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d @local/prompts/all-specialists.json
```

## `all-specialists.json` — the one to copy

Produces a report with a section per specialist, all eight genuinely consulted.
Measured: 8/8 consulted, 4 pages read, 7917 chars, 548s, no timeout.

Needs `SPECIALIST_TIMEOUT_S` well above (specialists x per-call seconds) — on one
local GPU the consults serialize at ~45-60s each, so the upstream default of
180s kills the turn. 900 works.

Three clauses are load-bearing:

- **"call consult_specialist EIGHT TIMES IN ONE BATCH … do not consult them one
  at a time."** Without it the Executive consults one per iteration, runs out of
  steam after about five, and **writes the remaining sections itself** — a
  report with eight headings where five specialists ran.
- **"Only include a section for a specialist you actually consulted."** The
  backstop for the same failure.
- **"Answer now, in this message. Do not defer."** See below.

## `single-pass-research.json` — one perspective, ~90s

The Executive researches and answers alone. Use when you want an answer rather
than a panel; roughly a sixth of the wall time.

## `original-that-failed.json` — kept as a warning

The first attempt. It timed out at 300s having produced nothing, then on later
runs produced a deferral note instead of a report. It differs from
`single-pass-research.json` by exactly two sentences:

```
-  I'm going to bed so you can take your time.  I'll expect a big report report for tomorrow morning!
+  Answer now, in this message. Do not defer, do not promise a report for later,
+  and do not tell me what you are going to do - just do it and give me the finished report.
```

That swap alone took a run from 177s producing a 1982-char deferral ("I have
prepared a strategic report for your review tomorrow morning") to 83s producing
a 5717-char report — and incidentally stopped raw `thought` / `<channel|>`
render markers leaking into the answer, which were an artifact of the deferral
state rather than a separate bug.

**The transferable lesson: anything that sounds like permission to take time,
the model reads as permission not to finish.** "Take your time", "when you get a
chance", "no rush" are all invitations to defer on a turn-based harness with no
background worker.

## Judging a run

Read the debug events, not the prose. Every run in this series produced a
confident, well-structured report; they differed enormously in whether any of it
was grounded.

- `specialist_done` count should equal the number of sections.
- Citations `[S1]`, `[S2]` can only come from `read`, so their presence is
  evidence pages were opened (see `../README.md`).
- `turn_complete.timed_out` must be `false`.
