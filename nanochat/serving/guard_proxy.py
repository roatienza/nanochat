#!/usr/bin/env python3
"""Guard proxy in front of nanochat vllm: listens on 8000, upstream on 8010.

Why this exists: the harness client sends max_tokens=32768 and cannot protect
itself from a degenerate generation (the 'mmmm...' byte-BPE loop), which burned
16-21 min of GPU per request and made every subsequent request hang behind it.
vllm has no server-side completion cap that the client cannot override, so the
guard lives here:

  1. max_tokens is clamped to GUARD_MAX_TOKENS (client value is ignored).
  2. Upstream is ALWAYS streamed (abort power for every client shape);
     non-stream clients get the chunks reassembled into a full JSON response.
  3. Degenerate output (>=48 consecutive identical chars, or >=200 chars of
     only m/whitespace) aborts the upstream request mid-stream; the client
     still gets a clean, OpenAI-compatible finish chunk + [DONE]. Degenerate
     tool-call slots are dropped so the harness never executes garbage.
  4. Wall-clock cap GUARD_MAX_SECONDS aborts anything still running.
  5. Sampling parameters are passed through untouched (gates must hold).
  6. Everything else is a transparent proxy (SSE streaming preserved).

Aborts are logged to guard_aborts.log.
"""
import json
import os
import re
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("GUARD_UPSTREAM", "http://127.0.0.1:8010")
PORT = int(os.environ.get("GUARD_PORT", "8000"))
MAX_TOKENS = int(os.environ.get("GUARD_MAX_TOKENS", "4096"))
# Wall-clock cap. Calibrated against the real model: the slowest legitimate
# turn observed was 8.8s, so 45s leaves 5x headroom while still bounding a
# runaway generation long before the user's "it hangs" threshold.
MAX_SECONDS = float(os.environ.get("GUARD_MAX_SECONDS", "45"))
ABORT_LOG = os.environ.get("GUARD_LOG", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "guard_aborts.log"))
LAST_CONTENT_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "guard_last_content.txt")

RUN_RE = re.compile(r"(.)\1{47,}")  # any single char repeated 48+ times
RUN6_RE = re.compile(r"(.)\1{5,}")  # any single char repeated 6+ times (tool args)
TRAIL_M_RE = re.compile(r"[\s.]*m{6,}[\s.]*$")  # trailing degenerate run in content

# --- short-period cycle detection -------------------------------------------
# The 'mmmm' loop is period-1. The '\n\n\n' (literal backslash-n) loop is
# period-2: the byte-BPE alternates two chars, so a "same char repeated N
# times" test can never see it. Detect ANY short cycle.
CYCLE_PERIODS = range(2, 9)
CYCLE_MIN_REPS = 8
CYCLE_MIN_LEN = 32  # a full trailing cycle must be at least this long

# Characters that only ever appear in a degeneration loop, never in prose.
# A trailing perfect cycle is only called degenerate when its repeated block
# is made of these: a backslash (the literal '\n' pathology) or only
# m/whitespace/dot (the 'mmm' flood). This keeps legitimate repeated text
# (bullet lists, table rows, code) from being flagged.
_PATHOLOGY_CHARS = set("m \t\r\n./\\")


def _max_period_suffix(s: str, p: int) -> int:
    """Longest suffix of s that is a perfect repetition with period p."""
    n = len(s)
    if n < p:
        return 0
    k = p
    while k < n and s[n - 1 - k] == s[n - 1 - k + p]:
        k += 1
    return k


def cycle_run_len(s: str, min_reps: int = CYCLE_MIN_REPS) -> int:
    """Length of the trailing short-period cycle in s, 0 if there is none.

    A cycle of period p is p chars repeated >= min_reps times. Returns the
    number of trailing characters that belong to that cycle.
    """
    if not s:
        return 0
    best = 0
    for p in CYCLE_PERIODS:
        k = _max_period_suffix(s, p)
        if k >= p * min_reps and k > best:
            best = k
    return best


def _pathological_block(block: str) -> bool:
    """True when a repeated block looks like a degeneration artifact."""
    if not block:
        return False
    if "\\" in block:
        return True  # the literal '\n' pathology
    return all(c in _PATHOLOGY_CHARS for c in block)  # the 'mmm' flood


def _max_period_run(s: str, p: int) -> int:
    """Longest run anywhere in s that is a perfect repetition with period p.

    Returns the length of the longest such run (0 if none). The run need not
    be at the end of the string: a degeneration loop can sit inside a JSON
    tool-argument blob, e.g. {"city":"Oslo","note":"\\n\\n\\n\\n..."}.
    """
    n = len(s)
    if n < p:
        return 0
    best = run = 0
    i = 0
    while i + p <= n:
        # does a period-p repetition start at i?
        j = i + p
        while j < n and s[j - p] == s[j]:
            j += 1
        if j - i > best:
            best = j - i
        i = j if j > i else i + 1
    return best


# --- semantic repetition -----------------------------------------------------
# A third degeneration shape: the model repeats a short SEGMENT with a
# variable number of separators between copies, e.g.
#     "## 11<C>\n\n\n\n\n## 11<C>\n\n\n\n\n\n\n## 11<C>..."
# No perfect period exists (the gap varies), so a period test cannot see it.
# Normalise the separators away and look for identical segments in a row.
SEG_REPEAT_MIN = 4      # copies of the same segment in a row
SEG_MIN_LEN = 4        # the segment must be at least this long
# Split on REAL and LITERAL newlines only. Splitting on all whitespace would
# break "## 11<C>" into "##" and "11<C>" and hide the repetition.
_SEP_RE = re.compile(r"(?:\\n|\r\n|\n|\r)+")


def _segments(s: str) -> list:
    """Split into segments, treating real and literal newlines as separators."""
    return [g.strip() for g in _SEP_RE.split(s) if g.strip()]


def repeated_segment_run(s: str) -> int:
    """Length (in chars) of the trailing run of identical repeated segments.

    0 when the text does not repeat a segment in a row.
    """
    if not s:
        return 0
    segs = _segments(s)
    if len(segs) < SEG_REPEAT_MIN:
        return 0
    last = segs[-1]
    n = 1
    i = len(segs) - 2
    while i >= 0 and segs[i] == last:
        n += 1
        i -= 1
    if n < SEG_REPEAT_MIN or len(last) < SEG_MIN_LEN:
        return 0
    idx = s.rfind(last)
    return len(s) - idx


def low_entropy(s: str, window: int = 200, max_distinct: int = 12) -> bool:
    """True when the tail has very few distinct characters (a loop)."""
    if len(s) < window:
        return False
    tail = s[-window:]
    return len(set(tail)) <= max_distinct


# --- near-period repetition --------------------------------------------------
# A fourth degeneration shape: the model repeats a STRUCTURE whose content
# drifts slightly, e.g.
#   "## The wind is light...## The wind is blowing from the north..."
#   or "## See your best dates DryDay.com\n\n##\n## See your best dates..."
# The period is not exact, so an exact-match period test cannot see it. But
# the repeated frame (e.g. "## " or "## Heading\n") IS stable.
FRAME_MIN_REPS = 4
FRAME_MIN_LEN = 2


def _frame_candidates(s: str) -> list:
    """Candidate repeating frames: 1..6 chars that occur often enough."""
    from collections import Counter
    c = Counter()
    n = len(s)
    for p in range(1, 7):
        for i in range(0, n - p):
            c[s[i:i + p]] += 1
    return [(k, v) for k, v in c.items()
            if v >= FRAME_MIN_REPS and len(k) >= FRAME_MIN_LEN]


def near_period_repetition(s: str) -> bool:
    """True when a short frame recurs many times (a drifting loop).

    Counts every 1..6 char frame in the string; if some frame occurs a lot
    more often than the text's length would suggest for a natural answer,
    the model is looping. The threshold scales with the length so short
    answers are never flagged.
    """
    if not s or len(s) < 200:
        return False
    cands = _frame_candidates(s)
    if not cands:
        return False
    n = len(s)
    # A frame repeated k times in a loop occupies at least k*len(frame)
    # characters of a repeating structure. Require the frame to account for
    # a large share of the text.
    best = 0
    for k, v in cands:
        if v >= FRAME_MIN_REPS and v * len(k) >= 0.5 * n:
            best = max(best, v)
    return best >= FRAME_MIN_REPS


# --- literal-escape and heading-density signals ------------------------------
# Calibrated on 20 REAL model answers (18 good, 2 degenerate) plus a captured
# degenerate answer. Separation was clean: 0 good answers contained 4+ literal
# backslash-n pairs, all 3 degenerate ones did.
LIT_N_TOTAL_MIN = 4       # total literal '\n' pairs in the text
LIT_N_CONSEC_MIN = 3     # consecutive literal '\n' pairs
HEADING_DENSITY_MIN = 1.5  # '##' per 100 chars
SEG_MAX_FREQ = 3          # how often one segment may repeat before it is a loop
# A runaway answer is cut here. Calibrated against the real model: the
# longest legitimate answer observed was 839 chars and the training pads top
# out at ~2600, so 2000 is a safe ceiling that still stops a runaway long
# before the client's 32768-token budget.
MAX_ANSWER_CHARS = 2000   # a longer answer than this is a runaway loop


def count_literal_n(s: str) -> int:
    """How many literal backslash-n pairs the text contains."""
    return s.count("\\n")


def consecutive_literal_n(s: str) -> int:
    """Longest run of consecutive literal backslash-n pairs."""
    best = 0
    cur = 0
    i = 0
    n = len(s)
    while i < n - 1:
        if s[i] == "\\" and s[i + 1] == "n":
            cur += 1
            if cur > best:
                best = cur
            i += 2
        else:
            cur = 0
            i += 1
    return best


def heading_density(s: str) -> float:
    """'##' occurrences per 100 characters."""
    return s.count("##") / max(1, len(s)) * 100


def segment_diversity(s: str, min_segments: int = 12) -> bool:
    """True when the text is built from very few distinct segments.

    A looping answer like
        "### October 20, 2026 / - Neil: ... / - Zero: ... / ### October 20 ..."
    repeats a small set of segments over and over, so the number of
    DISTINCT segments stays tiny while the total count grows. Natural prose
    has nearly as many distinct segments as total ones.
    """
    segs = _segments(s)
    if len(segs) < min_segments:
        return False
    distinct = len(set(segs))
    return distinct <= max(3, len(segs) // 5)


def max_segment_frequency(s: str, min_len: int = 4) -> int:
    """How often the most common segment (>= min_len chars) repeats."""
    from collections import Counter
    c = Counter(x for x in _segments(s) if len(x) >= min_len)
    return max(c.values()) if c else 0


def is_degenerate(s: str, min_len: int = CYCLE_MIN_LEN) -> bool:
    """True when a stored/generated text has entered a degeneration loop.

    Covers period-1 runs (48+ identical chars), short-period cycles
    (period 2..8, >= 8 reps, >= min_len chars) such as the literal '\n'
    loop, semantic segment repetition with varying gaps, near-period
    (drifting) frame repetition, literal-escape spam, and the
    'm'/whitespace flood.
    """
    if not s:
        return False
    if RUN_RE.search(s):
        return True
    for p in CYCLE_PERIODS:
        k = _max_period_run(s, p)
        if k >= max(p * CYCLE_MIN_REPS, min_len):
            # the repeated block itself must look pathological
            if _pathological_block(s[k - p:k]):
                return True
    if repeated_segment_run(s):
        return True
    if near_period_repetition(s):
        return True
    # literal-escape spam: the model emitting '\n' as two characters
    if count_literal_n(s) >= LIT_N_TOTAL_MIN:
        return True
    if consecutive_literal_n(s) >= LIT_N_CONSEC_MIN:
        return True
    # heading-spam loop: "## heading\n\n## heading\n\n..." with drifting text
    if len(s) >= 200 and heading_density(s) >= HEADING_DENSITY_MIN:
        return True
    # a segment repeated many times (a list that loops over a few entries)
    if max_segment_frequency(s) >= SEG_MAX_FREQ:
        return True
    # runaway enumeration: "- Day 1: ... - Day 2: ..." where every line is
    # unique, so no repetition test can see it. Calibrated against the real
    # model: legitimate answers top out at ~840 chars and the training pads
    # at ~2600, so 4000 is a safe ceiling.
    if len(s) >= MAX_ANSWER_CHARS:
        return True
    if len(s) >= 200 and re.fullmatch(r"[\sm]{200,}", s[-200:]):
        return True
    return False


def _earliest_period_run(s: str, p: int, min_len: int) -> int:
    """Index of the earliest maximal period-p run of length >= min_len."""
    n = len(s)
    i = 0
    best = -1
    while i + p <= n:
        j = i + p
        while j < n and s[j - p] == s[j]:
            j += 1
        if j - i >= min_len:
            if best < 0 or i < best:
                best = i
            i = j
        else:
            i += 1
    return best


def first_degeneracy_pos(s: str) -> int:
    """Index where the degeneration begins, or -1 if the text is clean.

    Used to truncate an aborted generation to its clean prefix, so the
    client receives the part that was still coherent.
    """
    if not s:
        return -1
    n = len(s)
    best = -1

    def offer(pos):
        nonlocal best
        if pos >= 0 and (best < 0 or pos < best):
            best = pos

    # literal '\n' spam: the model emitting '\n' as two characters. A
    # SINGLE literal '\n' can be legitimate (code quoting), so only cut when
    # there are several, and cut at the first one.
    lit = s.count("\\n")
    consec = consecutive_literal_n(s)
    if lit >= LIT_N_TOTAL_MIN or consec >= LIT_N_CONSEC_MIN:
        offer(s.find("\\n"))

    # 48+ identical chars
    m = RUN_RE.search(s)
    if m:
        offer(m.start())

    # period-p cycles
    for p in CYCLE_PERIODS:
        offer(_earliest_period_run(s, p, max(p * CYCLE_MIN_REPS, CYCLE_MIN_LEN)))

    # heading-spam loop
    if n >= 200:
        step = 50
        for start in range(0, n - 200 + 1, step):
            w = s[start:start + 200]
            if w.count("##") / 200 * 100 >= HEADING_DENSITY_MIN:
                offer(start)
                break

    return best


def strip_degenerate_tail(s: str) -> str:
    """Remove a degenerate run from the END of a generated answer.

    Handles the 'm' flood (period-1), the literal '\n' cycle (period-2),
    heading spam and a long run of one char near the end.
    """
    if not s:
        return s
    s = TRAIL_M_RE.sub("", s)
    for p in CYCLE_PERIODS:
        k = _max_period_suffix(s, p)
        if k >= p * CYCLE_MIN_REPS and _pathological_block(s[-p:]):
            s = s[: len(s) - k]
            break
    # a repeated-segment run (varying gaps) at the tail
    seg_run = repeated_segment_run(s)
    if seg_run:
        s = s[: len(s) - seg_run]
    # a long run of one char anywhere in the trailing region
    for m in reversed(list(RUN_RE.finditer(s))):
        if len(s) - m.end() < 64:  # only if it is at the tail
            s = s[: m.start()]
            break
    return s


def scrub_messages(messages: list) -> tuple:
    """Replace degenerate assistant turns in the request with a short marker.

    A poisoned history is an in-context attractor: if a previous assistant
    turn ended in a degeneration loop, the model copies that shape. Dropping
    the whole degenerate turn (not just its tail) removes the attractor.
    Returns (new_messages, n_scrubbed).
    """
    out, n = [], 0
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content")
            if isinstance(content, str) and is_degenerate(content):
                n += 1
                mm = dict(m)
                mm["content"] = "[earlier answer truncated]"
                mm.pop("tool_calls", None)
                out.append(mm)
                continue
        out.append(m)
    return out, n

app = FastAPI()
http = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)
)


def _log(line: str) -> None:
    try:
        with open(ABORT_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def degenerate(window: str, full: str = "") -> bool:
    """True when the generated text has entered a degeneration loop.

    `window` is the last ~600 chars of the generation; `full` is the whole
    generation so far. Checks:
      - 48+ identical chars in a row (the 'm' flood, period-1)
      - a short-period cycle (period 2..8, >= 8 reps): catches the literal
        '\n' (backslash + n) loop, which is period-2 and invisible to a
        same-char test
      - a repeated segment / near-period frame repetition
      - literal-escape spam (several literal '\n' pairs) — this must be
        counted over the WHOLE generation, not the tail window, because the
        spam can start early and the window slides past it
      - a long stretch of only m's/whitespace
    """
    if RUN_RE.search(window):  # 48+ identical chars in a row
        return True
    for p in CYCLE_PERIODS:
        k = _max_period_run(window, p)
        if k >= p * CYCLE_MIN_REPS and _pathological_block(window[k - p:k]):
            return True
    if repeated_segment_run(window):
        return True
    if low_entropy(window):
        return True
    # literal-escape spam: count over the whole generation
    probe = full or window
    if count_literal_n(probe) >= LIT_N_TOTAL_MIN:
        return True
    if consecutive_literal_n(probe) >= LIT_N_CONSEC_MIN:
        return True
    # the observed byte-BPE pathology: a long stretch of only m's/whitespace
    if len(window) >= 200 and re.fullmatch(r"[\sm]{200,}", window[-200:]):
        return True
    return False


def _upstream_headers(request: Request) -> dict:
    skip = {"host", "content-length", "accept-encoding", "connection"}
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


def _est_prompt_tokens(messages: list) -> int:
    n = 0
    for m in messages or []:
        n += len((m.get("content") or "")) if isinstance(m, dict) else 0
        tc = m.get("tool_calls") if isinstance(m, dict) else None
        if tc:
            n += len(json.dumps(tc))
    return max(1, n // 4)


async def _guarded_stream(body: dict, headers: dict, t0: float):
    """Yield parsed upstream SSE chunk dicts under the guard rules.

    Final item, if any: {"__aborted__": reason, "__ntok__": n} or
    {"__status__": code, "__body__": bytes, "__ctype__": str} for upstream
    non-200s. Upstream's own [DONE] is consumed, never forwarded.
    """
    req = http.build_request(
        "POST", f"{UPSTREAM}/v1/chat/completions", json=body, headers=headers
    )
    resp = await http.send(req, stream=True)
    aborted = None
    window, ntok = "", 0
    full_text = ""
    try:
        if resp.status_code != 200:
            data = await resp.aread()
            yield {
                "__status__": resp.status_code,
                "__body__": data,
                "__ctype__": resp.headers.get("content-type", "application/json"),
            }
            return
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            p = line[6:].strip()
            if p == "[DONE]":
                return
            try:
                c = json.loads(p)
            except json.JSONDecodeError:
                continue
            texts = []
            for ch in c.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    texts.append(d["content"])
                for tc in d.get("tool_calls") or []:
                    a = (tc.get("function") or {}).get("arguments")
                    if a:
                        texts.append(a)
            if texts:
                ntok += 1
                chunk_text = "".join(texts)
                window = (window + chunk_text)[-600:]
                full_text = (full_text + chunk_text)[-20000:]
                if os.environ.get("GUARD_TRACE") and ntok % 50 == 0:
                    _log(f"  .. stream ntok={ntok} full_len={len(full_text)} "
                         f"chunk_len={len(chunk_text)}")
                if len(full_text) >= MAX_ANSWER_CHARS:
                    aborted = "runaway-length"
                    break
                elif degenerate(window, full_text):
                    aborted = "degenerate-output"
                    break
            if time.monotonic() - t0 > MAX_SECONDS:
                aborted = "wall-clock"
                break
            yield c
    finally:
        await resp.aclose()  # closing early makes vllm abort the sequence
    if aborted:
        _log(
            f"{time.strftime('%F %T')} ABORT {aborted} "
            f"after {time.monotonic() - t0:.1f}s tokens~{ntok}"
        )
        yield {"__aborted__": aborted, "__ntok__": ntok}


def _chunk(model: str, delta: dict, finish: str | None, cid: str, created: int):
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
        ],
    }


def _clean_tool_slots(slots: dict) -> list:
    """Drop tool-call slots whose arguments degenerated; keep clean ones."""
    out = []
    for _, s in sorted(slots.items()):
        args = s["args"]
        # any 6+ identical-char run in tool args is the byte-BPE pathology
        # (legit JSON args never look like that); also drop non-JSON starts
        # and short-period cycles (the literal '\n' loop)
        if RUN6_RE.search(args) or is_degenerate(args, min_len=16):
            continue  # degenerate slot: never hand garbage to the harness
        if args and not args.lstrip()[:1] == "{":
            continue
        out.append(
            {
                "id": s["id"] or f"chatcmpl-tool-{os.urandom(4).hex()}",
                "type": "function",
                "function": {"name": s["name"], "arguments": args},
            }
        )
    return out


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = json.loads(await request.body())
    except Exception:
        return JSONResponse(
            {"error": {"message": "invalid JSON body"}}, status_code=400
        )
    headers = _upstream_headers(request)
    asked = body.get("max_tokens") or body.get("max_completion_tokens") or MAX_TOKENS
    body.pop("max_completion_tokens", None)
    body["max_tokens"] = max(1, min(int(asked), MAX_TOKENS))
    client_stream = bool(body.get("stream"))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    body["stream"] = True  # always stream upstream: abort power for every shape
    body.pop("stream_options", None)  # reassemble usage ourselves
    model_name = body.get("model", os.environ.get("UTAK_MODEL_NAME", "utak"))
    # Scrub degenerate assistant turns from the REPLAYED HISTORY. A poisoned
    # history is an in-context attractor: the model copies the shape of the
    # previous (degenerate) answer. Dropping the whole turn fixes it; trimming
    # only its tail does not.
    if body.get("messages"):
        body["messages"], n_scrub = scrub_messages(body["messages"])
        if n_scrub:
            _log(f"{time.strftime('%F %T')} SCRUB {n_scrub} degenerate history turn(s)")
    est_prompt = _est_prompt_tokens(body.get("messages"))
    if os.environ.get("GUARD_TRACE"):
        _log(f"{time.strftime('%F %T')} REQ msgs={len(body.get('messages') or [])} "
             f"max_tokens={body.get('max_tokens')} stream={client_stream} "
             f"est_prompt={est_prompt}")
    t0 = time.monotonic()
    cid = f"chatcmpl-guard-{os.urandom(6).hex()}"
    created = int(time.time())

    content_parts: list[str] = []
    slots: dict = {}
    ntok, usage, fin = 0, None, None
    aborted = None
    g = _guarded_stream(body, headers, t0)
    err = None
    try:
        async for item in g:
            if "__status__" in item:
                err = item
                break
            if "__aborted__" in item:
                aborted = item["__aborted__"]
                ntok = item["__ntok__"]
                fin = "stop"
                break
            for ch in item.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    content_parts.append(d["content"])
                    ntok += 1
                for tc in d.get("tool_calls") or []:
                    ntok += 1
                    i = tc.get("index", 0)
                    s = slots.setdefault(i, {"id": None, "name": "", "args": ""})
                    if tc.get("id"):
                        s["id"] = tc["id"]
                    f = tc.get("function") or {}
                    if f.get("name") and not s["name"]:
                        s["name"] = f["name"]
                    if f.get("arguments"):
                        s["args"] += f["arguments"]
                if ch.get("finish_reason"):
                    fin = ch["finish_reason"]
            if item.get("usage"):
                usage = item["usage"]
    finally:
        await g.aclose()
    if err is not None:
        return Response(
            err["__body__"], status_code=err["__status__"],
            media_type=err["__ctype__"],
        )

    content = "".join(content_parts)
    tool_calls = _clean_tool_slots(slots)
    if aborted:
        # keep any real content, drop degenerate tails; never execute garbage
        content = strip_degenerate_tail(content)
        # Truncate at the FIRST degeneracy point so the client keeps the
        # coherent prefix instead of the whole flagged blob.
        pos = first_degeneracy_pos(content)
        if pos > 0:
            content = content[:pos].rstrip()
        elif aborted == "runaway-length" and len(content) > MAX_ANSWER_CHARS:
            # a runaway enumeration has no repetition to cut at: keep the
            # first few complete lines and stop.
            lines = content.split("\n")
            keep, total = [], 0
            for ln in lines:
                if total + len(ln) > 1500:
                    break
                keep.append(ln)
                total += len(ln) + 1
            content = "\n".join(keep).rstrip() + "\n\n[output truncated: answer ran away]"
        if not tool_calls and not content:
            content = "(generation stopped early: repetitive output detected)"
    if os.environ.get("GUARD_TRACE"):
        _log(f"{time.strftime('%F %T')} RESP aborted={aborted} ntok={ntok} "
             f"content_len={len(content)} litN={count_literal_n(content)} "
             f"degen={is_degenerate(content)} tools={len(tool_calls)} "
             f"wall={time.monotonic() - t0:.1f}s")
        try:
            with open(LAST_CONTENT_LOG, "a") as fh:
                fh.write(f"--- {time.strftime('%F %T')} aborted={aborted} "
                         f"len={len(content)}\n{content}\n")
        except OSError:
            pass
    if usage is None:
        usage = {
            "prompt_tokens": est_prompt,
            "completion_tokens": ntok,
            "total_tokens": est_prompt + ntok,
        }

    if client_stream:
        async def sse_gen():
            yield "data: " + json.dumps(
                _chunk(model_name, {"role": "assistant", "content": content},
                       None, cid, created)
            ) + "\n\n"
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    yield "data: " + json.dumps(
                        _chunk(
                            model_name,
                            {"tool_calls": [
                                {"index": i, "id": tc["id"], "type": "function",
                                 "function": tc["function"]}]},
                            None, cid, created,
                        )
                    ) + "\n\n"
            yield "data: " + json.dumps(
                _chunk(model_name, {}, fin or "stop", cid, created)
            ) + "\n\n"
            if include_usage:
                yield "data: " + json.dumps(
                    {"id": cid, "object": "chat.completion.chunk",
                     "created": created, "model": model_name, "choices": [],
                     "usage": usage}
                ) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return JSONResponse(
        {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": fin or "stop",
                }
            ],
            "usage": usage,
        }
    )


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]
)
async def passthrough(request: Request, path: str):
    headers = _upstream_headers(request)
    body = await request.body()
    try:
        resp = await http.request(
            request.method, f"{UPSTREAM}/{path}", content=body, headers=headers
        )
        return Response(
            resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/plain"),
        )
    except httpx.HTTPError as e:
        return JSONResponse(
            {"error": {"message": f"upstream unavailable: {e}"}}, status_code=502
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")