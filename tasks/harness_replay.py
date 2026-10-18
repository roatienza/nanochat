"""
HarnessReplay task — teaches the model to answer the NEXT turn after its own
long technical answers are replayed as history (v5s touch-up).

Why this file exists
--------------------
Live probes against the served v5r with baby-onit's EXACT wire shape
(probe_harness_repro.py / probe_harness_ablation.py, Oct 18) reproduced the
user's report: with a resumed session (~51 turns, ~14k tokens) replayed as
user/assistant pairs, the model emits canned refusals ("I can only perform
tasks related to the functions provided"), document continuations ("The 0-5
day run is over", "revised version of your script") or "You're welcome!" —
instead of answering the new turn.

Ablations (two runs, consistent):
  * synthetic NEUTRAL history, same shape/length -> clean greeting
    => NOT length, NOT the pair shape, NOT the tool schemas.
  * last-2 real turns only -> still broken => the RECENT technical prose
    triggers it.
  * specific question over the same history -> answered fine => the model is
    register-confused, not destroyed; vague turns fall into canned attractors.
  * temp 0 -> identical canned output => attractor, not sampling noise.

Root cause: no training row ever shows the model's OWN long technical
assistant answers (RL-run reports, markdown tables, code blocks) replayed as
history, followed by a short vague user turn, with tools present. Stage-7b
history is short chit-chat (smoltalk) or tool rows that call immediately;
longctx rows are single-task synthetic docs. The harness shape — long
technical prose pairs + "hello" — is off-distribution, and a 2B model falls
into the nearest attractor (refusal / document continuation).

This task generates conversations in the EXACT harness shape:

  [system (harness-like, with today: DATE)]      (85% of rows)
  [user  task_1 + tools block]                   (tools in FIRST user turn,
  [asst  LONG technical answer 1]                 same as template v5)
  [user  task_2]
  [asst  LONG technical answer 2]
  ...
  [user  FINAL turn]                             <- the shape that breaks
  [asst  FINAL reply]                            <- supervised target

Final-turn shapes (the supervised behavior):
  H1 social-vague   "hello"/"how are you"/"thanks" -> short friendly reply,
                    NO call, NO refusal (history length 0..20 turns; 0 = the
                    fresh-session spurious-web_search case)
  H2 easy-knowledge "What is 2+2?" -> direct answer, no call
  H3 toolish->call  "weather today"/"news today"/"search for X" -> ONE
                    web_search call with sensible args (the harness's real
                    tool set has web_search, not get_weather)
  H4 date-from-sys  "date today" -> the date read from the system prompt's
                    "(today: DATE)" (harness injects it; model must use it,
                    not invent one)
  H6 toolish-no-call "What time zone is Cairo in?" -> prose from knowledge
  H7 summarize      "can you summarize that?" -> 1-2 sentence summary of the
                    LAST assistant answer

History register: the model's own answer style — markdown headers, tables,
code blocks, bullets, consistent numbers — on technical topics (training
runs, evals, deploys, GPU diagnostics, bug triage). User history turns are
technical questions AND pasted-log style turns. Some short answers mixed in.

All rows render well under max_seq_len 32768 (targets 2k-16k tokens), so the
v5r truncation-skip bug (mask sum 0 -> row silently dropped) cannot recur.
"""

import json
import random

from tasks.common import Task
from tasks.toolcalling import render_tools_prompt

# ---------------------------------------------------------------------------
# The harness's real tool set (baby-onit), as clean schemas. The harness's
# schema builder currently keeps only the last param per tool (verified
# harmless); we train on the clean schemas — behavior must not depend on it.
_HARNESS_TOOLS = [
    {"type": "function", "function": {"name": "web_search",
     "description": "Search the web. Use type='news' for dated/recent items.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "query (string)"},
         "max_results": {"type": "integer", "description": "max_results (integer)"},
         "type": {"type": "string", "description": "type (string)"}},
         "required": ["query"]}}},
    {"type": "function", "function": {"name": "fetch_content",
     "description": "Fetch a URL and return its text (handles PDFs).",
     "parameters": {"type": "object", "properties": {
         "url": {"type": "string", "description": "url (string)"}},
         "required": ["url"]}}},
    {"type": "function", "function": {"name": "bash",
     "description": "Run a shell command inside the working directory.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string", "description": "command (string)"}},
         "required": ["command"]}}},
    {"type": "function", "function": {"name": "write_file",
     "description": "Write text to a file (mode='append' to add).",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "path (string)"},
         "content": {"type": "string", "description": "content (string)"},
         "mode": {"type": "string", "description": "mode (string)"}},
         "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "read_file",
     "description": "Read a file's text (PDFs supported).",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "path (string)"},
         "mode": {"type": "string", "description": "mode (string)"}},
         "required": ["path"]}}},
    {"type": "function", "function": {"name": "edit_file",
     "description": "Replace an exact old_string with new_string.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "path (string)"},
         "old_string": {"type": "string", "description": "old_string (string)"},
         "new_string": {"type": "string", "description": "new_string (string)"}},
         "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "local_search",
     "description": "BM25 search over files in the working directory.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "query (string)"},
         "top_k": {"type": "integer", "description": "top_k (integer)"}},
         "required": ["query"]}}},
    {"type": "function", "function": {"name": "grep",
     "description": "Recursive regex search across files.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string", "description": "pattern (string)"},
         "path": {"type": "string", "description": "path (string)"},
         "file_pattern": {"type": "string", "description": "file_pattern (string)"}},
         "required": ["pattern"]}}},
]

# ToolStop's tool set (kept for mixture variety — behavior must hold for any
# tool list, not just the harness's).
_CLASSIC_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
        "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_web", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}},
        "required": ["query"]}}},
]

_EASY_QS = [
    ("What is 2+2?", "2+2 is 4."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by Jane Austen."),
    ("What is the largest planet in the solar system?", "Jupiter is the largest planet in the solar system."),
    ("What year did the Berlin Wall fall?", "The Berlin Wall fell in 1989."),
    ("How many continents are there?", "There are seven continents."),
    ("What is the square root of 144?", "The square root of 144 is 12."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
]

_SOCIAL_TURNS = [
    ("hello", ["Hello! How can I help you today?",
               "Hi! What can I do for you?",
               "Hello! Ready when you are — what do you need?"]),
    ("how are you", ["I'm doing well, thank you for asking! How can I help you today?",
                     "I'm good, thanks! What would you like to work on?"]),
    ("thanks", ["You're welcome! Anything else I can do?",
                "Happy to help! Let me know what's next."]),
    ("good morning", ["Good morning! What are we working on today?"]),
    ("what's up", ["Not much on my end — ready for your next task. What do you need?"]),
]

_TOOLISH_KNOWLEDGE = [
    ("What time zone is Cairo in?", "Cairo is in the Eastern European Time zone (UTC+2)."),
    ("How do stock splits work?",
     "A stock split divides each share into multiple ones, lowering the per-share price while keeping total value unchanged."),
    ("What does GDP stand for and measure?",
     "Gross Domestic Product measures the total value of goods and services an economy produces in a period."),
    ("What is the difference between a stock and a bond?",
     "A stock is ownership in a company; a bond is a loan to it that pays fixed interest."),
]

# ---------------------------------------------------------------------------
# Technical answer templates. Each returns (text, summary). Numbers are
# formatted from the row's rng so they stay consistent inside a row.
# Lengths matter: the real failure zone is 300-800-token answers replayed as
# history, so most templates emit long multi-section reports.
_ANSWER_TEMPLATES = [
    lambda r, t: (f"I ran the eval on {t['subject']} and the numbers moved the right way:\n\n"
                  f"| metric | before | after |\n|---|---|---|\n"
                  f"| val_bpb | {r.uniform(0.30, 0.43):.4f} | {r.uniform(0.28, 0.30):.4f} |\n"
                  f"| exact-match | {r.uniform(38, 52):.1f}% | {r.uniform(55, 71):.1f}% |\n"
                  f"| long-context | {'PASS' if r.random() < 0.7 else 'FAIL'} | PASS |\n\n"
                  f"## What changed\n- re-anchored the {t['noun']} rows so they fit the window\n"
                  f"- bumped epochs on the failing family\n"
                  f"- added a regression test so the silent skip can't recur\n\n"
                  f"## What I verified\n- the {t['noun']} gate at temp 0, 10 samples: {'0 over-calls' if r.random() < 0.8 else '2 over-calls'}\n"
                  f"- the 4-turn OpenAI loop end to end: correct\n"
                  f"- val bpb on the held-out split: {r.uniform(0.35, 0.45):.6f}\n\n"
                  f"## Residuals\n1. the 32k family still needs level-2 rows (queued)\n"
                  f"2. the parser artifact on parallel calls (pre-existing)\n\n"
                  f"Next step is a clean A/B against the current served checkpoint. "
                  f"Rollback exports are kept in the serve directory, and the logs "
                  f"carry the full gate output if you want to read them.",
                  f"The eval on {t['subject']} improved: val_bpb down, exact-match up."),
    lambda r, t: (f"## Diagnosis\n\nThe {t['noun']} failure is a {t['flaw']} — measured, not guessed:\n\n"
                  f"```\n{t['noun']}: {r.uniform(2, 9):.1f}x slower than expected\n"
                  f"mask sum: 0 on {r.randint(3, 12)} of {r.randint(40, 80)} rows\n```\n\n"
                  f"The {t['noun']} path recomputes state every call. Caching it drops "
                  f"the cost to near-zero; the fix is three lines in the loader.\n\n"
                  f"## Evidence\n- bisected layer by layer; the divergence starts at layer {r.randint(4, 20)}\n"
                  f"- the fp32 copy reproduces it, the bf16 copy does not\n"
                  f"- the log tail shows the same signature every time\n\n"
                  f"I'll patch it and re-run the suite. If the gate stays green I'll "
                  f"commit with the probe script so the diagnosis is reproducible.",
                  f"The {t['noun']} slowdown is a missing cache; the fix is three lines."),
    lambda r, t: (f"Deployed. The {t['subject']} is live and verified:\n\n"
                  f"1. packed and exported the checkpoint\n"
                  f"2. swapped the server on port 8000 ({r.choice(['GPU 0', 'GPU 5', 'GPU 7'])})\n"
                  f"3. ran the gate suite: {r.randint(5, 7)}/{r.randint(7, 7)} PASS\n\n"
                  f"## Verification\n| gate | result |\n|---|---|\n"
                  f"| 24k family | {'PASS' if r.random() < 0.8 else 'FAIL'} |\n"
                  f"| over-call | 0/10 |\n| 1-round answer | 5/5 |\n"
                  f"| 4-turn loop | correct |\n\n"
                  f"## Not changed\n- the parser (pre-existing artifact on parallel calls)\n"
                  f"- the sampler settings\n\n"
                  f"Rollback exports are kept. Logs are in the serve directory. "
                  f"The served name is unchanged so clients need no reconfiguration.",
                  f"The {t['subject']} deploy is live and gated."),
    lambda r, t: (f"Here's the script, with the two footguns removed:\n\n"
                  f"```bash\n#!/usr/bin/env bash\nset -u\ncd {t['dir']}\n"
                  f"exec .venv/bin/python -m scripts.{t['noun']}_runner --run {t['subject']} \\\n"
                  f"  --num-iterations {r.randint(30, 90)} \\\n"
                  f"  --max-seq-len {r.choice([8192, 32768, 131072])} \\\n"
                  f"  > logs/{t['noun']}.log 2>&1\n```\n\n"
                  f"## Why these flags\n- `set -u`: unset vars fail loudly\n"
                  f"- the log redirect keeps the buffer clean for the watcher\n"
                  f"- the iterations flag matches the touch-up budget\n\n"
                  f"It runs detached, writes to the log dir, and echoes EXIT so the "
                  f"status is checkable. If the run dies mid-way the log tail will "
                  f"show the last completed step, so a resume is cheap.",
                  f"Wrote the {t['noun']} runner script with logging and an exit code."),
    lambda r, t: (f"Short version: **{t['verdict']}**.\n\n"
                  f"- the {t['noun']} numbers are consistent across both runs\n"
                  f"- the {t['subject']} regression did not reproduce at temp 0\n"
                  f"- one residual remains: {t['flaw']} (queued, not blocking)\n\n"
                  f"## The numbers\n| run | bpb | match |\n|---|---|---|\n"
                  f"| first | {r.uniform(0.30, 0.42):.4f} | {r.randint(5, 8)}/9 |\n"
                  f"| rerun | {r.uniform(0.30, 0.42):.4f} | {r.randint(5, 8)}/9 |\n\n"
                  f"Nothing here changes the deploy decision. The one thing I'd watch "
                  f"is the 32k family: it needs level-2 rows, and until those exist "
                  f"the gate there is expected to stay red.",
                  f"Verdict: {t['verdict']}; one residual queued."),
    lambda r, t: (f"The {t['noun']} log shows the run is healthy:\n\n"
                  f"```\nstep {r.randint(20, 39)} | loss {r.uniform(0.2, 0.5):.4f} | lr {r.uniform(0.0001, 0.004):.6f}\n"
                  f"step {r.randint(40, 59)} | loss {r.uniform(0.15, 0.35):.4f} | lr {r.uniform(0.0001, 0.002):.6f}\n"
                  f"step {r.randint(60, 79)} | loss {r.uniform(0.12, 0.30):.4f} | lr {r.uniform(0.00005, 0.001):.6f}\n"
                  f"```\n\n"
                  f"Loss is descending smoothly, no spikes, memory flat at {r.randint(28, 38)} GB. "
                  f"Throughput is {r.randint(40, 90)}k tok/s, which matches the config. "
                  f"I'd let it run to the end of the schedule before touching anything; "
                  f"the checkpoint at step {r.choice([30, 60])} is already usable if you "
                  f"want to gate it early.",
                  f"The {t['noun']} run is healthy: loss descending, no spikes."),
    lambda r, t: (f"I compared the two checkpoints on {t['subject']}:\n\n"
                  f"| | A | B |\n|---|---|---|\n"
                  f"| val_bpb | {r.uniform(0.30, 0.40):.4f} | {r.uniform(0.30, 0.42):.4f} |\n"
                  f"| tool exact-match | {r.randint(6, 8)}/9 | {r.randint(4, 7)}/9 |\n"
                  f"| long-context | PASS | FAIL |\n"
                  f"| over-call gate | 0/10 | 3/10 |\n\n"
                  f"A wins on every axis that matters for the deploy; B's only edge is "
                  f"a metric we don't serve. The A/B was run at temp 0 with the same "
                  f"seeds, so the comparison is apples to apples. One caveat: the "
                  f"val split for the 24k family is small, so the bpb difference is "
                  f"within noise — the tool gates are the decisive axis.",
                  f"Checkpoint A beats B on {t['subject']} and long-context."),
    lambda r, t: (f"Done — {t['subject']} is fixed and committed.\n\n"
                  f"The bug: {t['flaw']} in the {t['noun']} path, so the row was silently "
                  f"skipped and the mixture never saw it. The fix:\n\n"
                  f"1. assert on the mask sum so a skipped row fails loudly\n"
                  f"2. a census print at startup showing per-family row counts\n"
                  f"3. a regression test that renders one row of each family\n\n"
                  f"Committed and pushed; the suite is green. The next touch-up should "
                  f"pick this up automatically since the census now prints what the "
                  f"mixture actually contains.",
                  f"Fixed the silent-skip bug in {t['noun']} and added a regression test."),
]

_SUBJECTS = ["stage7b-v5", "the multi_tool family", "the 24k eval", "the tool parser",
             "the longctx suite", "the RL run", "the export pipeline", "the gate suite",
             "the packing fix", "the over-call patch"]
_NOUNS = ["eval", "loader", "renderer", "trainer", "exporter", "sampler", "tokenizer", "probe"]
_FLAWS = ["a stale cache", "an off-by-one in the window", "a silent truncation",
          "a wrong default", "a race in the refill", "a mask sum of zero"]
_DIRS = ["/data/rowel/nanochat", "/data/rowel/sandbox", "/data/rowel/robot-decisions"]
_VERDICTS = ["ship it", "the fix holds", "deploy the new checkpoint", "the regression is gone"]

_USER_TECH = [
    "what did the eval show?",
    "why is the {noun} slow?",
    "compare the two checkpoints on {subject}",
    "write the runner script for {subject}",
    "analyze the log and tell me if the run is healthy",
    "is the {subject} regression fixed?",
    "deploy it and verify the gates",
    "summarize the {noun} results",
    "what's left before we can deploy?",
    "double-check the {noun} numbers",
]
_USER_PASTED = [
    "proceed to the next stage on the better checkpoint. it's better on val_bpb and identical on chat metrics.",
    "the suite failed again overnight — here's the tail of the log:\n```\nAssertionError: mask sum 0\n```\nwhat happened?",
    "run the full gate chain and give me a go/no-go.",
    "the report you wrote yesterday — can you extend it with the 32k numbers?",
]

_SYS_HARNESS = ("You are {name}, a helpful AI agent that completes tasks by calling tools.\n\n"
                "Working directory: {dir} (today: {today}). Every file path you write starts here.\n\n"
                "Tool routing:\n- Files in the working directory: read_file, grep, local_search.\n"
                "- Public facts, news, current events: web search; fetch pages for details.\n"
                "- Anything that can change needs a tool result, never memory.\n"
                "- Finish with a concise answer; cite the files or URLs you used.")
_SYS_SHORT = ["You are a helpful AI agent. Working directory: {dir} (today: {today}).",
              "You are a helpful assistant. Today is {today}.",
              "You are {name}, a concise technical assistant."]
_NAMES = ["baby-onit", "onit", "an agent", "a helpful assistant"]

_DATES = ["2026-10-18", "2026-10-17", "2026-10-16", "2026-10-12", "2026-09-30",
          "2026-10-01", "2026-10-05", "2026-10-09", "2026-10-14", "2026-10-15"]

_TOOLISH_CALLS = [
    # (final user turn, tool name, args, needs_type)
    ("weather today", "web_search", lambda r: {"query": "weather today"}, False),
    ("what's the weather like today?", "web_search", lambda r: {"query": "current weather"}, False),
    ("news today", "web_search", lambda r: {"query": "today's top news", "type": "news"}, True),
    ("any news this morning?", "web_search", lambda r: {"query": "morning news", "type": "news"}, True),
    ("search for the latest on the eval suite", "web_search", lambda r: {"query": "eval suite latest"}, False),
    ("look up when the next release ships", "web_search", lambda r: {"query": "next release date"}, False),
    ("read the README in the working directory", "read_file", lambda r: {"path": "README.md"}, False),
    ("what files are in the working directory?", "bash", lambda r: {"command": "ls -la"}, False),
    ("find where the loss is logged", "grep", lambda r: {"pattern": "loss"}, False),
]


def _est_tokens(text):
    return max(1, len(text) // 4)


class HarnessReplay(Task):
    """Conversations in the exact agent-harness replay shape: long technical
    assistant answers as history, tools present, final turn vague/simple/toolish."""

    def __init__(self, n_rows=6000, seed=909, max_tokens=16000, **kwargs):
        super().__init__(**kwargs)
        rng = random.Random(seed)
        self.examples = []
        for i in range(n_rows):
            ex = self._gen_row(rng, i)
            if ex is not None:
                self.examples.append(ex)
        self.length = len(self.examples)

    # ------------------------------------------------------------------
    def _mk_tools(self, rng):
        """85% of rows carry tools: the harness's 8 (70%), the classic 3 (15%),
        or a subset (15%). 15% of rows have no tools at all."""
        u = rng.random()
        if u < 0.15:
            return None
        if u < 0.55:
            return list(_HARNESS_TOOLS)
        if u < 0.70:
            return list(_CLASSIC_TOOLS)
        pool = _HARNESS_TOOLS if rng.random() < 0.7 else _CLASSIC_TOOLS
        k = rng.randint(2, len(pool))
        out = rng.sample(pool, k)
        rng.shuffle(out)
        return out

    def _mk_system(self, rng):
        u = rng.random()
        today = rng.choice(_DATES)
        if u < 0.45:
            return _SYS_HARNESS.format(name=rng.choice(_NAMES), dir=rng.choice(_DIRS), today=today), today
        if u < 0.75:
            return rng.choice(_SYS_SHORT).format(dir=rng.choice(_DIRS), today=today, name=rng.choice(_NAMES)), today
        if u < 0.85:
            return rng.choice(_NAMES).capitalize() + " speaking.", None
        return None, None  # no system message

    def _gen_history(self, rng, budget_tokens, today):
        """2-50 prior user/assistant pairs; technical register; returns msgs."""
        msgs = []
        used = 0
        n_turns = rng.choice([0, 2, 2, 3, 3, 4, 4, 5, 6, 8, 8, 10, 12, 16, 20, 24, 30, 40, 50, 60])
        if n_turns == 0:
            return msgs
        topic = {"subject": rng.choice(_SUBJECTS), "noun": rng.choice(_NOUNS),
                 "flaw": rng.choice(_FLAWS), "dir": rng.choice(_DIRS),
                 "verdict": rng.choice(_VERDICTS)}
        for t in range(n_turns):
            if rng.random() < 0.2:
                u = rng.choice(_USER_PASTED)
            else:
                u = rng.choice(_USER_TECH).format(noun=topic["noun"], subject=topic["subject"])
            if t == n_turns - 1 and rng.random() < 0.3:
                # last history answer short (the A2 shape: recent short-ish prose)
                a = rng.choice(["Done — the suite is green.", "Yes, the fix holds at temp 0.",
                                "The numbers are in the table above; nothing regressed."])
            else:
                a, _ = rng.choice(_ANSWER_TEMPLATES)(rng, topic)
                if rng.random() < 0.4:
                    # extend: real sessions carry 300-800-token answers
                    a += ("\n\n## Notes\n- " + rng.choice([
                        "the numbers above are at temp 0, single seed",
                        "the same gate ran clean twice in a row",
                        "the config is unchanged from the last deploy",
                        "the log dir has the full output if you want it",
                        "the rollback path is the previous export, untouched",
                        "one flag changed: the window size, documented in the log",
                    ]) + "\n- " + rng.choice([
                        "the next touch-up should pick this up automatically",
                        "the census print now shows what the mixture contains",
                        "the val split is small; treat the bpb delta as noise",
                        "the tool gates are the decisive axis, not bpb",
                        "the parser artifact is pre-existing and unrelated",
                        "the suite is green on both GPUs I tried",
                    ]) + "\n- " + rng.choice([
                        "happy to run the A/B again with more seeds",
                        "the watcher will flag it if the loss spikes",
                        "the gate output is appended to the results file",
                        "the served name is unchanged for clients",
                        "the checkpoint at the last save is usable",
                        "nothing here needs a code change",
                    ]))
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
            used += _est_tokens(u) + _est_tokens(a)
            if used > budget_tokens:
                break
        # E-anchor: 12% of histories END with a doc->CALL get_weather exchange
        # (the multi_tool shape; keeps the doc->CALL mapping anchored under the
        # harness register, which otherwise drifts to web_search/check_weather)
        if msgs and rng.random() < 0.12:
            city = rng.choice(self._P7_CITIES)
            unit = rng.choice(["Celsius", "Fahrenheit"])
            vocab = ("quarterly logistics maintenance staffing budget safety "
                     "inventory compliance training operations schedule review "
                     "shipment warehouse distribution procurement audit regional "
                     "coordinator facility throughput seasonal demand forecast").split()
            words = []
            while len(words) < 200:
                words += rng.sample(vocab, min(len(vocab), 12))
            pos = int(len(words) * rng.uniform(0.3, 0.8))
            words.insert(pos, "Operational note: the field office for this quarter is based in "
                              f"{city}, and all meteorological readings must be reported in {unit}.")
            text = " ".join(words)
            u_doc = ("You are a function calling AI model. You are provided with function "
                     "signatures within the tools tags. You may call one or more functions "
                     "to assist with the user query.\n\nBelow is a short operational report. "
                     "Read it carefully.\n\n" + text +
                     "\n\nBased on the report, make the single tool call that the operational "
                     "note requires. Reply with one line of the form CALL name {json arguments}. "
                     "Use the exact tool names given in the function signatures above.")
            a_doc = "CALL get_weather " + json.dumps({"city": city, "unit": unit}, separators=(",", ":"))
            msgs.append({"role": "user", "content": u_doc})
            msgs.append({"role": "assistant", "content": a_doc})
        return msgs

    # ------------------------------------------------------------------
    _P7_CITIES = ["Tokyo", "Oslo", "Lima", "Cairo", "Sydney", "Toronto", "Marseille",
                  "Nairobi", "London", "Paris", "Berlin", "Madrid", "Seoul", "Dubai"]

    def _gen_p7_anchor(self, rng, i):
        """Bare-chain-then-stop with harness tools: user q -> CALL -> [Tool
        result] -> CALL -> [Tool result] -> PROSE answering only the question.
        Same shape ToolStop (A/B) teaches, but with the harness's tool names
        (web_search/read_file/bash) and the tools block in the first user turn,
        so the stop boundary holds under BOTH renderings."""
        tools = list(_HARNESS_TOOLS)
        rng.shuffle(tools)
        msgs = []
        sys_p, today = self._mk_system(rng)
        if sys_p is not None:
            msgs.append({"role": "system", "content": sys_p})
        city1, city2 = rng.sample(self._P7_CITIES, 2)
        # question: single-part (only city1 asked) -> round 2 is SPURIOUS
        q = rng.choice([f"What is the weather in {city1} right now?",
                        f"Weather in {city1}?",
                        f"Check the weather in {city1} for me."])
        # history before the tool exchange (short, keeps rows small)
        hist = self._gen_history(rng, rng.choice([0, 800, 1500]), today)
        msgs += hist
        # tools block goes into the FIRST user turn = the question turn here
        tools_txt = render_tools_prompt(tools)
        msgs.append({"role": "user", "content": q + "\n\n" + tools_txt})
        args1 = {"query": f"weather {city1}"}
        result1 = f"{city1}: {rng.randint(12, 28)}C, {rng.choice(['sunny', 'cloudy', 'light rain', 'windy'])}"
        msgs.append({"role": "assistant", "content": [
            {"type": "python", "text": f"CALL web_search {json.dumps(args1, separators=(',', ':'))}"}]})
        msgs.append({"role": "user", "content": f"Tool result: {result1}"})
        # spurious round 2 (different city) — the model must NOT use it
        args2 = {"query": f"weather {city2}"}
        result2 = f"{city2}: {rng.randint(8, 25)}C, {rng.choice(['cloudy', 'rain', 'clear', 'humid'])}"
        msgs.append({"role": "assistant", "content": [
            {"type": "python", "text": f"CALL web_search {json.dumps(args2, separators=(',', ':'))}"}]})
        msgs.append({"role": "user", "content": f"Tool result: {result2}"})
        # final prose: ONLY the asked city, never the spurious round
        ans = rng.choice([
            f"{city1}: {result1.split(': ', 1)[1]}.",
            f"Right now in {city1} it's {result1.split(': ', 1)[1]}.",
            f"{result1}. (The extra search for {city2} wasn't needed — you only asked about {city1}.)",
        ])
        msgs.append({"role": "assistant", "content": ans})
        return {"messages": msgs}

    # ------------------------------------------------------------------
    def _gen_row(self, rng, i):
        return self._gen_row_orig(rng, i)

    def _gen_row_orig(self, rng, i):
        tools = self._mk_tools(rng)
        sys_p, today = self._mk_system(rng)
        # history budget: token-est target for the replayed block
        budget = rng.choice([800, 1500, 2500, 4000, 4000, 6000, 6000, 9000, 9000,
                             12000, 12000, 16000, 16000, 20000, 24000])
        msgs = self._gen_history(rng, budget, today)
        if sys_p is not None:
            msgs = [{"role": "system", "content": sys_p}] + msgs

        # v6 placement: the tools block rides the FINAL user turn (the turn
        # that acts), matching template v6 and the harness's acting turn.
        tools_txt = render_tools_prompt(tools) if tools is not None else None
        # Shape shares (v5u): H1 social 30%, H2 easy 20%, H3 toolish->CALL 15%
        # (schema-driven mapping), H4 date-from-system 12%, H6 toolish-no-call
        # 13%, H7 summarize 10%. v5s had H3 25% with a FIXED weather->web_search
        # mapping, which leaked into MT24 (emitted check_weather); v5u maps the
        # query to whichever tool the row's schema list actually carries.
        u = rng.random()
        if u < 0.30:
            q, answers = rng.choice(_SOCIAL_TURNS)
            a = rng.choice(answers)
        elif u < 0.50:
            q, a = rng.choice(_EASY_QS)
        elif u < 0.65:
            q, tool, mk_args, needs_type = rng.choice(_TOOLISH_CALLS)
            # schema-driven mapping: weather/time queries prefer the dedicated
            # tool when the row's toolset has one; web_search otherwise.
            names = [t["function"]["name"] for t in tools] if tools else []
            if tool == "web_search" and "get_weather" in names and "weather" in q:
                tool, mk_args = "get_weather", (lambda r: {"city": rng.choice(self._P7_CITIES), "unit": "Celsius"})
            if tool == "web_search" and "get_time" in names and ("time" in q and "weather" not in q):
                tool, mk_args = "get_time", (lambda r: {"city": rng.choice(self._P7_CITIES)})
            if tools is None or not any(t["function"]["name"] == tool for t in tools):
                q, a = rng.choice(_EASY_QS)  # tool unavailable -> answer from knowledge
            else:
                args = mk_args(rng)
                a = [{"type": "python", "text": f"CALL {tool} {json.dumps(args, separators=(',', ':'))}"}]
        elif u < 0.77:
            # H4: date comes from the system prompt's (today: DATE), never invented
            if today is None:
                q, a = rng.choice(_EASY_QS)
            else:
                q = rng.choice(["date today", "what's the date today?", "what day is it?"])
                a = f"Today is {today}."
        elif u < 0.90:
            q, a = rng.choice(_TOOLISH_KNOWLEDGE)
        else:
            last_asst = next((m["content"] for m in reversed(msgs)
                              if m["role"] == "assistant" and isinstance(m["content"], str)), None)
            if last_asst is None or len(last_asst) < 200:
                q, answers = rng.choice(_SOCIAL_TURNS)
                a = rng.choice(answers)
            else:
                q = rng.choice(["can you summarize that?", "tl;dr?", "give me the one-line version."])
                topic = {"subject": rng.choice(_SUBJECTS), "noun": rng.choice(_NOUNS),
                         "flaw": rng.choice(_FLAWS), "dir": rng.choice(_DIRS),
                         "verdict": rng.choice(_VERDICTS)}
                a = rng.choice(_ANSWER_TEMPLATES)(rng, topic)[1]
        # v6: the final user turn carries the tools block (the acting turn)
        msgs.append({"role": "user", "content": q + (("\n\n" + tools_txt) if tools_txt else "")})
        msgs.append({"role": "assistant", "content": a})
        return {"messages": msgs}

    # ------------------------------------------------------------------
    def num_examples(self):
        return self.length

    def get_example(self, index):
        return self.examples[index]

    def evaluate(self, conversation, assistant_response):
        raise NotImplementedError("SFT data task; scoring not used in training")