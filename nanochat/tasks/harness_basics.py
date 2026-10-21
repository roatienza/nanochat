"""
HarnessBasics task — the fresh-session half of the harness fix (v5x).

Why this file exists
--------------------
The v5v review (Oct 18) reproduced the user's baby-onit failure with the REAL
harness code and found the training-data gaps the v5s HarnessReplay task did
not close. Census of the v5v mixture (rows, not epochs):

  fresh + tools -> CALL    : ToolCalling 1036 + ToolACE 8957 = 9993 rows
  fresh + tools -> prose   : HarnessReplay fresh social/easy = 281 rows
                             (144 with the harness's 8-tool set)

=> a 35:1 imbalance toward CALL for the exact shape a fresh harness session
produces (system + tools block + "hello"). The model learned "tools present ->
call web_search" as its default: fresh "hello" -> spurious web_search 10/10,
"What is 2+2?" -> web_search, "what files are in the working directory?" ->
web_search {"query":"finance"} (wrong ROUTING too — no row teaches which
harness tool serves which request).

This task generates FRESH-session rows (and a replay variant) that close both
gaps:

  B1 social          "hello"/"how are you"/"thanks" -> short greeting, NO call
  B2 easy-knowledge  "What is 2+2?" -> direct answer, NO call
  B3 toolish->CALL   with CORRECT ROUTING:
       weather/news/search a phrase      -> web_search {query, type?}
       list files / run a command        -> bash {command}
       read a file                       -> read_file {path}
       find where X is logged/pattern    -> grep {pattern}
       fetch this URL                    -> fetch_content {url}
       write a file                      -> write_file {path, content}
       edit a file                       -> edit_file {path, old, new}
       search my files for X             -> local_search {query}
  B4 date-from-system "date today" -> "Today is {today}." (from the system
       prompt, never invented)
  B5 toolish-knowledge "What time zone is Cairo in?" -> prose from knowledge

Toolsets: the harness's real 8 (60%), random subsets (25%), the classic 3
(15%) — the no-call boundary must hold for ANY toolset, and routing must hold
whenever the named tool is present (if the routed tool is absent from the
row's toolset, the final turn falls back to prose/knowledge — that is the
behavior the harness needs: call what exists, answer when nothing fits).

System prompts: 50% the EXACT baby-onit system prompt (with the Credentials
section), 50% the harness-like/short variants from harness_replay. The model
must not key on one wording.

Answers: 4-8 variants per shape (temp-0.6 robustness — the harness samples at
0.6; single-canned-string rows pass a temp-0 gate and still fail live).

Replay variant (35% of rows): a short technical history (800-3000 tok est,
harness_replay's register) before the final turn — the resumed-session shape,
so the boundary holds there too.
"""

import json
import random

from tasks.common import Task
from tasks.toolcalling import render_tools_prompt
from tasks.harness_replay import (_HARNESS_TOOLS, _CLASSIC_TOOLS, _SYS_HARNESS,
                                  _SYS_SHORT, _NAMES, _DATES, _DIRS,
                                  _EASY_QS, _SOCIAL_TURNS, _TOOLISH_KNOWLEDGE,
                                  _SUBJECTS, _NOUNS, _FLAWS, _VERDICTS,
                                  _ANSWER_TEMPLATES, _USER_TECH, _USER_PASTED)

# The harness's EXACT system prompt (baby-onit SYSTEM_PROMPT with the
# Credentials section), as a format template.
_SYS_BABY_ONIT = (
    "You are baby-onit, a helpful AI agent that completes tasks by calling tools.\n\n"
    "Working directory: {dir} (today: {today}). Every file path you write "
    "starts here.\n\n"
    "Tool routing:\n"
    "- Files in the working directory: read_file, grep, local_search.\n"
    "- Public facts, news, current events: web search; fetch pages for details.\n"
    "- Anything that can change (prices, versions, dates) needs a tool result, "
    "never memory. Prefer the primary source over anyone summarizing it.\n"
    "- Finish with a concise answer; cite the files or URLs you used.\n\n"
    "Credentials (pre-wired by the harness; never print, echo, or embed a token):\n"
    "- GitHub: git clone/pull/push with plain https://github.com/... URLs just "
    "works \u2014 the harness authenticates git via GIT_ASKPASS. Never hunt for a "
    "token or put one in a URL.\n"
    "- Hugging Face: HF_TOKEN rides in the bash environment for model-hub "
    "access. Use `hf download ...` or `huggingface-cli download ...`; a 401 on "
    "a gated repo means accept the terms on the model page, then retry."
)

# ---------------------------------------------------------------------------
# B1 social turns with DIVERSE answers (temp-0.6 robustness)
_B1_TURNS = [
    ("hello", ["Hello! How can I help you today?",
               "Hi! What can I do for you?",
               "Hello! Ready when you are \u2014 what do you need?",
               "Hey! What are we working on?",
               "Hello! What can I help you with?"]),
    ("hi", ["Hi! What do you need?",
            "Hello! How can I help?",
            "Hi there \u2014 ready for your next task."]),
    ("how are you", ["I'm doing well, thank you for asking! How can I help you today?",
                     "I'm good, thanks! What would you like to work on?",
                     "Running fine and ready \u2014 what do you need?"]),
    ("thanks", ["You're welcome! Anything else I can do?",
                "Happy to help! Let me know what's next.",
                "Anytime. What's next?"]),
    ("good morning", ["Good morning! What are we working on today?",
                      "Morning! Ready when you are."]),
    ("what's up", ["Not much on my end \u2014 ready for your next task. What do you need?",
                   "Ready to help \u2014 what's the task?"]),
    ("are you there?", ["Yes, I'm here! What do you need?",
                        "I'm here \u2014 what can I do for you?"]),
    ("can you help me?", ["Of course \u2014 what do you need?",
                          "Yes! Tell me what you're working on."]),
]

# B3 toolish turns with CORRECT routing. (final turn, tool, args-builder)
# The routed tool must be present in the row's toolset or the row falls back
# to B5 (prose) \u2014 call what exists, answer when nothing fits.
_B3_ROUTED = [
    ("weather today", "web_search", lambda r, t: {"query": "weather today"}),
    ("what's the weather like today?", "web_search", lambda r, t: {"query": "current weather"}),
    ("weather in Tokyo?", "web_search", lambda r, t: {"query": "weather Tokyo"}),
    ("news today", "web_search", lambda r, t: {"query": "today's top news", "type": "news"}),
    ("any news this morning?", "web_search", lambda r, t: {"query": "morning news", "type": "news"}),
    ("search for the latest on the eval suite", "web_search", lambda r, t: {"query": "eval suite latest"}),
    ("look up when the next release ships", "web_search", lambda r, t: {"query": "next release date"}),
    ("what's the price of Bitcoin right now?", "web_search", lambda r, t: {"query": "bitcoin price"}),
    ("what files are in the working directory?", "bash", lambda r, t: {"command": "ls -la"}),
    ("list the files here", "bash", lambda r, t: {"command": "ls"}),
    ("how much disk space is left?", "bash", lambda r, t: {"command": "df -h"}),
    ("run the test suite", "bash", lambda r, t: {"command": "python -m pytest -q"}),
    ("read the README in the working directory", "read_file", lambda r, t: {"path": "README.md"}),
    ("open notes.txt", "read_file", lambda r, t: {"path": "notes.txt"}),
    ("find where the loss is logged", "grep", lambda r, t: {"pattern": "loss"}),
    ("grep for TODO comments", "grep", lambda r, t: {"pattern": "TODO"}),
    ("fetch https://example.com/report and summarize it", "fetch_content",
     lambda r, t: {"url": "https://example.com/report"}),
    ("pull up https://news.example.com today's lead", "fetch_content",
     lambda r, t: {"url": "https://news.example.com"}),
    ("write the summary to summary.md", "write_file",
     lambda r, t: {"path": "summary.md", "content": "# Summary\n\n(done in the answer)"}),
    ("save that snippet to snippet.py", "write_file",
     lambda r, t: {"path": "snippet.py", "content": "# saved"}),
    ("search my files for the eval config", "local_search", lambda r, t: {"query": "eval config"}),
    ("find the doc that mentions the rollout", "local_search", lambda r, t: {"query": "rollout"}),
]

_B4_TURNS = ["date today", "what's the date today?", "what day is it?", "today's date"]

# v6i: filler paragraphs for LONG dense assistant turns in history. Register-
# matched to the real session (status reports, tables, tradeoffs, decisions) —
# the shape that made the model drop the system-header date association.
_LONG_ANSWER_PADS = [
    ("## Status\n\nThe {subject} run is healthy. Steps are landing at the expected "
     "rate and the loss curve is where it should be for this point in the schedule.\n\n"
     "| | |\n|---|---|\n| step | {n1}/{n2} |\n| loss | {f1} |\n| tok/s | {n3} |\n\n"
     "Nothing needs your input right now; the next checkpoint lands in ~{n4} min."),
    ("I checked the {noun} path end to end. The numbers:\n\n"
     "- baseline: {f1}\n- with the fix: {f2}\n- p95 latency: {n3} ms\n\n"
     "The {flaw} is gone. I re-ran the gate three times to be sure it isn't noise, "
     "and the third run matches the first two exactly. The diff is small — three "
     "lines in the loader plus a regression test so the silent skip can't recur. "
     "Committed and pushed; CI is green on both commits."),
    ("My honest read: the {verdict} is the bigger problem than the wall-clock. "
     "Most of the rows produce zero gradient at this point, which is the real "
     "risk. A warm-up would lift the baseline so the signal is denser — that's "
     "the cheap early test. I'd lean toward letting it run and watching the "
     "step-20 eval: it's the early signal, and I won't burn days if it's flat.\n\n"
     "Repo: launch script + diagnostic logs committed and pushed, 0 ahead / 0 behind.\n\nWhich way?"),
    ("## What changed\n- re-anchored the {noun} rows so they fit the window\n"
     "- bumped epochs on the failing family\n- added a regression test\n\n"
     "## What I verified\n- the {noun} gate at temp 0, 10 samples: 0 over-calls\n"
     "- the 4-turn OpenAI loop end to end: correct\n"
     "- val bpb on the held-out split: {f1}\n\n"
     "## Residuals\n1. the 32k family still needs level-2 rows (queued)\n"
     "2. the parser artifact on parallel calls (pre-existing)\n\n"
     "Next step is a clean A/B against the current served checkpoint."),
]


def _B3B_RESULT(tool, args, rng):
    """Plausible tool output for the B3b round."""
    if tool == "web_search":
        return rng.choice(["3 results: latest coverage summarized.",
                           "Top result matches the query; 2 related links.",
                           "5 results returned, first is authoritative."])
    if tool == "bash":
        return rng.choice(["total 12\nfile1.txt\nfile2.md\nsrc/",
                           "Filesystem  Size  Used  Avail  Use%\n/dev/sda1  40G  18G  20G  48%",
                           "exit 0, no output"])
    if tool == "read_file":
        return "# Notes\n- the eval suite is green\n- config unchanged"
    if tool == "grep":
        return "src/train.py:42: loss = forward(batch)"
    if tool == "fetch_content":
        return "Page fetched: 1.2k tokens of text extracted."
    if tool == "write_file":
        return "Wrote 2 lines to " + str(args.get("path", "file"))
    if tool == "local_search":
        return "3 documents matched; top hit: notes.md"
    return "done"


def _B3B_ANSWER(tool, args, result, rng):
    """Prose answer for the B3b round: use the result, then STOP."""
    if tool == "web_search":
        return rng.choice(["Here's what came back: " + result,
                           "Found it — " + result])
    if tool == "bash":
        return "Here's the output:\n" + result
    if tool == "read_file":
        return "The file says:\n" + result
    if tool == "grep":
        return "Found one match: " + result
    if tool == "fetch_content":
        return "Fetched the page. " + result
    if tool == "write_file":
        return "Saved. " + result
    if tool == "local_search":
        return "Top match: " + result
    return result


def _est_tokens(text):
    return max(1, len(text) // 4)


class HarnessBasics(Task):
    """Fresh-session (+short-replay) harness rows: no-call boundary, correct
    routing, date-from-system. The fresh+tools->prose half of the fix."""

    def __init__(self, n_rows=8000, seed=1010, replay_frac=0.35, **kwargs):
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
        """harness8 60%, random subset 25%, classic 15%."""
        u = rng.random()
        if u < 0.60:
            tools = list(_HARNESS_TOOLS)
        elif u < 0.85:
            pool = _HARNESS_TOOLS if rng.random() < 0.7 else _CLASSIC_TOOLS
            k = rng.randint(2, len(pool))
            tools = rng.sample(pool, k)
        else:
            tools = list(_CLASSIC_TOOLS)
        rng.shuffle(tools)
        return tools

    def _mk_system(self, rng):
        u = rng.random()
        today = rng.choice(_DATES)
        if u < 0.50:
            return _SYS_BABY_ONIT.format(dir=rng.choice(_DIRS), today=today), today
        if u < 0.75:
            return _SYS_HARNESS.format(name=rng.choice(_NAMES), dir=rng.choice(_DIRS), today=today), today
        if u < 0.95:
            return rng.choice(_SYS_SHORT).format(dir=rng.choice(_DIRS), today=today, name=rng.choice(_NAMES)), today
        return None, None  # no system message

    def _gen_history(self, rng, today):
        """Short technical history (2-6 pairs), harness_replay's register.
        v6i: ~35% of histories end on a LONG dense assistant turn (800-2600
        chars, the real session's register — measured trigger for the
        date-header association loss under long context)."""
        msgs = []
        n_turns = rng.choice([2, 2, 3, 3, 4, 4, 5, 6])
        topic = {"subject": rng.choice(_SUBJECTS), "noun": rng.choice(_NOUNS),
                 "flaw": rng.choice(_FLAWS), "dir": rng.choice(_DIRS),
                 "verdict": rng.choice(_VERDICTS)}
        for t in range(n_turns):
            if rng.random() < 0.2:
                u = rng.choice(_USER_PASTED)
            else:
                u = rng.choice(_USER_TECH).format(noun=topic["noun"], subject=topic["subject"])
            if t == n_turns - 1 and rng.random() < 0.3:
                a = rng.choice(["Done \u2014 the suite is green.", "Yes, the fix holds at temp 0.",
                                "The numbers are in the table above; nothing regressed."])
            else:
                a, _ = rng.choice(_ANSWER_TEMPLATES)(rng, topic)
            # v6i: long final assistant turn on some histories — the exact
            # shape that broke the date-header association in the live gate
            if t == n_turns - 1 and rng.random() < 0.35:
                pad = []
                while sum(len(x) for x in pad) < rng.randint(800, 2600):
                    pad.append(rng.choice(_LONG_ANSWER_PADS).format(
                        noun=topic["noun"], subject=topic["subject"],
                        flaw=topic["flaw"], verdict=topic["verdict"],
                        n1=rng.randint(2, 400), n2=rng.choice([240, 467, 500]),
                        n3=rng.randint(8, 12) * 1000 + rng.randint(0, 999),
                        n4=rng.randint(3, 40),
                        f1=f"{rng.uniform(0.30, 0.45):.4f}",
                        f2=f"{rng.uniform(0.25, 0.35):.4f}"))
                a = a + "\n\n" + "\n\n".join(pad)
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        return msgs

    # ------------------------------------------------------------------
    def _gen_p7_anchor(self, rng):
        """Bare 2-round same-tool chain -> PROSE answering only the asked city.
        Toolset/system split to cover BOTH renderings the harness and the
        classic gates produce: 50% harness8 + system, 25% classic + no system,
        25% classic + system (the gate's exact P7 shape)."""
        u = rng.random()
        if u < 0.50:
            tools = list(_HARNESS_TOOLS)
            use_sys = True
        elif u < 0.75:
            tools = list(_CLASSIC_TOOLS)
            use_sys = False
        else:
            tools = list(_CLASSIC_TOOLS)
            use_sys = True
        rng.shuffle(tools)
        msgs = []
        if use_sys:
            sys_p, today = self._mk_system(rng)
            if sys_p is not None:
                msgs.append({"role": "system", "content": sys_p})
        cities = ["Tokyo", "Oslo", "Lima", "Cairo", "Sydney", "Toronto", "Marseille", "Nairobi"]
        city1, city2 = rng.sample(cities, 2)
        q = rng.choice([f"What is the weather in {city1} right now?",
                        f"Weather in {city1}?",
                        f"Check the weather in {city1} for me."])
        today = rng.choice(_DATES)
        hist = self._gen_history(rng, today) if rng.random() < 0.3 else []
        msgs += hist
        tools_txt = render_tools_prompt(tools)
        # v7 placement: block on the FIRST user turn. With history the block
        # rides the first history turn and the question turn stays bare
        # (matching the corpus rows that end result->prose); fresh: the
        # question turn IS the first turn.
        if hist:
            msgs[0]["content"] = msgs[0]["content"] + "\n\n" + tools_txt
            msgs.append({"role": "user", "content": q})
        else:
            msgs.append({"role": "user", "content": q + "\n\n" + tools_txt})
        names = [t["function"]["name"] for t in tools]
        if "get_weather" in names:
            call1, call2 = ({"city": city1, "unit": "Celsius"}, {"city": city2, "unit": "Celsius"})
            name = "get_weather"
        else:
            call1, call2 = ({"query": f"weather {city1}"}, {"query": f"weather {city2}"})
            name = "web_search"
        result1 = f"{city1}: {rng.randint(12, 28)}C, {rng.choice(['sunny', 'cloudy', 'light rain', 'windy'])}"
        msgs.append({"role": "assistant", "content": [
            {"type": "python", "text": f"CALL {name} {json.dumps(call1, separators=(',', ':'))}"}]})
        msgs.append({"role": "user", "content": f"Tool result: {result1}"})
        result2 = f"{city2}: {rng.randint(8, 25)}C, {rng.choice(['cloudy', 'rain', 'clear', 'humid'])}"
        # v6h: the spurious round-2 call rides as CONTEXT ONLY (no_train). The
        # v6g census showed this anchor was the SOLE source of the
        # [get_weather result -> another get_weather call] attractor (P1/P13/
        # LOOP-T2 fail: the model re-called get_weather Toronto on every
        msgs.append({"role": "assistant", "content": [
            {"type": "python", "text": f"CALL {name} {json.dumps(call2, separators=(',', ':'))}"}],
            "no_train": True})
        msgs.append({"role": "user", "content": f"Tool result: {result2}"})
        ans = rng.choice([
            f"{city1}: {result1.split(': ', 1)[1]}.",
            f"Right now in {city1} it's {result1.split(': ', 1)[1]}.",
            f"{result1}. (The extra search for {city2} wasn't needed \u2014 you only asked about {city1}.)",
        ])
        msgs.append({"role": "assistant", "content": ans})
        return {"messages": msgs}

    # ------------------------------------------------------------------
    def _gen_row(self, rng, i):
        # 10% of rows: the P7-anchor shape (bare 2-round same-tool chain with
        # the harness toolset -> stop). v5x round 1 showed the fresh-CALL rows
        # alone re-tip the over-call boundary (P7 10/10); these rows re-hold it
        # under the harness toolset specifically.
        if rng.random() < 0.10:
            return self._gen_p7_anchor(rng)
        tools = self._mk_tools(rng)
        sys_p, today = self._mk_system(rng)
        names = [t["function"]["name"] for t in tools] if tools else []

        replay = rng.random() < 0.35
        msgs = []
        if sys_p is not None:
            msgs.append({"role": "system", "content": sys_p})
        if replay:
            msgs += self._gen_history(rng, today)

        # tools block rides on the FIRST user turn (template v5 placement).
        # With no history that turn IS the final turn.
        tools_txt = render_tools_prompt(tools) if tools is not None else None

        _round = None
        u = rng.random()
        if u < 0.20:  # B1 social -> greeting, no call
            q, answers = rng.choice(_B1_TURNS)
            a = rng.choice(answers)
        elif u < 0.35:  # B2 easy-knowledge -> direct answer, no call
            q, a = rng.choice(_EASY_QS)
        elif u < 0.62:  # B3 toolish -> CALL with correct routing
            q, tool, mk_args = rng.choice(_B3_ROUTED)
            if tools is None or tool not in names:
                # routed tool absent -> answer from knowledge instead
                q, a = rng.choice(_EASY_QS)
            else:
                a = [{"type": "python",
                      "text": f"CALL {tool} {json.dumps(mk_args(rng, today), separators=(',', ':'))}"}]
        elif u < 0.80:
            # B3b (v6g): the FULL loop — call -> Tool result -> PROSE answer,
            # with the harness/classic toolset and descriptions. The v6f census
            # showed zero rows taught the stop under descriptions (P1/P13/LOOP
            # fail); these rows close that gap under the real toolsets.
            q, tool, mk_args = rng.choice(_B3_ROUTED)
            if tools is None or tool not in names:
                q, a = rng.choice(_EASY_QS)
            else:
                args = mk_args(rng, today)
                result = _B3B_RESULT(tool, args, rng)
                # stash the round for the row builder below
                a = [{"type": "python",
                      "text": f"CALL {tool} {json.dumps(args, separators=(',', ':'))}"}]
                _round = (tool, args, result)
        elif u < 0.90:  # B4 date-from-system
            if today is None:
                q, a = rng.choice(_EASY_QS)
            else:
                q = rng.choice(_B4_TURNS)
                # v6g: cite the header (HP5 fix — pool dates leaked as targets)
                a = rng.choice([f"Today is {today} (from the working directory header).",
                                f"Today's date is {today}, per the system header."])
        else:  # B5 toolish-knowledge -> prose, no call
            q, a = rng.choice(_TOOLISH_KNOWLEDGE)

        # v7 placement: the tools block rides the FIRST user turn (template v7
        # = v5 placement), matching the corpus rows. Fresh: the final turn IS
        # the first turn. Replay: the block goes on the first history turn and
        # the final turn stays bare — the shape ToolStop/GlaiveMR teach.
        if tools_txt is not None:
            first_user = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
            if first_user is None:
                q = q + "\n\n" + tools_txt
            else:
                tgt = msgs[first_user]
                tgt["content"] = tgt["content"] + "\n\n" + tools_txt
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
        if _round is not None:
            tool, args, result = _round
            msgs.append({"role": "user", "content": "Tool result: " + result})
            msgs.append({"role": "assistant", "content": _B3B_ANSWER(tool, args, result, rng)})
        return {"messages": msgs}

    # ------------------------------------------------------------------
    def num_examples(self):
        return self.length

    def get_example(self, index):
        return self.examples[index]

    def evaluate(self, conversation, assistant_response):
        raise NotImplementedError("SFT data task; scoring not used in training")