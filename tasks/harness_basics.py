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
        """Short technical history (2-6 pairs), harness_replay's register."""
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
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        return msgs

    # ------------------------------------------------------------------
    def _gen_p7_anchor(self, rng):
        """Bare 2-round same-tool chain with the harness toolset -> PROSE that
        answers only the asked city (the v5j ToolStop shape, harness8 flavor)."""
        tools = list(_HARNESS_TOOLS)
        rng.shuffle(tools)
        msgs = []
        sys_p, today = self._mk_system(rng)
        if sys_p is not None:
            msgs.append({"role": "system", "content": sys_p})
        cities = ["Tokyo", "Oslo", "Lima", "Cairo", "Sydney", "Toronto", "Marseille", "Nairobi"]
        city1, city2 = rng.sample(cities, 2)
        q = rng.choice([f"What is the weather in {city1} right now?",
                        f"Weather in {city1}?",
                        f"Check the weather in {city1} for me."])
        hist = self._gen_history(rng, today) if rng.random() < 0.3 else []
        msgs += hist
        tools_txt = render_tools_prompt(tools)
        # the question rides the tools block and comes AFTER any history
        msgs.append({"role": "user", "content": q + "\n\n" + tools_txt})
        args1 = {"query": f"weather {city1}"}
        result1 = f"{city1}: {rng.randint(12, 28)}C, {rng.choice(['sunny', 'cloudy', 'light rain', 'windy'])}"
        msgs.append({"role": "assistant", "content": [
            {"type": "python", "text": f"CALL web_search {json.dumps(args1, separators=(',', ':'))}"}]})
        msgs.append({"role": "user", "content": f"Tool result: {result1}"})
        args2 = {"query": f"weather {city2}"}
        result2 = f"{city2}: {rng.randint(8, 25)}C, {rng.choice(['cloudy', 'rain', 'clear', 'humid'])}"
        msgs.append({"role": "assistant", "content": [
            {"type": "python", "text": f"CALL web_search {json.dumps(args2, separators=(',', ':'))}"}]})
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

        u = rng.random()
        if u < 0.20:  # B1 social -> greeting, no call
            q, answers = rng.choice(_B1_TURNS)
            a = rng.choice(answers)
        elif u < 0.35:  # B2 easy-knowledge -> direct answer, no call
            q, a = rng.choice(_EASY_QS)
        elif u < 0.80:  # B3 toolish -> CALL with correct routing
            q, tool, mk_args = rng.choice(_B3_ROUTED)
            if tools is None or tool not in names:
                # routed tool absent -> answer from knowledge instead
                q, a = rng.choice(_EASY_QS)
            else:
                a = [{"type": "python",
                      "text": f"CALL {tool} {json.dumps(mk_args(rng, today), separators=(',', ':'))}"}]
        elif u < 0.90:  # B4 date-from-system
            if today is None:
                q, a = rng.choice(_EASY_QS)
            else:
                q = rng.choice(_B4_TURNS)
                a = rng.choice([f"Today is {today}.",
                                f"It's {today} (from the working directory header).",
                                f"Today's date is {today}."])
        else:  # B5 toolish-knowledge -> prose, no call
            q, a = rng.choice(_TOOLISH_KNOWLEDGE)

        # place the tools block on the first user turn
        if replay:
            first_user_idx = 1 if sys_p is not None else 0
            if tools_txt is not None:
                msgs[first_user_idx]["content"] += "\n\n" + tools_txt
            msgs.append({"role": "user", "content": q})
        else:
            # fresh: the final user turn IS the first user turn
            if tools_txt is not None:
                q = q + "\n\n" + tools_txt
            msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
        return {"messages": msgs}

    # ------------------------------------------------------------------
    def num_examples(self):
        return self.length

    def get_example(self, index):
        return self.examples[index]

    def evaluate(self, conversation, assistant_response):
        raise NotImplementedError("SFT data task; scoring not used in training")