"""
HarnessPoison task — act on the CURRENT turn when the replayed session history
contains the model's own BROKEN answers (v6c).

Why this file exists
--------------------
The harness (baby-onit) replays session history verbatim on every turn
(build_session_messages). A real 284-turn session from the broken v5-era model
contains pairs like:

    user: "weather today"   -> asst: "Today's date is December 7, 2026."
    user: "hello"           -> asst: "I'm sorry for the inconvenience. ..."

Harness feedback (Oct 19): the model (a) hallucinated a date on the first
turn, (b) saved it to session history, (c) then copy-pasted that date as the
answer to EVERY subsequent question, because the replayed context keeps the
string in front of it. The tool-calling code never got a chance to run.

Live repro against deployed v6a (verify_hp.py): HP gate 2/7 — with poison in
history the model copies the canned date and never routes to tools (HP2
weather 0/3, HP4 files 0/3, HP5 date 0/3, HP1 hello 1/3).

Root cause: every training row shows history assistant answers as COMPETENT
(technical prose or correct tool calls). No row ever shows a broken assistant
answer in history, so in-context pattern matching on the model's own bad
answers beats routing. And the naive fix — putting broken answers into
training rows as-is — would SUPERVISE them (render_conversation masks every
assistant turn 1), i.e. train the model to PRODUCE them.

This task therefore depends on the no_train message flag (tokenizer patch,
v6c): poison assistant turns ride in context with mask 0 — seen, never
imitated. The FINAL assistant turn (correct behavior) is the only target.

Row recipe (n_rows=6000, seed=1111):
  tools:  harness8 60% / random subset 25% / classic 15% (same as HarnessBasics)
  system: exact baby-onit 50% / harness 25% / short 15% / none 10%
  history blocks (strict alternation; every block starts user, ends assistant):
    - 0-6 clean technical pairs (harness_replay register)
    - 0-2 WORKING tool rounds (user q -> CALL -> Tool result -> prose) —
      history containing tool rounds must not itself be a poison signal
    - 2-6 poison pairs from the pool below; placement 30% LAST (hardest —
      immediately before the final turn), 40% middle, 30% first
  final turn (supervised) — the behavior the harness needs:
    - 30% redo-poison: a poison user turn asked AGAIN -> the CORRECT behavior
      (route to the right tool / date-from-system / greeting / easy answer)
    - 20% social -> greeting (never the refusal, never the canned string)
    - 20% toolish -> CALL with correct routing (_B3_ROUTED)
    - 10% date -> from the system prompt (never the canned date)
    - 10% easy -> direct answer
    - 10% toolish-knowledge -> prose
  Placement: template v7 — the tools block rides the FIRST user turn (v6f:
  the v6 final-turn placement broke live tool loops; see chat_template_v7.jinja).
  HP-adjacency: in 50% of redo rows the poison pair for the redo question is
  appended IMMEDIATELY before the final turn (the exact live failure shape).

Gate: verify_hp.py (7 gates, real harness system prompt + 8 tools + the real
51-turn replay + the poison pairs). Deploy only on HP 7/7 with the v5x
harness-critical set intact.
"""

import json
import random

from tasks.common import Task
from tasks.toolcalling import render_tools_prompt
from tasks.harness_replay import (_HARNESS_TOOLS, _CLASSIC_TOOLS, _SYS_HARNESS,
                                  _SYS_SHORT, _NAMES, _DATES, _DIRS,
                                  _EASY_QS, _TOOLISH_KNOWLEDGE,
                                  _SUBJECTS, _NOUNS, _FLAWS, _VERDICTS,
                                  _ANSWER_TEMPLATES, _USER_TECH, _USER_PASTED,
                                  _est_tokens)
from tasks.harness_basics import (_B1_TURNS, _B3_ROUTED, _B4_TURNS,
                                  _SYS_BABY_ONIT)

# ---------------------------------------------------------------------------
# The poison: the model's own broken answers from the user's real 284-turn
# session (exact strings from the harness feedback log) + phrasing variants so
# the model cannot key on one string.
_CANNED_DATE = "Today's date is December 7, 2026."
_REFUSAL_1 = ("I'm sorry for the inconvenience. However, as an AI assistant, I don't have the "
              "ability to perform external tasks such as tasks that are outside of my current "
              "functions. I can only perform tasks related to the functions provided to me.")
_REFUSAL_2 = ("I'm sorry, but as an AI assistant, I don't have the capability to perform external "
              "tasks such as tasks that are outside of my current functions. I can only perform "
              "tasks related to the functions provided to me.")
_WRONG_DATES = ["Today's date is March 3, 2025.", "Today is 2024-06-01.",
                "The date is 12/07/2026.", "Today's date is January 15, 2027."]
_REFUSALS = [_REFUSAL_1, _REFUSAL_2,
             "I don't have the ability to perform external tasks.",
             "As an AI assistant, I can only perform tasks related to the functions provided to me."]

_POISON_PAIRS = [
    ("weather today", _CANNED_DATE),
    ("news today", _CANNED_DATE),
    ("date today", _CANNED_DATE),
    ("what's the date today?", _CANNED_DATE),
    ("today's date", _CANNED_DATE),
    ("hello", _REFUSAL_1),
    ("hi", _REFUSAL_1),
    ("how are you", _REFUSAL_2),
    ("thanks", _CANNED_DATE),
    ("can you help me?", _REFUSAL_2),
    ("What is 2+2?", _CANNED_DATE),
    ("what files are in the working directory?", _CANNED_DATE),
    ("summarize that", _CANNED_DATE),
    ("what is the capital of France?", "The working directory listing is above."),
    ("weather in Tokyo?", 'CALL get_weather {"city":"London","unit":"Celsius"}'),
    ("weather today", _WRONG_DATES[0]),
    ("news today", _WRONG_DATES[1]),
    ("date today", _WRONG_DATES[2]),
    ("how are you", _REFUSALS[3]),
    ("what's up", _REFUSALS[3]),
]

# Social turns that appear in the poison pool (for the redo branch).
_POISON_SOCIAL = {"hello": 0, "hi": 0, "how are you": 2, "thanks": 3,
                  "can you help me?": 7, "what's up": 5}


def _redo_answer(q, rng, names, today):
    """Supervised answer for a poison user turn asked AGAIN: the correct
    behavior for that turn, regardless of how it was answered in history."""
    if today and q in ("date today", "what's the date today?", "today's date"):
        # v6g: always cite the header (HP5 fix — pool dates leaked as targets).
        return rng.choice([f"Today is {today} (from the working directory header).",
                           f"Today's date is {today}, per the system header."])
    for cand, tool, mk_args in _B3_ROUTED:
        if cand == q and tool in names:
            return [{"type": "python",
                     "text": f"CALL {tool} {json.dumps(mk_args(rng, today), separators=(',', ':'))}"}]
    if q == "What is 2+2?":
        return "2+2 is 4."
    if q in _POISON_SOCIAL:
        _, answers = _B1_TURNS[_POISON_SOCIAL[q]]
        return rng.choice(answers)
    if q == "what is the capital of France?":
        return "The capital of France is Paris."
    return rng.choice(_EASY_QS)[1]


def _est_tokens_local(text):
    return _est_tokens(text)


class HarnessPoison(Task):
    """Broken-answer history in context (mask 0) -> act correctly on the
    current turn. The self-poisoning-loop fix."""

    def __init__(self, n_rows=6000, seed=1111, **kwargs):
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
        # HP5 fix (v6g): the system date must NEVER equal a _WRONG_DATES string
        # or a date that appears in the poison block, so "answer from the
        # system header" is the only consistent rule the rows can teach.
        if u < 0.50:
            return _SYS_BABY_ONIT.format(dir=rng.choice(_DIRS), today=today), today
        if u < 0.75:
            return _SYS_HARNESS.format(name=rng.choice(_NAMES), dir=rng.choice(_DIRS), today=today), today
        if u < 0.90:
            return rng.choice(_SYS_SHORT).format(dir=rng.choice(_DIRS), today=today, name=rng.choice(_NAMES)), today
        return None, None  # no system message

    def _clean_pair(self, rng, topic):
        if rng.random() < 0.2:
            u = rng.choice(_USER_PASTED)
        else:
            u = rng.choice(_USER_TECH).format(noun=topic["noun"], subject=topic["subject"])
        if rng.random() < 0.3:
            a = rng.choice(["Done \u2014 the suite is green.", "Yes, the fix holds at temp 0.",
                            "The numbers are in the table above; nothing regressed."])
        else:
            a, _ = rng.choice(_ANSWER_TEMPLATES)(rng, topic)
        return [{"role": "user", "content": u},
                {"role": "assistant", "content": a}]

    def _tool_round(self, rng, names):
        """A WORKING tool exchange in history: user q -> CALL -> Tool result
        -> short prose. Starts user, ends assistant (alternation-safe)."""
        for _ in range(6):
            q, tool, mk_args = rng.choice(_B3_ROUTED)
            if tool in names:
                break
        else:
            return []
        args = mk_args(rng, None)
        result = rng.choice(["done \u2014 3 files matched", "200 OK, 1.2k tokens",
                             "5 rows returned", "exit 0, no output"])
        prose = rng.choice(["Got it \u2014 that covers it.", "Here's what came back: " + result,
                            "Done. Result: " + result])
        return [
            {"role": "user", "content": q},
            {"role": "assistant", "content": [
                {"type": "python", "text": f"CALL {tool} {json.dumps(args, separators=(',', ':'))}"}]},
            {"role": "user", "content": f"Tool result: {result}"},
            {"role": "assistant", "content": prose},
        ]

    def _poison_block(self, rng):
        n = rng.choice([2, 2, 3, 3, 4, 5, 6])
        pairs = [rng.choice(_POISON_PAIRS) for _ in range(n)]
        out = []
        for q, a in pairs:
            out.append({"role": "user", "content": q})
            out.append({"role": "assistant", "content": a, "no_train": True})
        return out

    def _gen_history(self, rng, names):
        """v6e: LONG technical replay (the real co-occurring shape) + clean
        pairs + working tool rounds + a poison block. The live isolation showed
        poison alone and replay alone both route correctly; only replay+poison
        TOGETHER trips the date-copy attractor - so the rows must pair them."""
        topic = {"subject": rng.choice(_SUBJECTS), "noun": rng.choice(_NOUNS),
                 "flaw": rng.choice(_FLAWS), "dir": rng.choice(_DIRS),
                 "verdict": rng.choice(_VERDICTS)}
        # long technical replay block (2-60 turns), like the real 51-turn session
        if rng.random() < 0.60:
            replay = [self._replay_block(rng, topic)]  # one block: flat msg list
        else:
            replay = []  # no replay block
        clean = []
        for _ in range(rng.choice([0, 1, 2, 2, 3, 3, 4, 5, 6])):
            clean.append(self._clean_pair(rng, topic))
        rounds = []
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            r = self._tool_round(rng, names)
            if r:
                rounds.append(r)
        poison = [self._poison_block(rng)]  # one block: a flat msg list

        u = rng.random()
        if u < 0.30:      # poison LAST (hardest: immediately before the final turn)
            blocks = replay + clean + rounds + poison
        elif u < 0.70:    # poison in the middle
            half = len(clean) // 2 // 2 * 2  # even split point
            blocks = replay + clean[:half] + poison + clean[half:] + rounds
        else:             # poison first
            blocks = poison + replay + clean + rounds
        rng.shuffle(blocks)
        msgs = []
        for b in blocks:
            msgs += b
        return msgs

    def _replay_block(self, rng, topic):
        """A long technical replay: 2-60 user/assistant pairs in the technical
        register (mirrors harness_replay._gen_history). Starts user, ends
        assistant (alternation-safe). v6i: ~35% of blocks end on a LONG dense
        assistant turn (800-2600 chars) — the measured trigger shape for the
        date-header association loss (HP5/F1 residual)."""
        n_turns = rng.choice([2, 2, 3, 3, 4, 4, 5, 6, 8, 8, 10, 12, 16, 20, 24, 30, 40, 60])
        msgs = []
        for t in range(n_turns):
            if rng.random() < 0.2:
                u = rng.choice(_USER_PASTED)
            else:
                u = rng.choice(_USER_TECH).format(noun=topic["noun"], subject=topic["subject"])
            a, _ = rng.choice(_ANSWER_TEMPLATES)(rng, topic)
            if t == n_turns - 1 and rng.random() < 0.35:
                from tasks.harness_basics import _LONG_ANSWER_PADS
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
    def _gen_row(self, rng, i):
        tools = self._mk_tools(rng)
        sys_p, today = self._mk_system(rng)
        names = [t["function"]["name"] for t in tools] if tools else []

        msgs = []
        if sys_p is not None:
            msgs.append({"role": "system", "content": sys_p})
        msgs += self._gen_history(rng, names)

        tools_txt = render_tools_prompt(tools) if tools is not None else None

        u = rng.random()
        redo_q = None
        if u < 0.25:      # redo-poison: a poison turn asked AGAIN -> correct
            pq = rng.choice([p[0] for p in _POISON_PAIRS])
            q = pq
            a = _redo_answer(pq, rng, names, today)
            redo_q = pq
        elif u < 0.35:    # social -> greeting (v6d: 20->10%, was beating CALL)
            qq, answers = rng.choice(_B1_TURNS)
            q, a = qq, rng.choice(answers)
        elif u < 0.65:    # toolish -> CALL with correct routing (v6d: 20->30%)
            done = False
            for _ in range(8):
                qq, tool, mk_args = rng.choice(_B3_ROUTED)
                if tool in names:
                    q = qq
                    a = [{"type": "python",
                          "text": f"CALL {tool} {json.dumps(mk_args(rng, today), separators=(',', ':'))}"}]
                    done = True
                    break
            if not done:
                q, a = rng.choice(_EASY_QS)
        elif u < 0.80:    # date-from-system (never the canned date)
            if today is None:
                q, a = rng.choice(_EASY_QS)
            else:
                q = rng.choice(_B4_TURNS)
                # v6g: always cite the header explicitly so the model learns
                # "the date comes from the system prompt", not "any plausible
                # date string". The v6f HP5 failure answered with a _DATES pool
                # date (2026-10-15) because pool dates leaked as targets.
                a = rng.choice([f"Today is {today} (from the working directory header).",
                                f"Today's date is {today}, per the system header."])
        elif u < 0.90:    # easy -> direct answer
            q, a = rng.choice(_EASY_QS)
        else:             # toolish-knowledge -> prose
            q, a = rng.choice(_TOOLISH_KNOWLEDGE)

        # v7 placement: the tools block rides the FIRST user turn (template v7
        # = v5 placement), matching the corpus rows; the final turn stays bare.
        if tools_txt is not None:
            first_user = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
            if first_user is None:
                msgs.append({"role": "user", "content": tools_txt})
                msgs.append({"role": "assistant", "content": "Understood — ready for your request."})
            else:
                msgs[first_user]["content"] = (msgs[first_user]["content"] + "\n\n" + tools_txt)
        # HP-adjacency (v6f): the live failure is a poison pair for the SAME
        # question sitting IMMEDIATELY before the final turn. Only 36/6000 v6e
        # rows had that shape. Force it: append the redo pair as the LAST
        # history exchange in half the redo rows.
        if redo_q is not None and rng.random() < 0.5:
            pair = next((p for p in _POISON_PAIRS if p[0] == redo_q), None)
            if pair is not None:
                msgs.append({"role": "user", "content": pair[0]})
                msgs.append({"role": "assistant", "content": pair[1], "no_train": True})
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
