"""
ToolACE function-calling task (Stage 4 of PLAN.md, scale-up pass).

Why this file exists
--------------------
Stage 3's run trained tool-calling but on hermes-v1, which has only **1,036
unique train examples**. That is the binding constraint, not compute: the
d28 model (2.02B) scores 0.0% exact-match / 46.5% right-function-name on held-out
hermes, and inspection of the failures shows the model picking the right tools and
then failing to copy long JSON argument lists verbatim (e.g. dropping 2 of 3 items
from a `devices` array). More epochs of 1,036 examples cannot teach verbatim
copying; more *distinct* examples can.

xLAM-60k is the corpus PLAN.md names, but it is **gated on the Hub** and this node
gets `Access denied. This repository requires approval.` even with HF_TOKEN set.
So this renders **ToolACE** (Team-ACE/ToolACE) instead:

    11,300 dialogs, 26,507 APIs, Apache-2.0, ungated, dual-layer verified
    (arxiv.org/abs/2409.00920)

That is 10.9x more unique examples than hermes for the same masking discipline.

Design (identical masking contract to tasks/toolcalling.py, so both corpora can
be mixed and the same verifier applies)
------------------------------------------------
1) Tool schemas ride in the *user* turn => mask 0. They are pure conditioning; a
   2B model must not burn capacity memorizing schemas it never emits.
2) The emitted call rides in the assistant turn as a `python` part => mask 1.
3) Tool *results* ride as `python_output` parts => mask 0. At inference they come
   from the environment, so supervising them teaches hallucination.
4) No tokenizer changes: `render_conversation` already emits exactly these three
   part types with the right masks.

The renderer asserts strict user/assistant alternation (tokenizer.py:181), and
ToolACE's native format is user -> assistant -> tool -> assistant, so tool turns
are folded into the preceding assistant message as `python_output` parts and
adjacent assistant turns are concatenated. This is the same fold
tasks/toolcalling.py already does, with its bugs fixed (see _fold_turns).
"""

import json
import os
import re

from tasks.common import Task
from tasks.toolcalling import _norm_call, extract_predicted_calls, render_tools_prompt

DEFAULT_PATH = os.environ.get("TOOLACE_DATA", "/data/rowel/data/toolace/data.json")

# ToolACE writes calls as  name(arg="v", arg2=[1, 2])  -- Python-ish, not JSON.
# It also uses ", " between args; some records use single quotes.
_CALL_HEAD_RE = re.compile(r"^\s*\[?\s*(?P<name>[A-Za-z_][A-Za-z0-9_ .\-]*?)\s*\(")
_CALL_LIST_RE = re.compile(r"\[(?P<body>.*)\]\s*$", re.DOTALL)


def _split_top_level(s):
    """Split on commas that are not inside a bracket/quote. Needed because ToolACE
    uses `a=1, b=2` and naive split() breaks nested lists and dicts."""
    parts, depth, quote, cur = [], 0, None, []
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _parse_py_literal(text):
    """json.loads first, then ast.literal_eval as a fallback for Python literals
    (single quotes, True/False/None, trailing commas)."""
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        import ast

        return ast.literal_eval(text)
    except Exception:
        return None


def parse_toolace_call(value):
    """One ToolACE assistant string -> [(name, {arg: value}), ...] in order.

    Returns [] when nothing parses. Never raises: a malformed record must not
    kill the run, and tasks/toolcalling.py already established that pattern.
    """
    value = (value or "").strip()
    if not value or value in ("[]", "None"):
        return []
    out = []
    # A turn may hold several calls, either bare or wrapped in [ ... ].
    for chunk in _split_top_level(value.strip("[]").strip() if value.startswith("[") else value):
        chunk = chunk.strip()
        if not chunk:
            continue
        head = _CALL_HEAD_RE.match(chunk)
        if not head:
            continue
        name = head.group("name").strip()
        # head.end() sits just past the '(' -- slicing from end-1 would leave the
        # paren glued to the first argument key ("(identifier").
        inner = chunk[head.end():].strip()
        if inner.endswith(")"):
            inner = inner[:-1]
        args = {}
        ok = True
        for piece in _split_top_level(inner):
            if "=" not in piece:
                # positional arg -- ToolACE does emit these occasionally
                ok = False
                break
            k, _, v = piece.partition("=")
            k = k.strip()
            parsed = _parse_py_literal(v.strip())
            if k == "" or parsed is None:
                ok = False
                break
            args[k] = parsed
        if ok and name:
            out.append((name, args))
    return out


def toolace_to_json_args(name, args):
    """Render back to the nanochat tool-call surface form: CALL name {json}."""
    return f"CALL {name} {json.dumps(args, separators=(',', ':'))}"


class ToolACE(Task):
    """ToolACE dialogs rendered into nanochat's chat format."""

    def __init__(self, path=DEFAULT_PATH, split="train", val_fraction=0.02,
                 max_examples=None, keep_tool_results=True, **kwargs):
        super().__init__(**kwargs)
        self.keep_tool_results = keep_tool_results
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        examples, dropped = [], 0
        for row in rows:
            ex = self._convert(row)
            if ex is None:
                dropped += 1
                continue
            examples.append(ex)
        # Deterministic disjoint split, same convention as tasks/toolcalling.py.
        n_val = max(1, int(len(examples) * val_fraction))
        self.examples = examples[:n_val] if split == "val" else examples[n_val:]
        if max_examples is not None:
            self.examples = self.examples[:max_examples]
        self.length = len(self.examples)
        self.n_dropped = dropped

    def _convert(self, row):
        convs = row.get("conversations")
        if not isinstance(convs, list) or len(convs) < 2:
            return None
        sys_text = row.get("system") or ""

        # ToolACE puts the schema list in the system prompt; pull out the JSON
        # array so it can be rendered through render_tools_prompt (mask 0, and
        # with our explicit CALL-format instruction, which ToolACE never states).
        tools = _extract_schema_array(sys_text)
        if not tools:
            return None

        user_text = next((m["value"] for m in convs
                          if m.get("from") == "user" and isinstance(m.get("value"), str)), None)
        if not user_text:
            return None
        prompt = render_tools_prompt(tools, extra_instruction=sys_text.split("Here is a list of functions")[0].strip() or None)

        messages = [{"role": "user", "content": prompt}]
        for turn in convs:
            frm = turn.get("from")
            val = turn.get("value")
            if not isinstance(val, str):
                continue
            if frm == "assistant":
                calls = parse_toolace_call(val)
                if not calls:
                    continue
                # Emit each call as its own assistant turn so the reference
                # sequence is exactly what we train on; multiple calls in one
                # turn become multiple python parts (the renderer allows many
                # parts per assistant message).
                parts = [{"type": "python",
                          "text": toolace_to_json_args(n, a)} for n, a in calls]
                messages.append({"role": "assistant", "content": parts})
            elif frm == "tool" and self.keep_tool_results:
                messages.append({"role": "assistant",
                                 "content": [{"type": "python_output", "text": val}]})

        folded = _fold_turns(messages)
        if folded is None:
            return None
        if not any(isinstance(m["content"], list) and
                   any(p.get("type") == "python" for p in m["content"]) for m in folded):
            return None  # nothing to learn: no emitted call
        return {"messages": folded}

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return self.examples[index]

    def score(self, conversation, assistant_response):
        """Same partial-credit dict as tasks/toolcalling.py so numbers are
        comparable across the two corpora."""
        assert isinstance(assistant_response, str)
        ref = []
        for m in conversation["messages"]:
            if m["role"] == "assistant" and isinstance(m["content"], list):
                for p in m["content"]:
                    if p.get("type") == "python":
                        ref.append(_norm_call(p["text"]))
        if not ref:
            return {"exact": 0, "fn": 0, "count": 0, "n_ref": 0}
        pred = extract_predicted_calls(assistant_response)
        fn = sum(1 for r, p in zip(ref, pred) if p[0] == r[0])
        return {
            "exact": int(pred == ref),
            "fn": fn / len(ref),
            "count": int(len(pred) == len(ref)),
            "n_ref": len(ref),
        }

    def evaluate(self, conversation, assistant_response):
        return self.score(conversation, assistant_response)["exact"]


def _extract_schema_array(sys_text):
    """Pull the JSON array of function schemas out of ToolACE's system prompt.

    Cannot just use rfind(']'): the prompt ends with a *format instruction*
    containing brackets, e.g. "...Put it in the format of [func1(params_name=
    params_value...), func2(params)]", so the last ']' is not the end of the
    schema array. Instead decode from the first '[' with json.JSONDecoder's
    raw_decode, which returns the end offset of the value it consumed.
    """
    if not sys_text:
        return []
    start = sys_text.find("[")
    if start == -1:
        return []
    try:
        tools, _end = json.JSONDecoder().raw_decode(sys_text, start)
    except Exception:
        return []
    return tools if isinstance(tools, list) and tools else []


def _fold_turns(messages):
    """Make the turn list satisfy the renderer's strict alternation assert.

    - assistant `python` parts and `python_output` parts concatenate cleanly
      into one assistant message, which is exactly right: the model emits the
      call, the environment answers, the model emits the next call.
    - two adjacent assistant *text* turns are interposed with an empty user turn.
    """
    fixed = []
    for m in messages:
        if fixed and m["role"] == fixed[-1]["role"] == "assistant":
            prev = fixed[-1]
            if isinstance(prev["content"], list) and isinstance(m["content"], list):
                prev["content"] = list(prev["content"]) + list(m["content"])
                continue
            fixed.append({"role": "user", "content": "Continue."})
        fixed.append(m)
    if len(fixed) < 2 or fixed[0]["role"] != "user":
        return None
    if fixed[-1]["role"] != "assistant":
        fixed = fixed[:-1]
    if len(fixed) < 2:
        return None
    return fixed
