"""
Tool calling SFT data task (Stage 4 of PLAN.md — "the one stage you must write").

Why this file exists
--------------------
nanochat's stock SFT mixture is SmolTalk(smol) + MMLU + GSM8K. The smol-smoltalk
card states it deliberately EXCLUDES function calling for small models, so nothing in
the stock mixture teaches the model to emit a call. PLAN.md's Stage 4 is therefore
the one stage that is real data engineering, and this file is it.

Design decisions (the ones that matter)
----------------------------------------
1) *Loss lands only on the emitted call.* The JSON schema of the available tools is
   huge and is pure conditioning — it belongs in the prompt (mask 0), never in the
   target. nanochat's `render_conversation` gives us exactly this for free: a user
   message is always masked 0, and assistant text is always masked 1. So the tool
   list rides in the user turn and the call rides in the assistant turn. We never
   have to touch the tokenizer.

2) *Tool results are masked out.* A `tool` role message is not a role nanochat's
   tokenizer knows about, so we fold the result back into the conversation as an
   assistant-visible but *unsupervised* part, using the existing `python_output`
   part type, which `render_conversation` already emits with mask 0. That keeps
   multi-turn trajectories trainable without teaching the model to hallucinate
   tool outputs (which at inference time come from the environment, not the model).

3) *We re-derive the call from the record, not by regex on the target.* The hermes
   records embed zero-width joiners in their <tool_call> tags. Parsing the
   assistant string is brittle. Instead we parse the *tools* field for names/schemas
   and parse the assistant turn only to recover which name+arguments to emit.

License: hermes-function-calling-v1 is Apache-2.0. (xLAM-60k, which PLAN.md names,
is gated on the Hub and was not accessible from this node — see EXECUTION_LOG.md.)
"""

import json
import os
import re

from tasks.common import Task

# The hermes corpus uses a zero-width space inside its tool-call tags; normalize.
_TOOL_CALL_RE = re.compile(r"<\s*tool_call\s*>(.*?)<\s*/\s*tool_call\s*>", re.DOTALL)
_TOOL_RESP_RE = re.compile(r"<\s*tool_response\s*>(.*?)<\s*/\s*tool_response\s*>", re.DOTALL)
_ZERO_WIDTH = dict.fromkeys(map(ord, "‌‍﻿"), None)

DEFAULT_PATH = os.environ.get(
    "TOOLCALL_DATA",
    "/home/rowel/baby-sandbox/data/hermes/func-calling.json",
)


def _loads(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def extract_calls(assistant_text):
    """Return a list of (name, arguments_dict) parsed out of one assistant turn."""
    text = (assistant_text or "").translate(_ZERO_WIDTH)
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        obj = _loads(m.group(1))
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments")
        if isinstance(args, str):
            args = _loads(args)
        if name and isinstance(args, dict):
            calls.append((name, args))
    return calls


def render_tools_prompt(tools, extra_instruction=None):
    """
    Put the tool schemas in the *user* turn so the loss never sees them.
    This is the single most important recipe detail in the stage: a 2B model given
    5 schemas will otherwise burn its capacity memorizing schemas it never emits.
    """
    lines = [json.dumps(t, separators=(",", ":")) for t in tools]
    prompt = (
        "You have access to the following tools:\n"
        + "\n".join(lines)
        + "\n\nTo call a tool, reply with a single line of the form "
          "CALL <name> <json-args>. Call exactly one tool per reply."
    )
    if extra_instruction:
        prompt = extra_instruction + "\n\n" + prompt
    return prompt


class ToolCalling(Task):
    """Function-calling SFT data rendered into nanochat's chat format."""

    def __init__(self, path=DEFAULT_PATH, split="train", val_fraction=0.05,
                 max_examples=None, **kwargs):
        super().__init__(**kwargs)
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        examples = []
        for row in rows:
            ex = self._convert(row)
            if ex is not None:
                examples.append(ex)
        # deterministic split so eval never leaks into training
        n_val = max(1, int(len(examples) * val_fraction))
        if split == "val":
            examples = examples[:n_val]
        else:
            examples = examples[n_val:]
        if max_examples is not None:
            examples = examples[:max_examples]
        self.examples = examples
        self.length = len(examples)

    def _convert(self, row):
        """One hermes row -> a nanochat Conversation (or None if unusable)."""
        tools = _loads(row.get("tools"))
        if not isinstance(tools, list) or not tools:
            return None
        convs = row.get("conversations") or []
        messages = []
        pending = []  # assistant tool calls awaiting a result

        for m in convs:
            role, val = m.get("from"), m.get("value") or ""
            if role == "human":
                # flush any unanswered call into a text part so structure is kept
                if pending:
                    messages.append({"role": "assistant", "content": [
                        {"type": "text", "text": t} for _, t in pending]})
                    pending = []
                tools_prompt = render_tools_prompt(tools)
                if messages:
                    messages.append({"role": "user", "content": tools_prompt})
                else:
                    messages.append({"role": "user",
                                     "content": val + "\n\n" + tools_prompt})
            elif role == "gpt":
                calls = extract_calls(val)
                if calls:
                    parts = []
                    for name, args in calls:
                        parts.append({"type": "python",
                                      "text": f"CALL {name} {json.dumps(args, separators=(',', ':'))}"})
                    pending = [(name, json.dumps(args, separators=(",", ":")))
                               for name, args in calls]
                    messages.append({"role": "assistant", "content": parts})
                else:
                    if pending:
                        pending = []
                    messages.append({"role": "assistant", "content": val})
            elif role == "tool":
                # Tool results are environment output: render unsupervised.
                text = (val or "").translate(_ZERO_WIDTH)
                resp = _TOOL_RESP_RE.findall(text)
                if not resp:
                    resp = [text.strip()]
                out = [{"type": "python_output", "text": r.strip()} for r in resp if r.strip()]
                if out and messages and messages[-1]["role"] == "assistant":
                    # attach the result to the assistant turn that made the call
                    prev = messages[-1]
                    if isinstance(prev["content"], list):
                        prev["content"] = list(prev["content"]) + out
                # a tool turn must be followed by user or assistant; loop handles it

        if pending:
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": t} for _, t in pending]})
        # nanochat's renderer asserts strict user/assistant alternation starting at
        # user. The hermes corpus is NOT strictly alternating: a `tool` turn is
        # followed directly by a `gpt` turn (both map to assistant). We fold each
        # same-role run into the previous turn, and for two adjacent *text*
        # assistant turns we interpose an empty user turn, because the renderer
        # has no notion of "assistant speaks twice" and would assert.
        fixed = [messages[0]]
        for m in messages[1:]:
            if m["role"] == fixed[-1]["role"]:
                prev = fixed[-1]
                if isinstance(prev["content"], list) and isinstance(m["content"], list):
                    # python / python_output parts concatenate cleanly
                    prev["content"] = list(prev["content"]) + list(m["content"])
                    continue
                # two adjacent assistant text turns: interpose an empty user turn
                fixed.append({"role": "user", "content": "Continue."})
                fixed.append(m)
                continue
            fixed.append(m)
        if len(fixed) < 2 or fixed[0]["role"] != "user":
            return None
        if fixed[-1]["role"] != "assistant":
            fixed = fixed[:-1]  # renderer wants to end on the target
        if not any(isinstance(m["content"], list) and
                   any(p.get("type") == "python" for p in m["content"]) for m in fixed):
            return None  # no actual tool call to learn from
        return {"messages": fixed}

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return self.examples[index]

    def evaluate(self, conversation, assistant_response):
        """
        Does the sampled response call the same function with the same arguments?
        Exact-match on (name, args) — the right metric for a 2B model on this task.
        """
        assert isinstance(assistant_response, str)
        ref_msgs = conversation["messages"]
        ref = []
        for m in ref_msgs:
            if m["role"] == "assistant" and isinstance(m["content"], list):
                for p in m["content"]:
                    if p.get("type") == "python":
                        ref.append(p["text"])
        if not ref:
            return 0
        pred = [ln for ln in assistant_response.splitlines() if ln.strip().startswith("CALL")]
        return int(len(pred) == len(ref) and all(
            _norm_call(p) == _norm_call(r) for p, r in zip(pred, ref)))


def _norm_call(line):
    """Normalize a 'CALL name {json}' line for robust comparison."""
    line = line.strip()
    if not line.startswith("CALL"):
        return line
    body = line[4:].strip()
    name, _, argstr = body.partition(" ")
    try:
        args = json.loads(argstr)
        args = {k: args[k] for k in sorted(args)}
        return (name, json.dumps(args, sort_keys=True))
    except json.JSONDecodeError:
        return (name, argstr)


if __name__ == "__main__":
    t = ToolCalling(split="train")
    print(f"train examples: {len(t)}")
    v = ToolCalling(split="val")
    print(f"val examples:   {len(v)}")
    ex = t[0]
    print("\n--- example 0 ---")
    for m in ex["messages"]:
        if isinstance(m["content"], str):
            print(f"[{m['role']}] {m['content'][:220]}")
        else:
            for p in m["content"]:
                print(f"[{m['role']}/{p['type']}] {p['text'][:220]}")
    print("\n--- masks (1 = supervised) ---")
    from nanochat.tokenizer import get_tokenizer
    tok = get_tokenizer()
    ids, mask = tok.render_conversation(ex)
    sup = sum(mask)
    print(f"total tokens {len(ids)}, supervised {sup} ({sup/len(ids):.0%})")
