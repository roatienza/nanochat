"""
Tool-result follow-up SFT task (fixes the multi-turn tool gap in v5).

Why this file exists
--------------------
Stage 4 v3 (`_truncate_terminal_call`, commit 8ef9092) deliberately ended every
tool trajectory at "call -> <|assistant_end|>" because the model could not stop
after calls. That fixed termination but removed ALL supervision for the
transition the OpenAI tool loop needs next: tool result -> prose answer. The v5
checkpoint therefore re-emits CALL ... or echoes the result JSON when it receives
a tool result (measured live on the port-8000 server, Oct 16 2026).

This task rebuilds the dropped transitions from the same hermes corpus, in the
serving shape we adopt in chat_template.jinja v4:

  user:   question + "\n\n" + render_tools_prompt(tools)     (exactly Stage 4)
  asst:   CALL name {args}                                   (python part)
  user:   Tool result: {response}                            (NEW: user turn)
  asst:   original hermes prose answer                       (supervised)

Two call topologies, because hermes bundles all results of a turn into ONE
`tool` message (1100 tool messages over 1893 rows, never split per call):

  shape S (sequential, 1 call per turn):  A[CALL] U[result] A[CALL] U[result] A[prose]
  shape P (parallel, N calls in one turn): A[CALL;CALL] U[merged results] A[prose]

Shape P matches how the serving template merges consecutive OpenAI `tool`
messages into a single user turn, so train and serve agree. Single-call rows
(346 of the 937) appear once, as shape S with a single round.

Results are the FULL hermes response text (the tool_call wrapper is stripped by
the role mapping, the inner content kept verbatim): at serving time the
environment's response is what the model must read, so training on the full
text is the honest distribution. Rows whose rendered length exceeds
--max-tokens are dropped (they would be truncated mid-answer and teach a
degenerate stop).

The first user turn is built with tasks.toolcalling.render_tools_prompt so the
tool-list format is byte-identical to Stage 4 training and to the v3 serving
template's tools block.
"""

import json
import re

from tasks.common import Task
from tasks.toolcalling import (
    DEFAULT_PATH,
    _loads,
    extract_calls,
    render_tools_prompt,
)

# hermes tool messages wrap each result in its own tool_response JSON block.
# The corpus embeds zero-width spaces inside these tags (see toolcalling.py),
# so translate before matching, and reuse toolcalling's exact regex so the
# result text matches what Stage 4 showed the model as python_output parts.
from tasks.toolcalling import _TOOL_RESP_RE, _ZERO_WIDTH


def parse_tool_response(text):
    """One hermes `tool` message -> list of (name_or_None, result_text) in order.

    result_text is the verbatim inner text of each tool_response block (the full
    {"name":..., "content":...} object) -- the same string shape Stage 4
    rendered as python_output, so the result format stays in-distribution.
    """
    text = (text or "").translate(_ZERO_WIDTH).strip()
    out = []
    for inner in _TOOL_RESP_RE.findall(text):
        inner = inner.strip()
        if not inner:
            continue
        name = None
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict):
                name = obj.get("name")
        except (json.JSONDecodeError, ValueError):
            pass
        out.append((name, inner))
    if not out:
        return [(None, text)] if text else []
    return out


class ToolFollowUp(Task):
    """
    Hermes rows that contain: tool call(s) -> tool result(s) -> prose answer.
    Rendered in the v4 serving shape (results as user turns).
    """

    def __init__(self, path=DEFAULT_PATH, split="train", val_fraction=0.05,
                 max_tokens=6500, max_examples=None, **kwargs):
        super().__init__(**kwargs)
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        examples = []
        for row in rows:
            ex = self._convert(row, max_tokens)
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

    def _convert(self, row, max_tokens):
        tools = _loads(row.get("tools"))
        if not isinstance(tools, list) or not tools:
            return None
        convs = row.get("conversations") or []

        # walk the hermes turns, collecting rounds: call-turn, result-turn pairs
        # and the final prose answer that follows the last result.
        rounds = []          # list of (calls_in_turn, [(name, content), ...])
        pending_calls = None
        prose_answer = None
        question = None
        for m in convs:
            role, val = m.get("from"), m.get("value") or ""
            if role == "human" and question is None:
                question = val.strip()
            elif role == "gpt":
                calls = extract_calls(val)
                if calls:
                    pending_calls = calls
                elif rounds:
                    # prose turn after at least one completed round -> final answer
                    prose_answer = val.strip()
                    break
            elif role == "tool" and pending_calls is not None:
                rounds.append((pending_calls, parse_tool_response(val)))
                pending_calls = None
        if not rounds or not prose_answer or not question:
            return None

        # build the conversation in the v4 serving shape
        messages = [{"role": "user",
                     "content": question + "\n\n" + render_tools_prompt(tools)}]
        for calls, results in rounds:
            # assistant turn: one python part per call, compact JSON (Stage 4 form)
            parts = [{"type": "python",
                      "text": f"CALL {name} {json.dumps(args, separators=(',', ':'))}"}
                     for name, args in calls]
            messages.append({"role": "assistant", "content": parts})
            # user turn: "Tool result: ..." with ALL of the round's results
            # merged into ONE turn, \n-joined. The renderer asserts strict
            # user/assistant alternation (no two user turns in a row), and the
            # serving template merges consecutive OpenAI tool messages into a
            # single user turn the same way -- so this is the train==serve shape.
            merged = "\n".join(c for _, c in results)
            messages.append({"role": "user", "content": f"Tool result: {merged}"})
        messages.append({"role": "assistant", "content": prose_answer})

        # length guard: rendered size must fit the touch-up context window
        est = sum(len(m["content"] if isinstance(m["content"], str)
                      else json.dumps(m["content"], ensure_ascii=False)) for m in messages)
        est = int(est * 0.4)  # chars -> tokens, conservative
        if est > max_tokens:
            return None
        return {"messages": messages}

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return self.examples[index]

    def evaluate(self, conversation, assistant_response):
        raise NotImplementedError("SFT data task; scoring not used in training")
