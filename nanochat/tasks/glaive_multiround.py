"""
Glaive-v2 multi-round tool-calling task (completes the tool loop).

Why this file exists
--------------------
tasks/toolfollowup.py (hermes) supervises: tool result -> prose answer. But
hermes has ZERO rows where the assistant answers in prose and then calls a tool
again for a NEW user question (census: 0/1893). Glaive-v2 has 35,669 of them.
Without this transition, a served agent loop hallucinates the second lookup
instead of calling (measured live on the port-8000 v5b server, Oct 16 2026).

Glaive-v2 format: {"system": "SYSTEM: ... {json schema list}", "chat":
"USER: ... ASSISTANT: <functioncall> {...} </s>\n\nFUNCTION RESPONSE: {...}\n\n
ASSISTANT: prose </s>\n\nUSER: ..."}.

This task renders it into the v4 serving shape (results as user turns), with
the tool list in the FIRST user turn via render_tools_prompt (same as Stage 4
and toolfollowup). Each round becomes:

  user:   question [+ tools block on round 0]
  asst:   CALL name {args}            (python part; <functioncall> parsed out)
  user:   Tool result: {response}     (FUNCTION RESPONSE content)
  asst:   prose answer                (supervised)

Rows are truncated at the first malformed round (kept prefix must still end in
prose to be usable). The schema list rides in the user turn (mask 0) exactly as
Stage 4 does; the CALL recipe instruction is appended by render_tools_prompt.
"""

import json
import re

from tasks.common import Task
from tasks.toolcalling import render_tools_prompt, _ZERO_WIDTH

_GLAIVE_PATH = "/data/rowel/data/glaive/glaive-function-calling-v2.json"

def _balanced_obj(text, start):
    """text[start] == '{' -> the balanced {...} string (quote-aware), or None."""
    depth, i, in_str = 0, start, None
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == chr(92):
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return None


def _load_call_obj(obj_str):
    """Glaive call objects are python-dict reprs (single-quoted 'arguments'),
    not strict JSON. Try json first, then ast.literal_eval."""
    try:
        return json.loads(obj_str)
    except json.JSONDecodeError:
        pass
    try:
        import ast
        obj = ast.literal_eval(obj_str)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError):
        return None


def _parse_fc(text):
    """One ASSISTANT turn -> list of (name, args_dict). Empty if prose-only."""
    out = []
    text = text.translate(_ZERO_WIDTH)
    for m in re.finditer(r"<functioncall>\s*", text):
        brace = text.find("{", m.end())
        if brace == -1:
            continue
        obj_str = _balanced_obj(text, brace)
        if obj_str is None:
            continue
        obj = _load_call_obj(obj_str)
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                try:
                    args = json.loads(args.replace("'", '"'))
                except json.JSONDecodeError:
                    try:
                        import ast
                        args = ast.literal_eval(args)
                    except (ValueError, SyntaxError):
                        continue
        if name and isinstance(args, dict):
            out.append((name, args))
    return out


def _extract_schema_array(system_text):
    """Glaive system prompt embeds a JSON array of function schemas."""
    text = (system_text or "").translate(_ZERO_WIDTH)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return arr if isinstance(arr, list) and arr else None


class GlaiveMultiRound(Task):
    """Glaive-v2 rows with >= 2 rounds incl. a result->prose->new-question->call."""

    def __init__(self, path=_GLAIVE_PATH, split="train", val_fraction=0.02,
                 max_tokens=6500, max_examples=None, max_rows=None, **kwargs):
        super().__init__(**kwargs)
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if max_rows:
            rows = rows[:max_rows]
        examples = []
        for row in rows:
            ex = self._convert(row, max_tokens)
            if ex is not None:
                examples.append(ex)
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
        tools = _extract_schema_array(row.get("system"))
        if not tools:
            return None
        chat = (row.get("chat") or "").translate(_ZERO_WIDTH)
        if "FUNCTION RESPONSE:" not in chat:
            return None

        # split the chat into USER segments; segment 0 is empty preamble
        segs = re.split(r"USER:\s*", chat)[1:]
        if len(segs) < 2:
            return None
        tools_prompt = render_tools_prompt(tools)
        messages = []
        for i, seg in enumerate(segs):
            # A segment may hold MULTIPLE assistant turns (glaive sometimes
            # emits a prose preamble before the call, or a clarifying
            # question). Split into ASSISTANT chunks and fold each chunk:
            # chunk with calls -> python parts; chunk with prose -> text.
            qm = re.split(r"ASSISTANT:\s*", seg, maxsplit=1)
            if len(qm) != 2:
                return None
            question, rest = qm[0].strip(), qm[1]
            if not question:
                return None
            chunks = re.split(r"ASSISTANT:\s*", rest)
            # chunk 0: [calls] </s> [FUNCTION RESPONSE: ...] -- the first turn
            first = chunks[0]
            has_resp = "FUNCTION RESPONSE:" in first
            calls = _parse_fc(first.split("FUNCTION RESPONSE:")[0]) if has_resp else _parse_fc(first)
            resp = None
            if has_resp:
                resp = first.split("FUNCTION RESPONSE:", 1)[1].replace("</s>", "").strip()
            # chunks 1+: later assistant turns (prose, or more calls)
            tail_prose = None
            tail_calls = []
            for ch in chunks[1:]:
                ch_calls = _parse_fc(ch)
                ch_clean = ch.replace("</s>", "").strip()
                if ch_calls:
                    tail_calls.extend(ch_calls)
                elif ch_clean:
                    tail_prose = ch_clean
            if calls:
                if not resp:
                    return None
                content = question + ("\n\n" + tools_prompt if i == 0 else "")
                messages.append({"role": "user", "content": content})
                all_calls = calls + tail_calls
                parts = [{"type": "python",
                          "text": f"CALL {name} {json.dumps(args, separators=(',', ':'))}"}
                         for name, args in all_calls]
                messages.append({"role": "assistant", "content": parts})
                messages.append({"role": "user", "content": f"Tool result: {resp}"})
                if tail_prose:
                    messages.append({"role": "assistant", "content": tail_prose})
            else:
                # prose-only round: question -> prose answer
                prose = tail_prose or first.replace("</s>", "").strip()
                if not prose:
                    return None
                content = question + ("\n\n" + tools_prompt if i == 0 else "")
                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": prose})
        if len(messages) < 4:
            return None

        est = sum(len(m["content"] if isinstance(m["content"], str)
                      else json.dumps(m["content"], ensure_ascii=False)) for m in messages)
        est = int(est * 0.4)
        if est > max_tokens:
            return None
        return {"messages": messages}

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return self.examples[index]

    def evaluate(self, conversation, assistant_response):
        raise NotImplementedError("SFT data task; scoring not used in training")