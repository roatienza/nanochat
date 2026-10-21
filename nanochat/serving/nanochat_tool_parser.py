"""utak tool-call parser for vLLM.

The model emits tool calls in its Stage-4 SFT format, inside python tags:

    <|python_start|>CALL <name> <json-args><|python_end|>

Detokenized output: the special tokens are dropped by the tokenizer's
`skip_special_tokens=True` default, so the text the parser sees is:

    CALL <name> <json-args>

(one per assistant turn; the model was trained to emit exactly one call per
reply, ending the turn on <|assistant_end|>).

Registered via --tool-parser-plugin as parser name "nanochat":

    vllm serve ... --enable-auto-tool-choice --tool-call-parser nanochat \
        --tool-parser-plugin /path/to/nanochat_tool_parser.py

`supports_required_and_named = False` routes tool_choice="required"/named
through the auto path (this parser), because the model cannot be
grammar-constrained into bare-JSON output.
"""

import json

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    DeltaToolCall,
    DeltaFunctionCall,
    ExtractedToolCallInformation,
    ToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import partial_tag_overlap

logger = init_logger(__name__)

# The literal markers, as they would appear if special tokens were NOT
# stripped (skip_special_tokens=False). We accept both spellings.
_PY_START_RE = r"(?:<\|python_start\|>)?"
_PY_END_RE = r"(?:<\|python_end\|>)?"

# A CALL line: "CALL name {json}" possibly with markers around it.
_CALL_LINE_RE = re.compile(
    r"(?:<\|python_start\|>)?\s*(CALL\s+[A-Za-z_][\w.\-]*\s*\{.*?\})\s*(?:<\|python_end\|>)?",
    re.DOTALL,
)


def _parse_call_line(line: str):
    """'CALL name {json}' -> (name, args_str) or None."""
    line = line.strip()
    if not line.startswith("CALL"):
        return None
    body = line[4:].strip()
    name, _, argstr = body.partition(" ")
    argstr = argstr.strip()
    if not name:
        return None
    return name, argstr


class NanochatToolParser(ToolParser):
    """Parses nanochat's 'CALL <name> <json-args>' tool-call format."""

    # The model cannot be grammar-forced into bare JSON; route
    # required/named tool_choice through the auto path (this parser).
    supports_required_and_named = False

    def adjust_request(self, request):
        # No-op: the base implementation would set structured_outputs
        # (grammar-forced bare JSON) for required/named tool_choice, which
        # this model cannot satisfy — it only emits 'CALL <name> <json>'
        # lines. Leaving the request unconstrained lets it produce its
        # native format, which this parser then handles.
        return request

    def __init__(self, tokenizer: TokenizerLike, tools: list | None = None):
        super().__init__(tokenizer, tools)
        self.tool_call_start_token = "<|python_start|>"
        self.tool_call_end_token = "<|python_end|>"
        # streaming state
        self._sent_content_idx = 0
        self._streamed_calls = []  # per call: {"name": str|None, "args_sent": str}

    # ------------------------------------------------------------------
    # non-streaming
    # ------------------------------------------------------------------
    def extract_tool_calls(self, model_output: str, request):
        # The model emits exactly one CALL per reply; be liberal and accept
        # several (one per line) in case of degenerate output.
        text = model_output or ""
        # find all CALL lines (with or without markers)
        raw_calls = []
        # split on python_end markers first, then scan lines
        chunks = re.split(r"<\|python_end\|>", text)
        for chunk in chunks:
            chunk = chunk.replace("<|python_start|>", " ")
            for line in chunk.splitlines():
                line = line.strip()
                if line.startswith("CALL"):
                    parsed = _parse_call_line(line)
                    if parsed:
                        raw_calls.append(parsed)
        if not raw_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        tool_calls = []
        for name, argstr in raw_calls:
            # validate JSON; if invalid, keep the raw string (client decides)
            try:
                json.loads(argstr)
            except (json.JSONDecodeError, TypeError):
                logger.warning("nanochat tool call has invalid JSON args: %r", argstr)
            tool_calls.append(
                ToolCall(
                    id=make_tool_call_id(),
                    type="function",
                    function=FunctionCall(name=name, arguments=argstr),
                )
            )
        # content = text before the first CALL line (usually empty)
        first_idx = text.find("CALL")
        content = text[:first_idx].strip() if first_idx > 0 else None
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content or None
        )

    # ------------------------------------------------------------------
    # streaming
    # ------------------------------------------------------------------
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ) -> DeltaMessage | None:
        try:
            text = current_text or ""
            # content = everything before the first CALL (or python_start)
            first_call_idx = text.find("CALL")
            first_marker_idx = text.find(self.tool_call_start_token)
            starts = [i for i in (first_call_idx, first_marker_idx) if i >= 0]
            content_end = min(starts) if starts else len(text)

            # hold back a suffix that could be a partial "CALL" opener
            holdback = 0
            if content_end == len(text) and "CALL" not in text:
                holdback = partial_tag_overlap(text, self.tool_call_start_token)
                # also hold back a partial "CALL" word
                for k in range(min(4, len(text)), 0, -1):
                    if text.endswith("CALL"[:k]):
                        holdback = max(holdback, k)
                        break
            sendable_content_end = content_end - holdback if holdback else content_end
            sendable_content_end = max(sendable_content_end, self._sent_content_idx)

            content = None
            if sendable_content_end > self._sent_content_idx:
                content = text[self._sent_content_idx:sendable_content_end]
                self._sent_content_idx = sendable_content_end

            # parse complete CALL lines from the text (scan the whole text;
            # the content region before the first CALL is excluded by the
            # line-level CALL filter anyway)
            tool_call_deltas = []
            chunks = re.split(r"<\|python_end\|>", text)
            calls = []
            for chunk in chunks:
                chunk = chunk.replace("<|python_start|>", " ")
                for line in chunk.splitlines():
                    line = line.strip()
                    if line.startswith("CALL"):
                        parsed = _parse_call_line(line)
                        if parsed:
                            calls.append(parsed)

            for i, (name, argstr) in enumerate(calls):
                if i >= len(self._streamed_calls):
                    # Only open a new call slot when the name is complete
                    # (a space follows it) or args have started; otherwise
                    # the line is still mid-name and we'd stream a truncated
                    # function name.
                    if not argstr:
                        break
                    self._streamed_calls.append({"name": None, "args_sent": ""})
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")
                st = self._streamed_calls[i]
                if st["name"] is None:
                    st["name"] = name
                    self.prev_tool_call_arr[i]["name"] = name
                    self.prev_tool_call_arr[i]["arguments"] = argstr
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(name=name).model_dump(
                                exclude_none=True),
                        )
                    )
                # stream new args by diff
                if len(argstr) > len(st["args_sent"]) and argstr.startswith(st["args_sent"]):
                    diff = argstr[len(st["args_sent"]):]
                    st["args_sent"] = argstr
                    self.streamed_args_for_tool[i] = argstr
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=diff).model_dump(
                                exclude_none=True),
                        )
                    )

            if content or tool_call_deltas:
                return DeltaMessage(content=content, tool_calls=tool_call_deltas)
            return None
        except Exception:
            logger.exception("Error in nanochat streaming tool call parsing")
            return None