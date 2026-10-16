
## 2026-10-16 — Multi-turn tool-result follow-up fixed (v5e served on port 8000)

Root cause chain (all measured):
1. Stage 4 v3 `_truncate_terminal_call` dropped the 877/1036 hermes post-result
   prose turns => v5 had ZERO supervision for "tool result -> answer prose".
   Live: after a tool result the model echoed result JSON tails / re-emitted CALL.
2. Serving template rendered OpenAI `tool` messages as <|output_start|> AFTER
   <|assistant_end|> — out of distribution on top of the missing supervision.
3. Even with the result->answer transition learned, hermes has 0 rows with
   prose -> NEW user question -> another call, so the second user turn
   hallucinated instead of calling (glaive-v2 has 35,669 such rows).

Fixes (commits in this repo):
- chat_template.jinja v4: OpenAI `tool` message(s) render as a USER turn
  `Tool result: {content}`; consecutive tool messages merge into one turn
  (matches hermes bundled results). Single-turn behavior byte-identical to v3.
- tasks/toolfollowup.py: 881 train rows from hermes (result -> prose answer),
  first user turn = question + render_tools_prompt (Stage 4 exact).
- tasks/glaive_multiround.py: 33,484 train rows from glaive-v2 (24,297 with
  call rounds). Parser handles glaive's python-dict-repr call objects
  (balanced-brace extract + ast.literal_eval) and multi-assistant segments.
- chat_sft_long.py: --followup-epochs, --glaive-mr-epochs, --smoltalk-cap,
  --longctx-families flags.

Touch-up chain (all from the previous step's ckpt, ctx 8192, GPU 3):
  v5(800) -> v5b(120, FollowUp x3, lr 0.2x SFT) bpb .5518->.4220
          -> v5c(80, +Glaive x3 tiny) -> .4072  [loop T2 fixed, T3 not]
          -> v5d(100, Glaive x4 + SmolTalk cap 20k) -> .3993  [T3 fixed]
          -> v5e(60, + LongCtx multi_tool/niah/doc_qa L0-L1 re-anchor) -> .3977
  (bpb not comparable to v5's 0.5610: different val mixture, short rows dominate)

Gates on the served v5e (nanochat-d28b-rl100-128k-v5e, port 8000):
- 4-turn OpenAI loop: call -> result -> ANSWER USES DATA -> re-ask -> NEW CALL
  -> result -> answer. All correct (was impossible on v5).
- Conformant suite: 7/9 same as v5 (NIAH 24k/64k PASS, doc_qa 24k/64k PASS,
  multi_tool 24k PASS, tool short PASS; NIAH 105k = harness 131072-window
  limit, multi_tool 32k + long_summary remain open).
- Known quirks persist (pre-existing, not regressions): tool_choice=none leaks
  a CALL line as text; over-triggers with tools present; short no-tool chat
  degenerates. Parallel multi-call: emits calls concatenated in one args blob
  (parser artifact) — needs a Stage-4-style parallel-calls data pass to fix.

Artifacts: checkpoints chatsft_checkpoints/d28b-128k-sft-v5{b,c,d,e};
exports vllm_d28b/nanochat-d28b-rl100-128k-v5{b,c,d,e}; results
/data/rowel/test_128k_v5{d,e}_results.txt; serve logs vllm_serve_8000_128k_v5{b,c,d,e}.log
