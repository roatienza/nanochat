
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

## 2026-10-17 — Same-tool over-call fixed (v5j served on port 8000)

The agent-harness failure: with 2+ rounds of the SAME tool in history the
model emitted another call with hallucinated args (10/10 at temp 0), and easy
questions over-triggered calls (5/5). Prompt-level fixes all fail (measured).

Probe matrix (probe_overcall.py / probe_overcall2.py, live on each checkpoint):
- P1 1-round -> prose, P2 2 rounds diff tools w/ prose between, P3 2 rounds
  same tool WITH prose+user turns between: all fine on v5e (0 over-calls).
- P7 BARE chained 2-round same tool (NO prose, NO user turn between rounds):
  10/10 over-call on v5e. P13 (q answered by round 1, chained anyway): 5/5.
- P5 easy question + tools present: 5/5 over-trigger with hallucinated args.

Root cause (data census): the v4 serving shape has ZERO training rows of
[user "Tool result: ..."] -> [assistant CALL]. Stage-4/toolace supervise
result->CALL only in the OLD rendering (results as python_output inside the
assistant turn; 612 hermes + 650 toolace rows). glaive prose-only rounds
(tools present, no call) are 1,650 of 33,484 (4.9%) — swamped, hence the
over-trigger. When the model itself emits a bare chain it goes
off-distribution and keeps chaining.

ALSO FOUND + FIXED: chat_template v4 joined multi-tool schemas with NO
newline (the {{ '\n' }} between tools was emitted at top level and lost);
Stage-4 training joins with '\n' (render_tools_prompt). Template v5 joins
inside the block; byte-match vs the training render confirmed under
transformers 5.17.

Retrain ladder (all from v5e ckpt 60, 60 steps, GPU 3, ctx 8192):
  v5f ToolStop x3 (bare-chain-then-stop 45/spurious 20/no-tool 20/1-round 15)
      -> P7 fixed 0/10 but P1/T2 REGRESSED (result->CALL over-taught)
  v5g rebalanced (38/35/15/12) -> P7 fixed, P1 still 5/5
  v5h C+D only (no-tool 50/1-round 50, ZERO result->CALL rows)
      -> P1..P4,P6,P9,P10 clean but P7 back to 10/10
  v5i (no-tool 70/1-round 30, x4) -> P5 over-trigger still 5/5
  v5j (+C' toolish-no-tool rows 35, x4) -> BEST: P7 0/10, P13 0/5,
      P1/P2/P3/P4/P6/P9/P10 clean, 4-turn OpenAI loop all correct
  v5k (C 50/C' 20/D 30, x2, longctx x3) -> P7 back to 10/10 (re-dominated);
      v5j redeployed.

Fundamental tension (measured): with [result][assistant_start] and no user
turn, a 2B model cannot locally distinguish "chain" from "stop". Any
result->CALL rows in the v4 shape flip P1; zero such rows flip P7. v5j's
toolish-no-tool rows shift the boundary enough to fix P7 while keeping
1-round behavior.

Gates on served v5j: over-call P7 0/10 (was 10/10), P13 0/5 (was 5/5),
P1/P2/P3/P4 0 over-calls, P6 5/5 calls, 4-turn loop correct end-to-end.
Conformant suite 6/9 (v5e was 7/9): multi_tool 24k REGRESSED to search_web
(C' over-generalization); NIAH 24k/64k, doc_qa 24k/64k, tool short PASS;
NIAH 105k = window limit; multi_tool 32k + long_summary pre-existing FAILs.

Residuals (logged, not fixed): P5 over-trigger (easy q + tools -> spurious
call, then answers the tool result not the question — client should drop
tools= for non-tool queries or accept one wasted call); bare 3-round P11
5/5 (artificial shape, unreachable when the model never chains); parallel
multi-call parser artifact; tool_choice=none leak.

Artifacts: tasks/toolstop.py; chat_template_v5.jinja (master + served);
run_v5{f..k}.sh, serve_v5{f..k}.sh, probe_overcall{,2}.py (vllm_d28b/);
checkpoints d28b-128k-sft-v5{f..k}; exports nanochat-d28b-rl100-128k-v5{f..k}
(config.json patched to 131072 after packing — exporter writes 8192);
results /data/rowel/test_128k_v5j_results.txt.


## 2026-10-17 (late) — multi_tool re-anchor: root cause found, v5r deployed

Task: serve the latest checkpoint supporting multi-tool call. v5j had regressed
multi_tool 24k (search_web instead of get_weather). Ladder v5l..v5q isolated the
mechanism; v5r fixes it and is now served on port 8000.

Ladder (all from v5e ckpt 60, 60 steps):
  v5l ToolStop + E doc-call rows (C35/C'25/D30/E10) -> multi_tool 24k/32k PASS
      BUT P7 over-call re-flipped 10/10. E rows re-tipped the call boundary.
  v5m rebalanced (C40/C'30/D20/E10) -> same: multi_tool PASS, P7 re-flipped.
  v5n exact v5j dist + multi_tool-only LongCtx oversample x2 (seed 4243)
      -> P7 fine but multi_tool 24k STILL search_web (count is not the lever).
  v5o weather-forced oversample (multi_tool_weather family, randrange pinned)
      -> still search_web for EVERY city (not city memorization; 4k passes,
      8k+ fails even with NO mention in the doc).
  v5p weather-flavored C' templates removed -> still search_web.
  v5q max-seq-len 24576 -> still search_web (level-1 rows are 31.2k tokens,
      still truncated -> still skipped).
  v5r max-seq-len 32768 -> multi_tool 24k PASS (get_weather/Oslo/Celsius),
      P7 0/10, P13 0/5, P1 5/5, 4-turn loop correct. DEPLOYED.

ROOT CAUSE (measured): every touch-up since v5f trained at max-seq-len 8192.
Level-1 multi_tool rows render to ~31.2k tokens (the 1.3 tok/word estimate is
off for this filler vocab), so render_conversation(8192) cut the assistant CALL
target off -> mask sum 0 -> refill() skipped the row. The touch-ups never saw a
single 24k doc->CALL row; the 24k behavior drifted to the search_web attractor
("{city} logistics bulletin" is a memorized training target). v5e's original
stage7b-v5 run trained at max_seq_len=131072 and saw the rows in full — hence
v5e passed 24k and every touch-up failed. The 4k (level-0) rows always fit,
which is why 4k kept passing everywhere.

Also fixed: --mt-extra-epochs flag (multi_tool-only LongCtx copies) and the
multi_tool_weather family (randrange-pinned weather branch) in
tasks/longctx_sft.py; _gen_doc_call (E shape) kept in tasks/toolstop.py
(unreferenced at 0%); C' weather templates replaced with neutral finance/time
ones (kept: they are harmless, but the E/C' levers were not the mechanism).

v5r gates: multi_tool 24k PASS, 32k FAIL (pre-existing, v5e also failed; the
32k eval renders ~41k tokens, beyond the touch-up's 31.2k rows), NIAH 24k/64k
PASS, doc_qa 24k/64k PASS, tool short PASS, long_summary FAIL (pre-existing),
P7 0/10, P13 0/5, P1 5/5, LOOP correct. Val bpb 0.392691 (v5j 0.425338 — the
24k val rows now fit and are supervised, so the numbers are not comparable).

Serving: nanochat-d28b-rl100-128k-v5r (served name nanochat-d28b-rl100),
GPU 7 (GPU 5 was crowded by other jobs), port 8000, template v5, parser
nanochat. Checkpoints/exports v5l..v5r kept for rollback.

Artifacts: run_v5{l..r}.sh; pack_v5{l..r}.sh, serve_v5{l..r}.sh,
chain_v5{l..r}.sh, verify_v5{l..r}.py, patch_config_131072.py (vllm_d28b/);
results /data/rowel/test_128k_v5{r}_results.txt + test_128k_v5j_results.txt
(overwritten by the v5r conformant run).


## 2026-10-18 (midday) - harness-replay fix: baby-onit resumed-session collapse, v5v deployed

Task: "using the latest checkpoint on an agent harness generates unreliable
answers. fix the issue." Harness: baby-onit (github.com/sibyl-oracles/baby-onit),
OpenAI-chat client, host 202.92.132.48:8000/v1. Symptom: after resuming a
273-turn session, every turn answered with canned refusals, document
continuations ("The 0-5 day run is over") or "You're welcome!" - never the
answer to the new turn.

Reproduction (probe_harness_repro.py, probe_harness_ablation.py in vllm_d28b/):
imports baby_onit itself so system prompt, history replay and tool schemas are
byte-identical to the harness's payload. Real session_history.jsonl (51 recs,
~14k tok) replayed as user/assistant pairs -> collapse on every vague turn.
Ablations (2 runs, consistent):
  synthetic NEUTRAL history, same shape/length -> clean greeting
    => NOT length, NOT pair shape, NOT tool schemas.
  last-2 real turns only -> still broken => recent technical prose triggers it.
  specific question over full replay -> answered => register-confusion, not
    destruction; vague turns fall into canned attractors.
  temp 0 -> identical canned output => attractor, not sampling noise.
Root cause: no training row ever shows the model's OWN long technical
assistant answers (RL-run reports, markdown tables, code blocks) replayed as
history followed by a short vague user turn with tools present. Stage-7b
history is short chit-chat; tool rows call immediately; longctx rows are
single-task docs. The harness shape is off-distribution.
(Also found: baby_onit.tool() keeps only the LAST param per tool schema -
verified harmless; do not chase.)

Fix: tasks/harness_replay.py (HarnessReplay). Rows in the EXACT harness shape:
harness-like system prompt with "(today: DATE)" (85%), 0-60 history turns of
technical register (markdown tables/code blocks, 300-800 tok answers,
user turns technical questions AND pasted-log style), tools block in the FIRST
user turn (template v5 placement), final turn shapes:
  H1 social-vague (30%) / H2 easy-knowledge (20%) / H3 toolish->CALL (15%,
  schema-driven mapping: weather->get_weather when the row's toolset has it,
  web_search otherwise) / H4 date-from-system (12%) / H6 toolish-no-call (13%)
  / H7 summarize-last (10%). Lengths med 1.9k / p90 6.9k / max 13k tok,
  0 mask-zero rows (the v5r truncation-skip bug cannot recur).
Trainer: --harness-epochs flag in scripts/chat_sft_long.py.

Ladder (all from v5e step 60, 60 iters, max-seq-len 32768, GPU 0, ~24 min each):
  v5s harness x3, toolstop x4            -> 3/8  (H-REPLAY 6/6 but P7 10/10)
  v5t + 20% P7-anchor rows, H3 15%       -> 1/8  (P1 0/5 REGRESSED - anchors
                                             hurt; reverted)
  v5u harness x2, toolstop x6, schema-   -> 3/8  (= v5s; MT24 still
       driven H3, H4 12%                          check_weather)
  v5v u + E-anchor (doc->CALL get_weather exchange at the END of 12% of
      histories)                         -> 5/8  DEPLOYED
  v5w v + --mt-extra-epochs 2            -> 3/8  (P7 re-flipped 10/10, same as
                                             v5l/v5m; rejected)

v5v gate (verify_v5v.py, live on port 8000):
  MT24 multi_tool ~24k : FAIL (check_weather - residual, queued)
  P7  bare 2-round     : 0/10 PASS     P13: 0/5 PASS
  P1  1-round answer   : 5/5 PASS      LOOP 4-turn: PASS
  H-REPLAY real 51-rec replay, 6 probes: 6/6 PASS (hello->greeting, 2+2->4,
      capital->Paris, weather/news->prose-or-call, no refusals, no continuations)
  H-FRESH fresh hello + weather tools: spurious 5/5 (P5 root; without weather
      tools in the list: 0/5 clean - client-side mitigation works)
  H-DATE date today + weather tools: calls get_weather (same root); WITHOUT
      tools: "Today is 2026-10-18." correct from the system prompt
  doc_qa 24k/64k exact-family probes: PASS/PASS
  NIAH 24k: PASS; NIAH 64k + MT-32k: FAIL (pre-existing 64k-class residuals)
Val bpb: v5s 0.403126, v5t 0.402814, v5u 0.401430, v5v 0.401558, v5w 0.401075.

Serving: nanochat-d28b-rl100-128k-v5v (served name nanochat-d28b-rl100),
GPU 0, port 8000, template v5, parser nanochat. Checkpoints/exports
v5s..v5w kept for rollback; v5r export intact.

Artifacts: tasks/harness_replay.py; run_v5{s..w}.sh (nanochat/);
pack_v5{s..w}.sh, serve_v5{s..w}.sh, verify_v5{s..w}.py, probe_v5v_longctx.py,
patch_hr*.py (vllm_d28b/); results /data/rowel/test_128k_v5{s,t,u,v,w}_results.txt.
Commit 23482b8, pushed to fork.
