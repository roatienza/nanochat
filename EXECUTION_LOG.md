2026-10-06 ~10:30 PDT — Stage 3 final ChatCORE measured; "regression" retracted

MEASUREMENT
  Ran scripts/chat_eval.py on the Stage 3 FINAL checkpoint
  (d28b-v2/model_015028.pt, val_bpb 0.308864) on GPU 1, mirroring the
  in-training ChatCORE loop exactly (categorical = full test set, generative
  = 24 samples, batch 2, temp 0.0, top_k 50, single Engine).
    ARC-Easy 24.62% (2376)  ARC-Challenge 26.54% (1172)  MMLU 24.65% (14042)
    GSM8K 0.00% (24)  HumanEval 12.50% (24)
    ChatCORE = 0.0272   ChatCORE_cat = 0.0036
  The Stage 3 final run (tpkzb8yr) used --chatcore-every=-1, so ChatCORE was
  never measured in-training (wandb summary chatcore_metric: N/A). This is the
  first real measurement of that checkpoint.

RETRACTION
  Every status file Oct 5-6 cited "Stage 3's final was 0.082" and claimed it
  was "verified from the Stage 3 log." It is NOT in any Stage 3 log (all three
  have zero ChatCORE: lines). The 0.0821 is wandb run 9w53jmjw (Sept 30):
    --run=sft-v1 --model-tag=d12 --total-batch-size=131072 --eval-every=100
  i.e. a d12 (0.286 B) ABLATION SFT at step 414 — different model, batch,
  scale. Comparing Stage 4 v2/v3 (d28) against it was apples-to-oranges.

CORRECTED COMPARISON (no regression)
  Stage 3 final model_015028.pt  ChatCORE 0.0272  (measured now)
  Stage 4 v2   model_015548.pt   ChatCORE 0.1367
  Stage 4 v3   model_013630.pt   ChatCORE 0.1360
  v2/v3 are ~0.11 ABOVE Stage 3 final, not below. v3 is statistically tied
  with v2 on ChatCORE and better on val_bpb (0.3022 vs 0.3148).

DECISION
  Stage 5 (RL) starts from model_013630.pt (v3). No blend needed.

  Detail: CORRECTION_2026-10-06_CHATCORE.md

2026-10-03 11:20 PDT — Stage 2 v2 status + repo sync

REPO
  branch main, HEAD ddb6176 == origin/main, 0 ahead / 0 behind. Nothing to pull
  or push before this commit.

RUN IS ALIVE
  PID 1297981 (torchrun) + 1299407-1299410 (one per GPU)
  started Thu Oct  1 20:17:56, etimes 140,514 s
  env from /proc/1299407/environ: CUDA_VISIBLE_DEVICES=4,5,6,7
                                NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2
  nvidia-smi compute-apps: the four PIDs hold GPUs 4,5,6,7 at ~38.5 GB each,
  matched by GPU UUID. GPUs 1,2 hold an unrelated 36.7 GB job; GPU 0 at 13.3 GB.

PROGRESS
  step 189500 / 277760 = 68.22%
  loss 2.400, lrm 0.51 (warmdown active since step 97,216)
  73.9k tok/s, 43.6% BF16 MFU, epoch 1
  379 validations logged

  val_bpb curve, run best 0.753907 @ 189,000
     5,000  0.963750     95,000  0.811202
    20,000  0.872783    120,000  0.793145
    40,000  0.840180    140,000  0.779938
    60,000  0.825719    160,000  0.768816
    80,000  0.816633    175,000  0.761084
   100,000  0.808317    185,000  0.755766
                    189,000  0.753907  <- best
  Monotone over the whole run. No turnaround. Finding 8 holds.

COMPARISON TO V1
  v1 best val_bpb 0.885878 @ step 24,000, reached within the first 49 of its
  106 logged validations. It died at 76,723 steps (27.6%) with val_bpb already
  deteriorated to 0.939570 @ 76,500.
  v2 is 0.131971 bpb ahead of v1's best, and v1 never got past 0.885878.
  The gap has widened monotonically all run: 0.064 @ 72k, 0.125 @ 176k,
  0.132 @ 189k.

ETA - wall clock, not the log field
  etimes 140,511 s over 189,500 steps = 1.3486 s/step = 61.79 min per 5k steps.
  Cross-checked against checkpoint mtime deltas, 5 consecutive blocks:
  61.83, 61.83, 61.80, 61.78, 61.83 min. Two independent sources agree.
  Remaining 88,260 steps = 18.18 h.  Finish ~ Oct 4, 05:30 PDT.
  The log's own eta: field reads 652.8m (10.9 h) and understates real time by
  40% because total_training_time accumulates only forward/backward dt,
  excluding eval, saves and wandb. Finding 12 stands.

  Correction: the 08:40 entry above said "Finish ~ Oct 3, 05:35 PDT". That was
  wrong by one day - 21.0 h from 08:40 PDT on Oct 3 is Oct 4, not Oct 3. The
  hour was right, the date was not. STATUS_2026-10-03.md said Oct 4 correctly,
  so the two files disagreed and EXECUTION_LOG.md was the one at fault.

CHECKPOINTS
  37 model_*.pt through model_185000.pt, in
  /data/rowel/nanochat_store/v2/base_checkpoints/d28b-v2/
  Store 504 GB.  / 191 GB free (89%).  /data 2.4 TB free (83%).
  ~18 saves remaining x 14.4 GB = 259 GB. No disk risk.
  Note the path is base_checkpoints/d28b-v2/, not base_checkpoints/ directly.
  The 316 GB base_checkpoints.tmp/ is v1's live store, not stale - do not delete.

STILL OPEN
  - No watchdog. Same single point of failure; a kill stops the run silently and
    the only recovery point is the last 5,000-step checkpoint.
  - --eval-tokens untuned: 2.56x training tokens. A 4x cut would recover ~28 h,
    but needs a restart from checkpoint. Not done without explicit approval.
===============================================================================
2026-10-03 12:40 PDT — Finding 16
===============================================================================

STATUS
  Stage 2 v2 (headline d28)  195,940 / 277,760 steps = 70.55%
  loss 2.547   lrm 0.48 (warmdown, active since step 97,216)
  73.9k tok/s   43.6% BF16 MFU   epoch 1 (of ~1.03)
  GPUs 4,5,6,7 @ 100%, 38.5 / 40.9 GB each. PIDs 1299407-1299410, live,
  etimes 145,245 s. uuid-matched to devices 4-7.

VALIDATION
  val_bpb 0.750556 @ 195,500 — run best, still falling.
  Curve 189k -> 195.5k, every eval:
    189000 0.753907 | 190000 0.753390 | 191000 0.752855 | 192000 0.752270
    192500 0.752285 | 193000 0.751896 | 193500 0.751427 | 194000 0.751238
    194500 0.751183 | 195000 0.750805 | 195500 0.750556
  Single blip at 192,500 (+0.000015) is noise. Monotone otherwise; Finding 8
  holds. ~55,000 steps of LR decay left, so a real tail gain is still expected.

VS V1
  v1 best 0.885878 @ 24,000 (106 validations, resume_24000.log). v2 is
  0.135322 bpb ahead. v1 had deteriorated to 0.939570 by step 76,723, where it
  died. Gap trajectory: 0.064 @72k, 0.125 @176k, 0.132 @189k, 0.135 @195.5k.

ETA (wall clock, NOT the log's eta field)
  Log says 605.2m = 10.1 h. Wrong by ~40% — total_training_time accumulates
  only fwd/bwd dt, excludes eval, saves, wandb. Finding 12 stands.
    process etimes: 145,245 / 195,940 = 0.7413 s/step
    ckpt mtime deltas: ~62 min per 5,000-step block, consistent all run
  81,820 steps left = 16.85 h  ->  Sun Oct 4, 05:29 PDT.

CHECKPOINTS
  39 model_*.pt through model_195000.pt (12:29) + 156 optim_*.pt
  Path: /data/rowel/nanochat_store/v2/base_checkpoints/d28b-v2/
  Store 532 GB.  / 191 GB free (89%).  /data 2.3 TB free (83%).
  16 saves left x 14.4 GB = 230 GB. No disk risk.

CORRECTIONS TO EARLIER ENTRIES
  - Log path. It is /data/rowel/serve_logs/stage2/restart170.log, not
    nanochat/out/. out/ does not exist. Every prior status in this repo cited
    out/ as the source. The figures were still correct — same log — but the
    recorded provenance was wrong. Located via /proc/1299407/fd/1.
  - Val field offset. Key is "Step N | Validation bpb: X", value in $6.
    Earlier spot-checks used $4/$5 and returned "Validation" / "bpb:" as the
    minimum. Caught and redone; every number above comes from $6.
  - base_checkpoints.tmp/ (316 GB) is v1's live store, still not stale.
    Do not delete.

STILL OPEN
  - No watchdog. Single point of failure, unchanged. Killed once already
    (Sep 30, full disk — since fixed by relocating store to /data).
  - --eval-tokens untuned: 2.56x training tokens, ~40% of wall clock.
    A 4x cut would recover ~28 h but needs a restart from checkpoint.
    Not done without explicit approval.

===============================================================================
2026-10-03 19:51 PDT — Finding 17: Stage 2 v2 at 83.06%, val_bpb 0.733716
===============================================================================

STATUS
  Stage 2 v2 (headline d28, 2.019 B params)  230,699 / 277,760 = 83.06%
  loss 2.418 (last logged line), lrm 0.30 (warmdown, active since step 97,216)
  73.5k tok/s, 43.4% BF16 MFU, epoch 1 (of ~1.03)
  GPUs 4,5,6,7 @ 99-100%, 38,546 / 40,960 MiB each (GPU7 38,034).
  PIDs 1299407-1299410 live, etimes 171,205 s, UUID-matched to devices 4-7.
  Env from /proc/1299407/environ: CUDA_VISIBLE_DEVICES=4,5,6,7
                               NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2

VALIDATION — 462 logged evals, run best 0.733716 @ 230,500
  Best per 25k block:
       0-25k  0.861983     125-150k 0.774648
     25-50k  0.832547     150-175k 0.761076
     50-75k  0.818663     175-200k 0.748355
    75-100k  0.808570     200-225k 0.736429
   100-125k  0.789909     225-250k 0.733716  <- run best
  Still monotone all the way down. No turnaround. Finding 8 holds.
  Last 8 evals: 228500 0.734558 | 229000 0.734293 | 229500 0.733941 |
               230000 0.733841 | 230500 0.733716

VS V1 (resume_24000.log, 106 evals)
  v1 best 0.885878 @ 24,000. v1's last logged eval before death: 0.939570
  @ 76,500, already deteriorating; died at 76,723 steps (27.6%).
  v2 is 0.152162 bpb ahead of v1's best. Gap trajectory:
    0.064 @72k -> 0.125 @176k -> 0.132 @189k -> 0.135 @195.5k ->
    0.139 @205k -> 0.147 @215k -> 0.152 @230.5k. Widening monotonically.

ETA — wall clock, NOT the log's eta field
  Log says 348.3m = 5.8 h. Wrong by ~67%. total_training_time accumulates
  only forward/backward dt, excluding eval, saves and wandb. Finding 12 stands.
    process etimes: 171,205 / 230,699 = 0.74211 s/step = 44.53 min per 5k
    checkpoint mtime deltas: 62 min per 5,000-step block, consistent all run
  47,061 steps left = 9.70 h  ->  Sun Oct 4, 05:33 PDT.

CHECKPOINTS
  46 model_*.pt through model_230000.pt (19:43) + 4 optim shards per save
  Path: /data/rowel/nanochat_store/v2/base_checkpoints/d28b-v2/
  v2 store 627 GB. / 192 GB free (89%). /data 2.2 TB free (84%).
  9 saves left x 14.4 GB = 130 GB. No disk risk.

  The Sep 30 kill cannot recur: root fs then had 18 GB free against a 14.62 GB
  checkpoint write (6.31 GB model + 4 x 2.08 GB optim). Now 192 GB free on /
  and the store lives on /data.

STILL OPEN
  - No watchdog. Single point of failure, unchanged. Killed once already
    (Sep 30, full disk — since fixed by relocating the store to /data). A kill
    still stops the run silently; the only recovery point is the last 5,000-
    step checkpoint. Now largely moot for this run: only 9.7 h remain, so even
    a full loss costs one restart from model_230000.pt.
  - --eval-tokens untuned: 2.56x training tokens, ~40% of wall clock.
    A 4x cut would recover ~28 h but needs a restart from checkpoint.
    Not done without explicit approval. Now moot for this run for the same
    reason — 9.7 h remain, a restart costs more than it saves. Do NOT
    action this on the current run.

===============================================================================
2026-10-04 07:55 PDT — Finding 18: Stage 2 v2 COMPLETE at step 277,760
===============================================================================

STAGE 2 IS DONE. Not a crash — verified.

  The process table shows no train.py. Before calling that a death I checked for
  a traceback: there is none. The log ends with a clean wandb teardown at 05:39,
  and meta_277760.json records step 277760 with min_val_bpb matching the log.
  The job completed the full schedule.

FINAL RESULT — d28, 2.019 B params
  step            277,760 / 277,760  = 100%
  val_bpb         0.7154372433401933  <- run best, attained on the LAST step
  CORE metric     0.2982
  train loss 2.363   lrm 0.05032   (final_lr_frac 0.05, warmdown complete)
  throughput 73,655 tok/s   MFU 43.47% BF16
  training time  2,055.27 m fwd/bwd ; 33.7 h wall clock
  peak memory    34,071.77 / 40,960 MiB (83%)
  epoch 1.03 (each shard seen ~once)   557 evals
  checkpoint     model_277760.pt + 4 optim shards, written 05:39

VALIDATION — best per 25k block, re-derived programmatically
     0- 25k  0.861983 @ 24,500      125-150k  0.774648 @ 149,000
    25- 50k  0.832547 @ 49,500      150-175k  0.761076 @ 174,500
    50- 75k  0.818663 @ 74,500      175-200k  0.748355 @ 199,500
    75-100k  0.808570 @ 99,500      200-225k  0.736429 @ 224,500
   100-125k  0.789909 @ 124,500     225-250k  0.725395 @ 249,500
                                   250-275k  0.716312 @ 274,500
                                        275k+  0.715437 @ 277,760  <- best
  Strictly monotone? No. 29 of 557 steps ticked up, mean +0.000098, max
  +0.000346. All noise; last blip at 75,500. Descending throughout warmdown.
  Finding 8 holds.

  527 of 557 evals set a new running best. Total improvement 2.4542 bpb
  (3.169678 at step 0 -> 0.715437). Warmdown began at step 97,216; best before
  it was 0.810519 at 96,500, so warmdown delivered 0.095082 of that. The LR
  tail was still paying off at the final step.

VS V1 (resume_24000.log)
                        v1                       v2
  steps        76,723 (27.6%)            277,760 (100%)
  best val_bpb 0.885878 @ 24,000         0.715437 @ 277,760
  val at death 0.939570 @ 76,500 (worse)
  tokens seen  2.52 B (21.1 passes)      9.102 B (1.03 passes over 9.150 B)
  v2 is 0.170441 bpb better than v1's best = 19.2% lower, on 3.6x the tokens.

WHY V2 WON — corpus size, not the LR schedule
  v1 pulled 8 shards = 0.431 B tokens against a 2.019 B-param model at
  param_data_ratio 8: 21.1 passes over the same data. Val turned up at ~40,000
  and reached 0.939570 by the step it died. v2 pulled 170 shards = 9.150 B
  tokens, 1.03 passes, became a ~1-epoch run, improved to the last step.
  I misdiagnosed this as LR-schedule trouble before the Sep 30 restart, costing
  ~12 h of redone work. Finding 8 (no warmdown turnaround) had been re-derived
  in the log but never reconciled against Finding 12 (dataset sizing).

OPEN ITEMS — both now moot, the run is finished
  - No watchdog: moot.
  - --eval-tokens untuned at 2.56x training tokens (~40% of wall clock): moot
    for this run. Still worth setting on Stage 3+ runs.
  Disk: / 192 GB free, /data 2.2 TB free. Sep 30 kill cannot recur.

CORRECTIONS TO EARLIER ENTRIES
  Three errors in my first draft of STATUS_2026-10-04.md, all caught by
  re-deriving from the log instead of carrying notes forward:
    1. Claimed all 557 evals improved. False — 29 ticked up.
    2. Tail gain stated as 0.2132 bpb. Actual 0.095082 (0.810519 at 96,500 is
       the pre-warmdown baseline, not a step-0 baseline).
    3. Duplicated the 125-150k row in the curve table.
  The "62 min per 5k block" rate in prior entries was measured on a mid-run
  window including eval load. Final span (Oct 3 08:40 to Oct 4 05:39) is
  20.98 h / 47,061 steps = 48.1 min per 5k. Both correct for their windows.

SOURCES
  /data/rowel/serve_logs/stage2/restart170.log
  /data/rowel/nanochat_store/v2/base_checkpoints/d28b-v2/meta_277760.json
  /data/rowel/serve_logs/stage2/resume_24000.log
  wandb run pgiv5whp

NEXT
  Stage 3 (SFT) is next in PLAN.md. Not started. Needs the d28 checkpoint
  model_277760.pt and a fresh wandb run.

========================================================================
STAGE 3 (SFT) — COMPLETE                                    2026-10-04 17:05
========================================================================
  Run v4 on GPUs 1,3,5,7. 15,028 steps, 123.5 min, 0 NaN.
  Final val_bpb 0.308864 (model_015028.pt). Clean exit.

  Killed runs: 1-GPU (disk-full), v3 (SIGINT @ 8,164; NaN was
  logging-only, weights never poisoned).

  RETRACTION: every "% complete" figure I reported for Stage 3 used
  a fabricated denominator. meta records num_iterations=-1, so
  chat_sft.py:465 never fires and the run stops on data exhaustion.
  The 15,012 "total" was an artifact of approx_progress, an estimate.
  Real final step: 15,028.

  NaN fix: f060833 in nanochat (local only, unpushed).

SOURCES
  /data/rowel/serve_logs/stage3/sft_d28_v4_4gpu.log
  /data/rowel/nanochat_store/v2/chatsft_checkpoints/d28b-v2/meta_015028.json
  wandb run tpkzb8yr

NEXT
  Stage 4 (tool calling) — built and verified in an earlier session
  (hermes-v1, masking verified token-by-token). Ready to launch.

========================================================================
STAGE 4 (TOOL CALLING) — RUNNING                          2026-10-04 23:15
========================================================================
  GPUs 3,4,5,7.  4-GPU, device-batch-size 2, grad_accum 2, batch identical
  to Stage 3 v4.  train dt 629 ms, 52.1k tok/s, 30.7% MFU.

  Stage 3 already trained tool-calling (hermes x2), so this stage could have
  been called done.  Measured on the Stage 3 checkpoint instead: 0.0% exact
  match, 46.5% right function name, 2.5% right call count.  Failures are
  verbatim-copy failures on the JSON arguments, and hermes has only 1,036
  unique train examples.  The constraint is data volume, not compute.

  xLAM-60k (named in PLAN.md) still gated: "Access denied. This repository
  requires approval."  Substituted ToolACE: 11,300 dialogs / 26,507 APIs,
  Apache-2.0, ungated -> 8,957 train / 182 val, 8.6x hermes.

  Masking verified on ALL 8,957 examples.  Four probe designs; v1/v2/v3
  false-alarmed 51/6/11 times on legitimate target content (argument keys named
  "parameters", enum values, URL templates).  v4 provenance probe: PASS.

  Fixed a latent bug from my own earlier work: _rebuild_fullsize_optimizer_state
  formatted args.model_step (None when --model-step omitted) into a filename.
  Only fires when SFT world size != pretrain world size, so Stage 3 missed it.

  CORRECTION: first attempt wasted 1,250 steps.  Default --eval-tokens
  20971520 => 1280 eval micro-steps => 263 s per eval vs 0.489 s/step train,
  i.e. 68% of wall-clock spent evaluating.  Relaunched at 4194304.  This is the
  dt-vs-wallclock error I made twice before, caught this time by measuring the
  eval boundary instead of reading the field.

  UNEXPLAINED: dt rose 492 -> 629 ms with only --eval-tokens changed, which
  cannot affect fwd/bwd.  Likely CPU contention from PID 765926
  (/data/rowel/robot-decisions, train_lora.py, 718% CPU) -- a different
  project, not mine, left untouched.  Ranks all at 99.3% CPU / 99-100% GPU.

SOURCES
  /data/rowel/serve_logs/stage4/stage4.log
  nanochat commit dccaa9a
  /data/rowel/reasoner-model/STATUS_2026-10-04_STAGE4.md

NEXT
  Stage 4 acceptance: eval_toolcalling.py held-out, compare to the 0.0%
  baseline recorded above.  Then Stage 5 (chat_rl.py on GSM8K).

2026-10-05 13:57 PDT  Stage 4 v2 launched.  GPUs 3,4,5,7, 490 ms/step, 66.5k
tok/s, 39.3% MFU, 818,702 rows.  Fix: collapse the final assistant turn to its
python parts so training targets end "call -> <|assistant_end|>".

  Two of my own earlier findings were wrong and are retracted:
  1. "0/54 never emits assistant_end" was an artifact of my probe --
     generate_batch() excludes terminal tokens (engine.py:294), so the check
     could never fire for ANY model.
  2. "ToolACE lacks the call -> END signal" is false: 99.8% already end on
     <|assistant_end|> and 93.2% already end call-only.

  Real defect is a repetition loop: exact match 0.0% but the reference is an
  exact PREFIX 43.8% of the time, with 11.12 predicted calls vs 1.88 reference.
  top_k=1 is byte-identical to top_k=0, so not a sampling pathology.

  nanochat e404ea0 -> roatienza/nanochat.  karpathy/nanochat untouched.
  Detail: STATUS_2026-10-05_STAGE4_V2.md
2026-10-05 16:35 PDT  Stage 4 v2 step-5,000 probe (corrected, mid-run).  The
acceptance metric moved off zero for the first time: exact match 3.7% (2/54)
on held-out hermes, vs 0.0% for Stage 4 v1 checkpoint 15560.  Real termination
(stopped before the token cap) 61.1% vs 13.0%; mean predicted calls 9.31 ->
7.33 against a reference of 2.20.  So truncate_to_terminal_call is doing what
it was meant to, but the repetition loop is NOT cured -- dumps still show
verbatim CALL repeats until the cap, plus token-level degeneration inside JSON
strings ("connect_to_all_ devices_ through_interface_interface...").  ToolACE
val is still 0.0% (18.39 predicted calls vs 1.91 reference).

  Measurement note: the old probe's "emitted assistant_end 0/54" was an
  artifact -- generate_batch() strips terminal tokens (engine.py:294), so that
  check is False by construction for any model.  Third retraction of that
  claim.  New probe_term_real.py measures "returned before the cap" instead.

  This is step 5,000 of ~15,533, pre-warmdown.  Run ETA ~2.2 h wall clock
  (~19:05 PDT); the v1 curve was still improving at 15,560 so this is an early
  read, not a verdict.  Re-probe the final checkpoint before accepting Stage 4.
  Detail: STATUS_2026-10-05_STAGE4_V2_PROBE.md
2026-10-05 21:15 PDT  Stage 4 v3 launched: hermes target-truncation fix (the
v2 root-cause finding applied to hermes, which v2 never got) + reweighted
mixture (tool rows 3.5% -> 9.3%).  GPUs 3,4,5,7, 66.5k tok/s, 39.3% MFU.
  Detail: STATUS_2026-10-05_STAGE4_V3.md
2026-10-06 09:30 PDT  Stage 4 v3 COMPLETE, ACCEPTANCE PASSES.  13,630 steps,
111.78 min, val_bpb 0.302180 (monotone after step-1k transient), clean
teardown.  Wide probe on final checkpoint, full populations, 5-GPU sharded:
hermes exact 0.0% -> 40.7% (22/54), fn accuracy 100.0%, mean pred calls
2.13 vs 2.20 ref, 96-98% stop before cap.  ToolACE exact 0.0% -> 2.2%, fn
78.6%, mean pred 1.90 vs 1.97 ref.  Budget 512 vs 1024 identical (token-cap
hypothesis stays dead).  Residuals carried to Stage 5: ToolACE argument
fidelity (78.6% fn).  ChatCORE did NOT regress: v3 0.1360 vs Stage 3 final
**0.0272** (measured 2026-10-06; the "0.082" baseline was a d12 ablation run
`9w53jmjw`, not Stage 3 final — see CORRECTION_2026-10-06_CHATCORE.md).
GSM8K chatcore 0.0 (24 samples) is a real weakness, shared with v2, for RL.  Checkpoint hygiene: v3 clobbered v2's 5000/10000
files in the d28b-v2 dir; v2's 15548 final intact.
  Detail: STATUS_2026-10-06_STAGE4_V3_FINAL.md
2026-10-10 09:35 PDT  Stage 6 agentic RL pilot COMPLETE + OFFLINE EVAL DONE.
  178/178 steps, 31.5 h on GPU 6, EXIT=0. Pass@1 0.067 -> 0.600 (+53 pts),
  Pass@2 0.120 -> 0.667. Curve monotone through step 120, then flat:
  120=140=160=150(offline)=177(offline)=0.600/0.667, per-block identical.
  Plateau CONFIRMED by offline eval of the two never-evaluated checkpoints
  (150, 177) on the same 150-instance held-out split — nothing hiding, no
  regression. Mastered 9 blocks (count_chars/math_fns/format_rows L1+L2,
  data_fix L2+L3, list_ops L1); zero on all L3s + list_ops@L2 + data_fix@L1
  (regressed at step 100, never recovered). Follow-on needs curriculum fix:
  replay mix of mastered blocks + frontier weighting toward L3/list_ops@L2.
  File-state gotcha found: chat_rl_agentic.py lines 136/209 in the CURRENT
  file lost their index subscripts; pyc bytecode forensics prove the file AS
  RUN had them (pilot numbers valid). Fix before any rerun.
  Detail: STATUS_2026-10-10_STAGE6_PILOT_FINAL.md
2026-10-13 09:15 PDT  Stage 6b (curriculum replay + scaffold) COMPLETE + OFFLINE EVAL DONE.
  178/178 steps, ~44 h on GPU 1, EXIT=0 (finished Oct 12 21:41). In-run eval
  curve: step0 0.600/0.667 -> steps 20-160 ALL 0.667/0.667 (Pass@1/Pass@2,
  150 held-out). Offline eval of ckpts 150+177 (GPU 6, finished 08:56):
  BOTH 0.667/0.667, per-block identical — confirms in-run curve, +6.7 pts
  Pass@1 over pilot final (0.600). Mastered 10/15 blocks: count_chars L1/L2,
  math_fns L1/L2, format_rows L1/L2, data_fix L1/L2/L3, list_ops L1.
  data_fix@L1 RECOVERED (1.00; regressed to 0 in pilot) — replay mix worked.
  Zero on 5: count_chars@L3, math_fns@L3, format_rows@L3, list_ops@L2/L3.
  Anneal at step 89 lifted scaffold; train reward dropped to 0.31-0.69 on
  uncracked blocks (no internalization), mastered set held. wandb
  stage6b-rl-curriculum run 8wof7d47. Checkpoints d28b-rl100: 25..177.
  Next mechanisms for the 5 dead blocks: repeated scaffold-anneal cycles,
  or within-block decomposition curriculum.
  Files: nanochat_store/v2/stage6b_offline_eval_results.json (+ .log).
