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
