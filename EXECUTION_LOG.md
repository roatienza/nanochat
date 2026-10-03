2026-10-03 08:40 PDT — Stage 2 v2 status + repo sync

REPO
  branch main, HEAD 65a102a == origin/main, 0 ahead / 0 behind. Nothing to pull or push.
  Corrects a tooling error in earlier updates: the sync check was run against
  origin/master, a ref that does not exist in this repo. Re-checked against origin/main.

RUN IS ALIVE
  PID 1297981 (torchrun) + 1299407-1299410 (one per GPU)
  env from /proc/1299407/environ: CUDA_VISIBLE_DEVICES=4,5,6,7
                                NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2
  nvidia-smi compute-apps: the four PIDs hold GPUs 4,5,6,7 at ~38.5 GB each

PROGRESS
  step 176045 / 277760 = 63.36%
  loss 2.406, lrm 0.58 (warmdown active since step 97,216)
  73.9k tok/s, 43.6% BF16 MFU, epoch 1
  353 validations logged

  val_bpb curve, run best 0.760384 @ 176,000
    19,500  0.874233
    39,500  0.840466
    59,500  0.825941
    79,500  0.816859
    99,500  0.808570
    119,500 0.793267
    139,500 0.780257
    159,500 0.768950
    176,000 0.760384
  Monotone, still improving through warmdown. Finding 8 holds.

COMPARISON TO V1
  v1 (run 4g748kis) best val_bpb 0.885878 @ step 24,000, reached in 49 of its
  277,760 steps - it died at 76,723 steps (27.6%) with val already deteriorating.
  v2 is 0.125494 bpb ahead of v1's best, and v1 never got past 0.885878.

ETA - wall clock, not the log field
  etimes 130,570 s over 176,045 steps = 0.7419 s/step = 61.8 min per 5k steps.
  Remaining 101,760 steps = 21.0 h.  Finish ~ Oct 3, 05:35 PDT.
  The log's own eta: field reads 752.3m (12.5 h) and understates real time by
  ~40% because it accumulates only forward/backward dt, excluding eval and saves.
  Finding 12 stands.

CHECKPOINTS
  35 model_*.pt through model_175000.pt, in
  /data/rowel/nanochat_store/v2/base_checkpoints/d28b-v2/
  Note the path is base_checkpoints/d28b-v2/, not base_checkpoints/ directly -
  the earlier updates in this log that reported "0 checkpoints" or grepped the
  parent dir were wrong.
  477 GB used.  / 194 GB free.  /data 2.4 TB free.
  21 saves remaining x 14.4 GB = 302 GB. No disk risk.

STILL OPEN
  - No watchdog. Same single point of failure; a kill stops the run silently.
  - --eval-tokens untuned: 2.56x training tokens. A 4x cut would recover ~28 h,
    but needs a restart from checkpoint. Not done without explicit approval.
