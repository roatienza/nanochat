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
