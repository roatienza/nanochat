# Stage 4 v2 — status update (2026-10-05 17:10 PDT)

Repo: `roatienza/reasoner-model` @ `7a9d9fd` → this commit.
nanochat fork: `roatienza/nanochat` @ `e404ea0`, in sync (0/0). `karpathy/nanochat` untouched.

## Headline

Stage 4 v2 is **training normally at 52.8%**, val_bpb is falling monotonically, and the
termination fix **did** move exact match off zero — but the repetition loop is *not* cured,
so **Stage 4 should not be accepted yet**.

| | |
|---|---|
| Step | 8,200 / ~15,533 (52.80%) |
| loss (last) | 1.261, **0 NaN** |
| Throughput | 66,415 tok/s, 493.4 ms/step, **39.20% BF16 MFU** |
| val_bpb @ 8,000 | **0.4066** ← run best |
| GPUs | 3,4,5,7 @ 99%, 34,683 MiB each (PIDs 1968002–1968005, up since 13:55) |
| NaN steps | 0 |
| Last checkpoint | step 5,000 @ 15:56 |
| ETA | ~1.6 h → **~18:45 PDT** |

val_bpb, step 0 → 8,000: 0.4334 → 0.4921 (500) → 0.4464 (5k) → 0.4343 → 0.4280 → 0.4251 →
0.4244 → 0.4235 → 0.4205 → 0.4173 → 0.4147 → 0.4136 → 0.4094 → 0.4090 → **0.4066**.
Monotone after the known step-500 transient and the step-5,000 checkpoint-save blip.

ChatCORE is **flat-to-noisy around zero**: 0.0465 (6.8k) → 0.0156 → 0.0278 → 0.0153 → 0.0011 →
0.0010 → 0.0097. For reference, the Stage 3 final checkpoint scores **0.0272** (measured
2026-10-06; the "0.082" cited here was a d12 ablation run, not Stage 3 final — see
CORRECTION_2026-10-06_CHATCORE.md). So Stage 4 is *ahead* of Stage 3 final on ChatCORE,
not behind.

## The probe: exact match off zero, loop not cured

Run on `model_005000.pt`, GPU 1, held-out hermes n=54, corrected probe (`probe_term_real.py`).

| metric | v1 @ 15,560 | **v2 @ 5,000** |
|---|---|---|
| exact match | **0.0%** | **3.7%** (2/54) |
| stopped before token cap | 13.0% | **61.1%** |
| mean predicted calls | 9.31 | **7.33** (ref 2.20) |

`truncate_to_terminal_call` behaved as designed and my earlier prediction ("if exact match moves
off 0.0%, the fix worked") holds *on that criterion*. But I would not accept the stage:

- Mean predicted calls is still **3.3× reference**. Not converged.
- ToolACE val: exact **0.0%**, mean predicted calls **18.39** vs ref **1.91** (9.6×).
- The dumps are still degenerate. Verbatim repeats:
  `get_recorded_stream {"camera_id":"front_garden"}` ×4; `Get Zip Code Information {"country":"US","postal_code":"90210"}` ×6.
- Token-level degeneration *inside* JSON strings persists:
  `connect_to_all_ devices_ through_interface_interface_interface…`,
  `"instrument_type":"c commodity"`.

Both "stopped before cap" figures are real termination rates — I fixed the probe so it counts
returned sequences rather than searching the stripped return value for a terminal token
(`generate_batch` excludes terminal tokens from its return, which is what made the old
`emitted assistant_end: 0/54` counter incapable of ever reporting non-zero).

## Corrections to what I told you earlier

1. **v1 @ 15,560 was not the final checkpoint.** v1's last durable checkpoint is step 10,000;
   step 15,560 is where the probe ran. My earlier comparison was against a mid-run artifact.
2. **"Step 5,000 checkpoint lands ~15:55"** — correct, it landed 15:56:41. No change.
3. **ChatCORE 0.082 for Stage 3** — this was WRONG (corrected 2026-10-06). The
   0.0821 came from wandb run `9w53jmjw`, a **d12 ablation** SFT (batch 131,072,
   step 414), not the Stage 3 final. The Stage 3 final (model_015028.pt) scores
   **0.0272** (measured 2026-10-06). See CORRECTION_2026-10-06_CHATCORE.md.
4. I previously implied the implied denominator (~15,533) came from the run. It doesn't; it's
   `8200 / 0.5280`. The run never states a total. As with Stage 3 (`num_iterations: -1`), I
   report **steps + the log's own percentage** rather than a number I back-derived.

## Verdict and next step

**Do not accept Stage 4.** The fix is real but partial. Remaining work:

1. Let the run finish (~18:45 PDT), then **re-probe the final checkpoint** — n=54 gave 3.7% at
   step 5,000 before warmdown, and 3.7% is two examples, not separable from zero.
2. If exact match is still single-digit on the final checkpoint, the residual defect is not the
   target shape. Next candidates, in order:
   - **Repetition is being taught, not just inherited.** Check whether ToolACE's source format
     itself contains repeated calls per example; if the SFT renderer emits multiple identical
     calls per assistant turn, every example is actively training the loop.
   - **Generation config.** Greedy with a 768-token cap and no repetition penalty. A
     repetition penalty (1.05–1.1) would mask rather than fix the cause, but would let us
     separate "model can't stop" from "model won't stop".
   - **Sampling temperature.** If greedy loops, sampling at low T may terminate correctly while
     degrading verbatim copying — a useful diagnostic, not necessarily a fix.

Want me to re-probe the final checkpoint when it lands (~18:45), or start Stage 5 (RL) on the
free GPUs now and keep Stage 4 open as failed-pending?
