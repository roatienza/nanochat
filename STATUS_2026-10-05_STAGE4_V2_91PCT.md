# Stage 4 v2 — 91.1% (live, 19:22 PDT)

## Live process
| | |
|---|---|
| torchrun master | PID 1967506, up 5h26m |
| ranks | PIDs 1968002–1968005, all `Rsl` @ ~100% CPU |
| GPUs | 3,4,5,7 — 34,683 / 40,960 MiB each |
| wandb | run `4of7osv7` |
| checkpoint dir | `/data/rowel/nanochat_store/v2/chatsft_checkpoints/d28b-v2/` |
| last durable checkpoint | `model_015000.pt` (step 15000, val_bpb 0.317277) |

## Progress (measured, not copied from log)
| | |
|---|---|
| Step | 14,259+ (log pct 91.1%) |
| loss | ~1.03–1.06 (last block), 0 NaN |
| lrm | 0.18 (warmdown, from ~step 7,506) |
| throughput | 66.4–66.7k tok/s, 492 ms/step train, 39.1–39.4 MFU |
| wall rate | 0.9833 steps/s = 1.017 s/step (60 s sample) |
| ETA | **~0.38 h → ~19:45 PDT** |

Note: the run does not report a step total. Implied total from the log's
percentage field is ~15,650 — do not quote as a denominator.

## val_bpb: monotone across 11 consecutive evals
```
14000  0.3269
14250  0.3245
```
Recent block: 0.3589 (11.5k) · 0.3556 · 0.3518 · 0.3489 · 0.3454 · 0.3426
· 0.3393 · 0.3356 · 0.3328 · 0.3303 · 0.3269 · 0.3245.
No plateau, no reversal. Best is the most recent eval.

ChatCORE is noisy and AHEAD of Stage 3 final (0.0963–0.1030 vs Stage 3 final's
**0.0272**, measured 2026-10-06; the "0.082" cited here was a d12 ablation run — see
CORRECTION_2026-10-06_CHATCORE.md):
0.0774 (12.8k) · 0.1007 · 0.0964 · 0.1015 · 0.1030 · 0.1010 · 0.0963.
Expected — Stage 4's mixture is mostly MMLU/GSM8K repeats; tool calling is its job.

## Acceptance test — Stage 4 still OPEN
Probe on `model_005000.pt`, held-out hermes n=54, corrected probe:
| metric | v1 | v2 @5k |
|---|---|---|
| exact match | 0.0% | 3.7% (2/54) |
| returned before token cap | 13.0% | 61.1% |
| mean predicted calls | 9.31 | 7.33 (ref 2.20) |
| ToolACE exact | 0.0% | 0.0% (18.39 pred vs 1.91 ref) |

`truncate_to_terminal_call` did what it was designed to do. The repetition loop
is NOT cured. 3.7% is two examples, measured before warmdown.

**Must re-probe the final checkpoint before accepting Stage 4.**

## Corrections carried forward
1. `emitted assistant_end: 0/54` was an artifact — `generate_batch`
   (engine.py:294) strips terminal tokens. Reported twice as a model defect.
   `probe_term_real.py` measures "returned before the token cap" instead.
2. v1 @ 15,560 was NOT v1's final checkpoint; v1's last durable one is 10,000.
3. The log's `eta:` field excludes eval time. Real cost is ~3.83 min/eval
   (160.7 min / 42 evals), which is why every ETA from the log field was short.

## Contention note
GPUs 0, 2, 6 hold unrelated jobs (PID 765926 `train_lora.py` on GPU 0; 40–38 GB
resident on 2 and 6). Free for Stage 5: **GPU 1 only** (0 MiB).

## Next
1. Re-probe final checkpoint at ~19:45. This decides Stage 4.
2. Then Stage 5 (RL) — do not start before (1), or the repetition defect gets
   baked into RL and its cause becomes unisolatable.
