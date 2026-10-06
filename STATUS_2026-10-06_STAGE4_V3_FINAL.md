# Stage 4 v3 — COMPLETE, ACCEPTANCE PASSES (2026-10-06)

Run: `stage4-toolace-d28-v3-termcall-reweight` (wandb `qm00mjt0`)
Finished: 2026-10-06 03:24 PDT, clean wandb teardown, no signal, no traceback.

## Run summary

| | |
|---|---|
| Steps | 13,630 (data-driven stop, `num_iterations=-1`) |
| val_bpb (min, last step) | **0.302180** |
| ChatCORE | 0.1360 (vs Stage 3 final 0.082 — see note below) |
| Train time | 111.78 min |
| GPUs | 3,4,5,7 (released) |
| Checkpoint | `d28b-v2/model_013630.pt` (note: v3 overwrote v2's 5000/10000-step files; v2's 15548 file is intact) |

val_bpb: 0.4314 (0) → 0.4542 (1k, known transient) → 0.4318 → 0.4205 → 0.4095 → 0.4009 → 0.3942 → 0.3888 → 0.3729 → 0.3582 → 0.3439 → 0.3300 → 0.3170 → 0.3060 → **0.3022**. Monotone after the step-1k blip.

## The fix that changed everything

v2's root-cause probe (`STATUS_2026-10-05_STAGE4_ROOTCAUSE.md`) showed the model
was *not* failing to learn tool calling — it was failing the **call → end-of-turn**
transition. All 1,036 hermes train conversations attach tool results to the
calling turn, so the model never saw a call turn ending in
`<|assistant_end|>`. v2 fixed this for ToolACE only; hermes was 1,036 of 818,702
rows and never got the fix.

v3 (nanochat `8ef9092`): hermes call-only turns 0% → 100%, end on
`<|assistant_end|>` 99.9%, calls/conversation preserved (2.00 → 2.00), val split
left untruncated (verified identical 54/54). Mixture reweighted: tool rows
3.5% → 9.3%.

## Acceptance probe — `model_013630.pt`, 5-GPU sharded, full populations

| | hermes (n=54, full val) | ToolACE (n=182) |
|---|---|---|
| **exact match @512** | **40.7%** (22/54) | **2.2%** (4/182) |
| exact match @1024 | 40.7% (22/54) | 2.2% (4/182) |
| stopped before cap | 96.3% / 98.1% | 99.5% / 99.5% |
| fn accuracy | **100.0%** | 78.6% |
| mean pred calls vs ref | 2.13 vs 2.20 | 1.90 vs 1.97 |

Budget 512 vs 1024: identical — the token-cap hypothesis stays dead.

### Trajectory across the four Stage 4 runs

| run | hermes exact | ToolACE exact | hermes mean pred |
|---|---|---|---|
| Stage 3 ckpt (baseline) | 0.0% | — | — |
| v1 @15560 | 0.0% | 0.0% | 9.31 (looping) |
| v2 @15548 | 0.0% | 0.0% | 5.81 |
| **v3 @13630** | **40.7%** | **2.2%** | **2.13** |

Acceptance criterion (set in `STATUS_2026-10-04_STAGE4.md`): "the run is only
worth anything if exact match moves off 0.0%." **Both suites moved off zero;
hermes is at 40.7% with 100% function-name accuracy and call count within 0.07
of reference. Stage 4 passes.**

## What remains (carried to Stage 5)

1. **ToolACE exact match is 2.2%, not 40%.** The hermes fix did not transfer:
   ToolACE fn accuracy is 78.6% (hermes 100%). ToolACE's schema/argument
   distribution is different (multi-arg, nested) and the model still gets
   arguments wrong there. This is the RL stage's job — RL rewards exact calls,
   which is exactly the residual defect.
2. **ChatCORE regressed 0.082 → 0.136.** The reweight (MMLU ×3→×1, GSM8K
   ×4→×1) traded general chat ability for tool ability. GSM8K chatcore is 0.0.
   Whether Stage 5 RL (which runs on a chat core) tolerates this needs a
   decision before launch.
3. **Checkpoint hygiene:** v3 wrote into the `d28b-v2` directory and clobbered
   v2's 5000/10000 checkpoints. v2's final (15548) is intact. Use distinct
   model tags per run going forward.

## Sources

- /data/rowel/tmp/s4v3.log
- /data/rowel/probe_results/v3_13630/*.json (5 shards × 2 tasks)
- /data/rowel/nanochat/wandb/run-20261005_211513-qm00mjt0/
- ~/.cache/nanochat/chatsft_checkpoints/d28b-v2/meta_013630.json
