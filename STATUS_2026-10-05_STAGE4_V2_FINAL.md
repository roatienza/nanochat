# Status — 2026-10-05, Stage 4 v2 COMPLETE (final probe)

## Headline

Stage 4 v2 finished cleanly at **19:53** and the final-checkpoint probe is in.
**Termination is cured. Exact match is back to 0.0%. Stage 4 FAILS acceptance.**

This is a *worse* headline than step 5,000 showed, and it is the number that counts.

## Run result

| | |
|---|---|
| Final step | **15,548** (data-driven stop; meta `num_iterations: -1`) |
| val_bpb | **0.31480308627713977** |
| Wall clock | **127.63 min** |
| Peak memory | 30,612 MiB |
| GPUs | 3,4,5,7 — now released |
| wandb | [4of7osv7](https://wandb.ai/rowel/nanochat-sft/runs/4of7osv7) |
| Checkpoint | `chatsft_checkpoints/d28b-v2/model_015548.pt` + 4 optim shards |

ChatCORE at exit: **0.13667** (ARC-C 0.329, ARC-E 0.409, GSM8K 0.125, HumanEval 0.125, MMLU 0.337).
Note this is *below* Stage 3's 0.082 on a worse scale — ChatCORE is flat/weak for this run and is
not the acceptance signal. The tool probe is.

## Acceptance probe — `model_015548.pt`, GPU 1, n=54 each, budget 512

| metric | v1 @5,000* | v2 @5,000 | **v2 @15,548 (final)** |
|---|---|---|---|
| hermes exact match | 0.0% | 3.7% | **0.0%** |
| hermes STOPPED before cap | 13.0% | 61.1% | **94.4%** |
| hermes mean pred calls | 9.31 | 7.33 | **5.81** (ref 2.20) |
| ToolACE exact match | 0.0% | 0.0% | **0.0%** |
| ToolACE STOPPED before cap | — | — | **98.1%** |
| ToolACE mean pred calls | 18.39 | — | **2.48** (ref 1.91) |

\* the v1 column is the mid-run artifact I flagged earlier; v1's true final is `model_015560.pt`.

## What this actually says

**The training fix worked, and it was not enough.**

1. **Termination is genuinely fixed.** 94.4% / 98.1% stop before the token cap, up from 13.0%.
   The `truncate_to_terminal_call` fix does what it claims — generations end.
2. **But over-call is the real defect, and it is not a termination defect.** With a hard stop
   enforced, the model emits **5.81 calls against a 2.20 reference** and simply runs out of budget
   mid-way. It is not looping forever any more; it is producing *too many correct-shaped calls*
   and getting truncated.
3. **ToolACE mean calls is nearly correct (2.48 vs 1.91)** yet exact match is still 0.0%. So the
   failure there is argument fidelity, not call count — see ex1: it alternates
   `"model":"Model S"` / `"model":"Prius"`, i.e. it is sampling plausible values instead of copying
   the schema verbatim.
4. **hermes regression 3.7% -> 0.0%** is almost certainly **n=54 noise**: 3.7% was 2 examples. Two
   examples cannot carry a trend, and the run-best val_bpb improved monotonically throughout
   warmdown. I am explicitly *not* claiming warmdown destroyed it; I am saying n=54 is too small
   to resolve anything below ~8%.

## Verdict

**Stage 4 does not pass.** 0.0% exact match on both suites, and the headline metric is exactly
where v1 started after burning 164 minutes and 4 GPUs.

## Root cause, restated from what the data now shows

Not data volume. Not termination. **Verbatim argument copying under a token budget.**

- Teacher-forced CALL-token accuracy was **98.1%** — the model knows *which* tool.
- Greedy decoding still produces plausible-but-wrong arguments and too many calls.
- The mixture (818,702 rows) is dominated by **MMLU x3 + GSM8K x4** with only ~x2/x3 on tool data;
  every extra MMLU/GSM8K epoch is MMLU/GSM8K gradient competing with verbatim copying.

## Next options (need your call — GPUs 1,3,4,5,7 are free)

1. **Re-probe at higher n and a larger budget** (~30 min). n=54 cannot resolve 0-8%. A 512->1024
   budget test separates "over-call" from "copy failure" cleanly. **Cheapest way to avoid another
   2-hour blind retrain.**
2. **Reweight the mixture** so tool data dominates (drop MMLU/GSM8K epochs), then retrain (~2 h).
3. **Start Stage 5 (RL) anyway** and let RL fix tool calling. Defensible — RL is exactly the tool
   that optimises *rewarded* exact calls. But it inherits a 0.0% starting point.

I recommend **(1) then (2)**, and I'd hold Stage 5 until exact match is off zero, because
otherwise RL starts from a checkpoint that cannot produce a single correct call.
