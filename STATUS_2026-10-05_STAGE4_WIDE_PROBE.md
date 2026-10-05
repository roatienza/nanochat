# STATUS — Stage 4 v2 WIDE acceptance probe (plan option 1), 2026-10-05

Checkpoint: `chatsft_checkpoints/d28b-v2/model_015548.pt`, val_bpb **0.31480**
Probe code: `probe_wide.py` (new), GPU 1/3/4/5/7, ~11 min wall clock.

## Why this probe existed

Every prior probe ran `--n 54` on `ToolCalling(split="val")`. That slice is
**54 examples = the entire hermes val population** (`ToolCalling.__init__`:
val_fraction=0.05 of 1090 convertible rows → train 1036 / val 54). So "3.7%"
at step 5,000 was **2 examples out of 54**, and nothing below ~8% was resolvable.
That is why I kept having to retract numbers. This probe runs the full
population of both val sets and adds a budget sweep.

## Result: n=236 across two corpora, two budgets — exact match is NOT 0.0%

| | hermes val | ToolACE val |
|---|---|---|
| n | **54** (whole population) | **182** (whole population) |
| exact match @512 | **0/54 = 0.0%** | **1/182 = 0.5%** |
| exact match @1024 | 0/54 = 0.0% | 1/182 = 0.5% |
| count_ok | 4/54 = 7.4% | 51/182 = 28.0% |
| STOPPED before cap | 51/54 = 94.4% | 178/182 = 97.8% |
| fn acc | **94.6%** | 58.2% |
| call-prefix acc | 48.1% | 10.7% |
| mean pred vs ref | 5.81 vs 2.20 | 2.46 vs 1.97 |

**Doubling the budget from 512 to 1024 changed nothing** — every metric moved
<0.2pt, and exact match did not move at all. Only 5.6% / 2.2% of generations
hit the cap. This falsifies the standing "calls are being truncated by the token
cap" explanation outright. That hypothesis is dead.

## The two real defects, separated

### Defect 1 — argument infidelity (hermes): the capability is nearly there
- fn acc **94.6%** — it picks the right tool almost every time.
- At the first point of divergence, **30/31 cases were the right function name
  with wrong arguments**. Only 1/31 was a wrong name.
- **23/54 examples matched the ENTIRE reference as a correct prefix, then kept
  emitting more calls** (mean +4.35 extra).

So on hermes the model knows what to call and then will not stop. This is a
**target-construction** defect, not a capability defect — consistent with
`truncate_to_terminal_call` (commit `e404ea0`) fixing termination but leaving
over-calling untouched.

### Defect 2 — tool selection + under-calling (ToolACE): a different failure
ToolACE is 3.4× larger and counts are nearly right (2.46 vs 1.97), yet exact
match is still 0.5% and fn acc collapses to 58.2% vs hermes' 94.6%. At the first
divergence, **69/174 were the wrong function name**. Mean extra calls by
reference size:

| ref calls | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| mean extra | +1.42 | +0.50 | **−0.45** | **−2.56** |

The model **under-calls as the reference gets longer**. Extrapolating the trend,
it is badly off the deep tail. hermes val has almost no 4-call examples (3) so
this was invisible at n=54.

## Answer to the question the probe was meant to settle

Exact match is not exactly zero, but it is ~0.5% — two independent failures, and
**neither is "needs more data" in the volume sense**. Retraining the same mixture
longer would not move either, which is consistent with the two previous Stage 4
retrains failing the same way. The probe earned its keep: it stopped a third
blind retrain and points at a specific, different fix for (2).

## Repo state

- nanochat: probe_wide.py added, pushed to fork. `karpathy/nanochat` untouched.
