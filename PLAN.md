# reasoner-model

A plan to train one model end-to-end — language modeling → instruction following →
reasoning → tool calling → long-horizon agentic tasks — on a single
**8 × NVIDIA A100 40 GB** node.

The constraint that shapes everything below: **A100 is Ampere, which has no FP8
support at all.** That is not a tuning preference, it is a hardware fact, and it
caps the whole ladder.

---

## 1. The decision that has to be made first

Two questions decide the entire curriculum, and both have a forced answer on
this hardware.

**Q: Can we reach GPT-2 grade capability on 8×A100-40GB?**

nanochat's `runs/speedrun.sh` targets GPT-2 grade on 8×H100 in ~1.5 hours, and
the leaderboard is scored on H100 wall-clock. It *does* run on Ampere — the
README says so explicitly — but there is no published A100 result for the
speedrun. We will not reproduce the leaderboard on this node. Reproducing the
*pipeline*, the *metric*, and the *method* is achievable and is the actual
teaching goal.

**Q: Which model size fits?**

Measured against nanochat's own architecture (aspect-ratio 64, head_dim 128,
vocab 32768), with optimizer state counted as Muon-on-matrices (10 B/param) plus
Adam-on-embeddings (12 B/param):

| Config | Params | Largest fitting batch | Peak mem | Verdict |
|---|---:|---:|---:|---|
| d12 (aspect 64) | 739 M | 32 (default) | 10.3 GiB | ✅ comfortable |
| d16 | 1,342 M | 32 | 18.4 GiB | ✅ comfortable |
| d18 | 1,721 M | 32 | 23.4 GiB | ✅ comfortable |
| **d20** | **2,155 M** | **32 (default)** | **29.2 GiB** | ✅ **headline — real headroom** |
| d22 | 2,646 M | 32 | 35.7 GiB | ⚠️ borderline, only ~2 GiB spare |
| d24 | 3,196 M | — | does not fit at bs=1 | ❌ **infeasible** |
| d26 (GPT-2 grade) | 3,808 M | — | does not fit | ❌ infeasible |

**d20 (≈2.15 B params, bf16) is the headline model.** It runs at nanochat's
*default* device batch size of 32 with ~10 GiB to spare, which is the margin you
want on shared teaching hardware. d22 technically fits but leaves no room for a
longer sequence or a bigger eval batch — do not plan around it. d24+ is
arithmetically impossible on 40 GB per GPU: optimizer state alone exceeds the card.

Reproduce this table with `python3 scripts/budget.py`.

The escape hatch if you want more depth later is a **narrower aspect ratio**, but
be clear about what it costs: d24 at aspect-32 is only **1.43 B** parameters
(18.7 GiB at bs=1, comfortable at bs=32) — *fewer* than d20 at aspect-64, not
more. Narrowing buys you depth at the price of width, and nanochat's single
`--depth` dial is tuned for the aspect-64 sweet spot. It is a different model,
not a bigger one. Do not start there.

---

## 2. Why one model, end to end

The user's requirement is a single model that walks the whole ladder. That is the
right goal, and it is achievable — but only if the transitions are treated as
continuations of a single run, not as separate projects.

```
tok_train → base_train (pretrain) → chat_sft → chat_rl (GSM8K, REINFORCE-style)
                                       ↓
                          + tool-calling SFT task (add it)
                                       ↓
                    agentic RL against constructed environments
```

Every stage loads the previous stage's checkpoint. Nothing is thrown away.

**One caveat worth stating up front:** nanochat's stock SFT mixture does **not**
teach tool use. `tasks/smoltalk.py` loads `HuggingFaceTB/smol-smoltalk`, and that
dataset's own card states it *"include[s] less task specific data compared to
SmolTalk (e.g no function calling ...)"* — function calling was deliberately
excluded for small models. So Stage 4 is the one place we write a real data
task, not just a command line.

---

## 3. The ladder

### Stage 0 — Tokenizer (hour 0)

```bash
python -m nanochat.dataset -n 8
python -m scripts.tok_train
python -m scripts.tok_eval        # compression ratio is the metric
```

vocab 2^15 = 32,768. Model and tokenizer are designed together; changing one
without the other wastes the run.

### Stage 1 — Pretraining, ablation scale (d12)

The workhorse. Every experiment runs here first.

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_train -- \
    --depth=12 --run=ablation-d12 \
    --device-batch-size=32 \
    --core-metric-every=999999 --sample-every=-1 --save-every=-1
```

**Monitor `val_bpb`, not raw loss.** Raw loss is vocab-size-dependent; bits per
byte is not. Data ablations routinely invert between the two — a corpus can give
lower loss while producing a worse model.

**Time:** ~15 node-hours at data:param ratio 12. **Cost:** $110–200 depending on
your rate. This is the stage a whole cohort can afford to run.

### Stage 2 — Pretraining, headline model (d20)

Same code, two flags different.

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_train -- \
    --depth=20 --target-param-data-ratio=8 \
    --device-batch-size=32 \
    --model-tag=d20 --run=headline-d20
```

ratio 8 rather than 12 — slightly undertrained beats compute-optimal on this
task, because you cannot afford to train past the point of diminishing returns.

**Time:** ~83 node-hours (ratio 8) or ~124 (ratio 12). **Cost:** $730–1,700
depending on your rate. Plan on four days of the node.

### Stage 3 — Instruction following + reasoning (SFT)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=sft-v1
```

Mixture is `SmolTalk(smol) + MMLU(auxiliary_train) + GSM8K(train)` as a
`TaskMixture`. This is where the model learns the chat format, multiple choice,
and step-by-step arithmetic.

**Reasoning does not only come from SFT.** SmolLM3 put ~140 B reasoning tokens in
*mid-training* and a further 1.8 B synthetic from a larger teacher, then 2 T
more. Reasoning is built in at least two places. Tell students this explicitly —
it is the single most common misconception in the course.

### Stage 4 — Tool calling (the one stage you must write)

**This is the actual work.** The stock mixture has no function-calling data, so:

1. Start from **xLAM-60k**: 3,673 APIs across 21 domains, CC-BY-4.0, every sample
   executed and semantically verified. It is the best public function-calling
   corpus.
2. Render it into nanochat's chat format, dropping the JSON schema out of the
   *target* tokens so the loss lands only on the emitted call.
3. Add it to the `TaskMixture` in `chat_sft.py` — one class, following the
   pattern of `tasks/gsm8k.py`, so the mixture upweights it.

The recipe detail that matters: **apply loss only on the tool-call arguments, not
the whole sequence.** Masking correctly is most of the skill.

### Stage 5 — Reasoning RL

nanochat ships `scripts/chat_rl.py` out of the box, running on GSM8K.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=rl-gsm8k
```

The author's own header is worth reading aloud to students, because it is
candid about what this is: GRPO *in quotes*, because after deleting the trust
region, the PPO ratio and clipping, it is closer to REINFORCE with DAPO-style
token-level normalization. Teach it as a stripped-down on-policy method — teach
the simplifications, not the acronym.

Report **pass@k** to see variance across k samples, so students see that a single
sample is not a measurement.

### Stage 6 — Long-horizon agentic

The genuinely hard stage, and the one where this hardware bites hardest.

nanochat's `nanochat/execution.py` already provides a subprocess sandbox (rlimits,
scrubbed env, destructive calls disabled, temp dir cleanup). Its docstring is
honest that it is **not a security boundary** — no seccomp, no containers,
network not blocked. That is fine for evaluation, and must be stated to anyone
who might deploy it.

**Environment count dominates, not model scale.** EnvScaler (ACL 2026 Findings)
measured the steepest gains going from 0 → 20 environments, with diminishing
returns after. So: build many cheap verifiable environments, not one deep one.

**Build a difficulty curriculum.** Order tasks easy → hard and let the model
graduate. This is the highest-leverage thing in the stage.

**Evaluate transfer, not just in-domain score.** Post-training on non-coding
office tasks has been reported to gain ~5.8 points on SWE-Bench Pro. Anything a
student reproduces in that direction is publishable.

---

## 4. Where the hardware actually hurts

Node-hours below are **derived** (FLOPs ÷ measured-MFU), for pretraining only.
SFT and RL figures are **order-of-magnitude estimates** — they depend on dataset
size and rollout count in ways that cannot be derived from architecture alone.
Treat them as planning bands, not commitments.

| Stage | Cost on this node | Basis | Verdict |
|---|---|---|---|
| Tokenizer | minutes | — | trivial |
| Pretrain d12 | ~15 node-h | derived | ✅ cheap, iterate freely |
| Pretrain d20 | ~83 node-h | derived | ✅ the headline run |
| SFT | ~10 node-h | estimate | ✅ |
| Tool-calling SFT | ~10 node-h | estimate | ✅ |
| Reasoning RL (GSM8K) | ~20 node-h | estimate | ✅ |
| **Agentic RL** | **200–1,000+ node-h** | estimate | ⚠️ **dominates everything** |
| d24+ pretraining | — | infeasible | ❌ |

Agentic RL is 2–12× the cost of pretraining the model in the first place. That
is the real shape of this project, not the pretraining.

---

## 5. The three traps to design the course around

1. **Training loss is not capability.** Build this into assignment 1, not
   assignment 5.
2. **Don't switch frameworks between ablation and final run.** HF call this
   "the road to suffering." Pick nanochat for stages 0–5; move to TRL/verl only at
   Stage 6 if you outgrow it.
3. **Capability is trainable at multiple stages.** Students who only SFT a base
   model will never see reasoning emerge, and will draw the wrong conclusion.

---

## 6. Hardware notes

- **No FP8.** `--fp8` is Hopper-only and will not work here. nanochat
  auto-detects `bfloat16` on SM 80+, which is correct for A100.
- **40 GB, not 80 GB.** This is the binding constraint on model size, and it is
  what rules out d24+. d20 fits at the default batch size with headroom; you
  probably will not need to touch `--device-batch-size` at all.
- **Single node only.** No tensor parallelism across nodes; DDP within 8 GPUs.
- Verify before first run: `nvidia-smi --query-gpu=name,memory.total --format=csv`

---

## 7. Sources

- nanochat — https://github.com/karpathy/nanochat
  (README confirms Ampere support and the <80 GB `--device-batch-size` guidance)
- nanochat speedrun leaderboard — `dev/LEADERBOARD.md`
- smol-smoltalk dataset card — https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
  (states function calling is excluded)
- SmolLM3 — https://huggingface.co/blog/smollm3
  (calibration point: ~14,000 tok/s/GPU on H100 at 29.4% MFU)
- Smol Training Playbook — https://huggingfacetb-smol-training-playbook.hf.space/
- nanotron — https://github.com/huggingface/nanotron
- OpenThoughts2 — https://github.com/open-thoughts/open-thoughts
- verl — https://github.com/verl-project/verl
- tau2-bench — https://github.com/sierra-research/tau2-bench
- EnvScaler — https://aclanthology.org/2026.findings-acl.407.pdf
- NeMo function-calling recipe — https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/function-calling
- A100 vs H100 precision support — https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth
