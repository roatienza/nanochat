# utak — the complete build-and-train recipe

**Model:** `utak` · **Checkpoint:** `utak-sft-c` (step 240) · **Params:** 2,018,511,234
**Context:** 131,072 · **Served as:** `utak` on an OpenAI-compatible `/v1` endpoint
**Final val bpb:** 0.393419

This is the step-by-step record of how `utak` is built, trained, served and
verified, and the automated scripts that reproduce it. Every number here was
read from a checkpoint's `meta_*.json` (`user_config`), a training log, or a
live measurement — not recalled.

**Everything in this document is executable.** The `recipe/` directory holds
one script per stage; each one carries the flags, the reasoning, and the
measured results from the run it reproduces.

---

## 0. Quick start

```bash
cd nanochat

# the whole thing, from an empty base directory
./recipe/run_all.sh

# or a single stage
./recipe/run_all.sh --only 9

# what is in the pipeline
./recipe/run_all.sh --list
```

Configuration lives in exactly one place — `recipe/config.sh`. Every path,
GPU, port, model name and checkpoint tag is defined there and inherited by
every stage. Change it there and the whole pipeline follows.

```bash
MODEL_NAME=utak                       # the served model id
MODEL_DIR_NAME=utak-128k              # the vLLM model directory
NANOCHAT_BASE_DIR=.../nanochat_store/v2
GPU_PRETRAIN=4,5,6,7  GPU_SERVE=0
PUBLIC_PORT=8000       UPSTREAM_PORT=8010
```

---

## 1. The model

| Component | Value |
|---|---|
| Layers | 28 |
| Attention heads / KV heads | 14 / 14 |
| Model dimension | 1792 |
| Head dimension | 128 |
| Vocabulary | 32,768 |
| Feed-forward | `relu²`, hidden multiplier 4 |
| Window pattern | `SSSL` (tiled across layers; the final layer is always `L`) |
| Short window | 512 |
| RoPE base | 100,000 |
| QK scale | 1.2 |
| Logit softcap | 15.0 |
| Weight tying | Untied |
| Total parameters | 2,018,511,234 (2.02 B) |

Defined in `nanochat/gpt.py` (`GPT`, `GPTConfig`). The parameter count is
measured, not estimated — `param_audit.py` instantiates the real `GPT` class
on the meta device and counts.

---

## 2. The lineage

The single most important thing to get right. The stage numbering in the
original logs is *not* the execution order, and two of the orderings are
load-bearing.

```
 0  data           tokenizer + ClimbMix corpus
 1  pretrain       base model, 2k ctx ........................ utak-base
 2  sft_tools      tool-calling SFT, 2k ctx ................. utak-sft-tools
 3  warmup         GSM8K warmup SFT, fresh optimizer ........ utak-warmup
 4  rl_math        math RL on GSM8K ......................... utak-rl-warmup
 5  rl_agentic     agentic RL, sandbox-verified ............. utak-rl-pilot
 6  rl_curriculum  weighted-block + exemplar curriculum .... utak-rl
 7  longctx        context extension 2k -> 128k ............. utak-128k-final
 8  sft_long       long-context SFT ........................ utak-sft
 9  touchup        the 6-rung touch-up ladder ............... utak-sft-c   <== SERVED
10  pack           checkpoint -> vLLM model dir
11  serve          guard on :8000 -> vLLM on :8010
12  gate           the v5x and HP gates
13  e2e            the end-to-end acceptance test
14  regression     the offline unit tests
15  scrub_test     the live-server hypothesis test (the guard's history scrub)
```

### The two orderings that matter

**The RL chain runs BEFORE the context extension.** Stages 4–6 produce a
2,048-context model; stage 7 rebuilds it at 131,072 context *from those RL
weights*. Running stage 7 first would train a different model. This is
visible in the checkpoint metadata: `base_checkpoints/d28b-128k/meta_001750.json`
records `source_phase: "rl"`, `model_tag: "d28b-rl100"`, `model_step: 177`.

**The touch-up ladder is a chain of six, and the last two are near-identical.**
The served checkpoint descends `v5e → v6a → v6j`, not `v5e → v6j`. The only
difference between the v6a and v6j configurations is
`harness_poison_epochs` (absent vs `3`) — everything else is byte-identical.
That is deliberate: it isolates the change, so the measured difference
belongs to the change.

### The full served chain, checkpoint to checkpoint

| Recipe tag | Original tag | Built from | Step | Context | Iters |
|---|---|---|---|---|---|
| `utak-base` | `d28b-v2` | — (pretrain) | 277,760 | 2,048 | — |
| `utak-sft-tools` | `d28b-v2` (sft) | `utak-base` @ 277,760 | 13,630 | 2,048 | 1 epoch |
| `utak-warmup` | `d28b-warmup` | `utak-sft-tools` @ 13,630 | 509 | 2,048 | 509 |
| `utak-rl-warmup` | `chatrl/d28b-warmup` | `utak-warmup` @ 509 | 100 | 2,048 | 467 |
| `utak-rl-pilot` | `chatrl/d28b-rl100` (pilot) | `utak-rl-warmup` @ 100 | 177 | 2,048 | 178 |
| `utak-rl` | `chatrl/d28b-rl100` | `utak-rl-pilot` @ 177 | 177 | 2,048 | 178 |
| `utak-128k` | `d28b-128k` | `utak-rl` @ 177 | 1,750 | 131,072 | 2,000 |
| `utak-128k-final` | `d28b-128k-final` | `utak-128k` @ 1,750 | 250 | 131,072 | 250 |
| `utak-sft` | `d28b-128k-sft-v5` | `utak-128k-final` @ 250 | 800 | 131,072 | 800 |
| `utak-sft-followup` | `…-v5b` | `utak-sft` @ 800 | 120 | 8,192 | 120 |
| `utak-sft-glaive` | `…-v5c` | `utak-sft-followup` @ 120 | 80 | 8,192 | 80 |
| `utak-sft-glaive-heavy` | `…-v5d` | `utak-sft-glaive` @ 80 | 100 | 8,192 | 100 |
| `utak-sft-a` | `…-v5e` | `utak-sft-glaive-heavy` @ 100 | 60 | 8,192 | 60 |
| `utak-sft-b` | `…-v6a` | `utak-sft-a` @ 60 | 240 | 32,768 | 240 |
| **`utak-sft-c`** | **`…-v6j`** | **`utak-sft-b` @ 240** | **240** | **32,768** | **240** |

---

## 3. Stage 0 — data

`recipe/00_data.sh`

### The tokenizer

A 32,768-token byte-level BPE, trained on the ClimbMix corpus with a
10,000-character document cap. It is a **pre-training artifact**: every
later stage — SFT, RL, and the serving stack — reads it from
`$NANOCHAT_BASE_DIR/tokenizer/`. It is trained once and kept.

```bash
python -m scripts.tok_train --max-chars 2000000000 --doc-cap 10000 --vocab-size 32768
```

### The pre-training corpus

`karpathy/climbmix-400b-shuffle` (NVIDIA Nemotron ClimbMix), 6,543 shards
upstream. The original run used **171 shards / ~15 GB** on disk.

```bash
python -m nanochat.dataset -n 170 -w 8
```

The downloader always pins the **last** shard (`shard_06542`) as validation
and uses everything else for training.

**The shard count is not a free parameter.** The pre-training schedule is
derived from `--target-param-data-ratio 8.0` against the data that is
actually present, so changing the shard count changes the iteration count
and therefore the compute budget.

### Packing

`nanochat/dataloader.py` → `tokenizing_distributed_data_loader_bos_bestfit`:
tokenize, then pack with a best-fit bin-packing so rows are dense and few
documents straddle a row boundary. `nanochat/dataset.py` handles
download-on-demand and iteration by parquet row group for throughput.

### The task corpora

Fetched on first use by the SFT and RL stages: GSM8K, MMLU, ARC, HumanEval
and SmolTalk, under `$NANOCHAT_BASE_DIR/task_data/`.

---

## 4. Stage 1 — pre-training

`recipe/01_pretrain.sh` · `scripts/base_train.py`

| Setting | Value |
|---|---|
| Steps | **277,760** (derived, not pinned) |
| Context | 2,048 |
| Device batch / total batch | 4 / 32,768 |
| Embedding / unembedding LR | 0.3 / 0.008 |
| Matrix / scalar LR | 0.02 / 0.5 |
| Weight decay | 0.28 |
| Target param:data ratio | 8.0 |
| Warmup / warmdown | 40 steps / 0.65 |
| Final LR fraction | 0.05 |
| Eval tokens | 41,943,040 |
| Save every | 5,000 |
| **Final val bpb** | **0.840180** |

```bash
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
  --run utak-pretrain \
  --depth 28 --aspect-ratio 64 --head-dim 128 \
  --max-seq-len 2048 --window-pattern SSSL \
  --device-batch-size 4 --total-batch-size 32768 \
  --embedding-lr 0.3 --unembedding-lr 0.008 \
  --matrix-lr 0.02 --scalar-lr 0.5 \
  --weight-decay 0.28 \
  --warmup-steps 40 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --num-iterations -1 --target-param-data-ratio 8.0 \
  --eval-every 500 --eval-tokens 41943040 --save-every 5000 \
  --model-tag utak-base
```

`--num-iterations -1` with `--target-param-data-ratio 8.0` is what sets the
277,760-step schedule. Pinning a step count instead would silently change
the compute budget.

> **A correction worth recording.** The technical report and model card
> previously stated "40,000 steps" for this stage. The checkpoint metadata
> says `meta_277760.json`, run `headline-d28-v2` — the run reached step
> **277,760**, about 7× the stated figure. The recipe uses the measured
> value.

---

## 5. Stage 2 — tool-calling SFT (2k context)

`recipe/02_sft_tools.sh` · `scripts/chat_sft.py`

The first SFT on top of the base model. The mixture, read from the
checkpoint metadata of the original run:

| Task | Epochs |
|---|---|
| ToolACE | 6 |
| Tool calling | 4 |
| GSM8K | 1 |
| MMLU | 1 |
| SmolTalk | 1 |

`--device-batch-size 2`, `--total-batch-size 32768`, `--init-lr-frac 0.8`,
`--load-optimizer 1`, `--num-iterations -1` (one epoch over the mixture).

Output: `chatsft_checkpoints/utak-sft-tools` @ 13,630.

---

## 6. Stage 3 — the GSM8K warmup

`recipe/03_warmup.sh` · `scripts/chat_sft.py`

A short, narrow mixture on a **fresh optimizer** (`--load-optimizer 0`):
GSM8K × 10 + Tool calling × 4, no SmolTalk, no MMLU, no ToolACE.

**Why it exists.** The first math-RL attempt, launched straight from the
tool-calling SFT checkpoint, died at step 7 with reward 0.0 and Pass@1
0.5%. The model did the arithmetic in-head and never emitted the `<calc>`
format, so REINFORCE had no gradient signal at all. This warmup gives it
one.

**What it cost.** Measured on the warmup gate, against the previous
checkpoint:

| Task | Warmup | Previous | Δ |
|---|---|---|---|
| ARC-Easy | 36.78% | 45.45% | −8.7 |
| ARC-Challenge | 31.91% | 34.73% | −2.8 |
| MMLU | 31.46% | 33.33% | −1.9 |
| **GSM8K** | **25.00%** | **0.00%** | **+25.0** |
| HumanEval | 0.00% | 16.67% | −16.7 |
| ChatCORE | 0.1171 | 0.1360 | −0.019 |

The narrow mixture trades breadth for the ability to be rewarded. The
ChatCORE change is inside the noise band (an adjacent checkpoint swung
0.0011 → 0.0485).

Output: `chatsft_checkpoints/utak-warmup` @ 509.

---

## 7. Stages 4–6 — the reinforcement-learning chain

`recipe/04_rl_math.sh`, `05_rl_agentic.sh`, `06_rl_curriculum.sh`

All three share the same optimization core.

### The algorithm

`scripts/chat_rl.py` is "GRPO in quotes", and the header is accurate: it is
REINFORCE with four deliberate simplifications.

1. **No trust region** — no KL regularization against a reference model.
2. **On-policy**, so there is no PPO probability ratio and no clipping.
3. **DAPO-style token-level normalization**, not sequence-level.
4. **Advantage is `(r − μ)`** — no z-score division by σ.

`scripts/chat_rl_agentic.py` and `scripts/chat_rl_curriculum.py` keep the
optimization core **byte-identical**; only the reward source changes.

### Stage 4 — math RL on GSM8K

| Setting | Value |
|---|---|
| Device batch / examples per step / samples | 2 / 16 / 16 |
| Max new tokens | 512 |
| Temperature / top-k | 1.0 / 50 |
| Embedding / unembedding / matrix LR | 0.2 / 0.004 / 0.02 |
| Init LR fraction | 0.05 |
| Eval every / examples | 20 / 400 |
| Save every | 100 |

**Measured.** Test-split Pass@1: 0.1350 (step 0) → **0.1775 (step 40, the
peak)** → 0.1575 (step 120, where it was stopped). The run is stopped at
the flat point on purpose: train reward saturates near 0.95 (exactly 1.0 at
steps 95–96), so saturated groups carry near-zero advantage and the
gradient dies.

**The caveat, measured and recorded.** The rollouts are on the GSM8K
**train** split — the same problems the warmup SFT'd for 10 epochs — so the
train reward is inflated by memorization. The honest metric is eval Pass@1
on the **test** split.

### Stage 5 — agentic RL on sandbox-verified environments

`tasks/agentic.py` generates fresh instances per index, so the rollout set
can never be memorized. That is the direct fix for the stage-4 failure
mode. The reward is 0/1, verified by `nanochat/execution.py`: the reference
solutions were validated at 15/15, and 900 wrong-solution trials leaked 0.

The eval holdout is the tail 5% of each (family, level) block, never
trained on.

**Measured.** Eval Pass@1 0.0667 (step 0) → 0.2667 (step 20). Fresh
verifiable reward moves the needle about **5× faster** than the memorized
GSM8K mixture did.

### Stage 6 — weighted-block curriculum + exemplar scaffold

Two changes, each with the evidence that motivated it.

**1. Weighted block pool instead of a sequential walk.** The pilot walked
blocks in order, so **42/178 steps (24%)** landed entirely inside uncracked
blocks where all 16 samples of every example scored 0 → advantage 0 → zero
gradient. Meanwhile `data_fix@L1`, trained once at steps 36–47 and never
revisited, regressed from 1.00 to 0.00 on eval. The fix: sample example
indices from a weighted pool every step, so replayed blocks keep their
capability and frontier blocks get maximal exposure.

**2. Train-only exemplar scaffold** for the 5 frontier blocks. The pilot
never scored a single success on them (0/256 samples per step), and with a
0/1 reward and advantage = `r − μ`, an all-zero sample group contributes
nothing no matter how often it is drawn. A worked example (instance 0 of the
block, with its verified solution) is prepended to the user prompt at
**train time only**, giving the policy a path to its first successes. It is
annealed off at `--exemplar-anneal-step 89`, and the eval never sees
exemplars, so the eval curve measures whether the capability was
*internalized* rather than parroted.

**The probe that chose "exemplar" over "hint":** prose hints lifted
*nothing* on the frontier (0/16 with and without, all 5 blocks), while
worked examples moved `count_chars@L3` 0.000 → 1.000 and `list_ops@L2`
0.000 → 1.000.

**Measured.** Pass@1 0.600 (pilot) → **0.667** with 10/15 blocks mastered.
`data_fix@L1` recovered from 0.00 to 1.00.

Output: `chatrl_checkpoints/utak-rl` @ 177 — **this is the checkpoint the
long-context stage builds on.**

### The tag plumbing

`chat_rl*.py` have no `--output-tag`. They read
`chatsft_checkpoints/<model-tag>` and write `chatrl_checkpoints/<model-tag>`.
The original lineage therefore hard-linked each RL output forward into the
`chatsft` tree under the next tag. The recipe does the same with
`cp -al` (hard links: no disk, no copy time).

---

## 8. Stage 7 — context extension (2k → 128k)

`recipe/07_longctx.sh` · `scripts/base_train_long.py`

The model is rebuilt at `sequence_len=131072` with the `SSSL` sliding-window
pattern and `rope_base=100000` — **without YaRN** — and trained in two
passes.

| Setting | 7a | 7b (anneal) |
|---|---|---|
| Output | `utak-128k` | `utak-128k-final` |
| Iterations | 2,000 | 250 |
| Warmdown ratio | 0.5 | 0.9 |
| Embedding / matrix LR | 0.09 / 0.006 | 0.09 / 0.006 |
| Weight decay | 0.028 | 0.028 |
| Total batch | 262,144 | 262,144 |
| Eval tokens | 262,144 | 262,144 |
| **Final val bpb** | | **0.893391** |

The val bpb rises from 0.840 to 0.893 across this boundary because the
validation *mixture* changed — it is now evaluated at 128k on long packed
rows, a strictly harder distribution. **The two numbers are not
comparable.** The same caveat recurs at every SFT stage.

---

## 9. Stage 8 — the long-context SFT

`recipe/08_sft_long.sh` · `scripts/chat_sft_long.py`

`scripts/chat_sft.py` is the upstream 2k-context SFT and cannot reach 128k.
`scripts/chat_sft_long.py` is the fork's long-context SFT: it renders with
the full tokenizer conversation format and packs at the full `max_seq_len`.

| Setting | Value |
|---|---|
| Iterations | 800 (100, then a resume from step 100) |
| Long-context epochs | **20** |
| Levels | 0, 1, 2, 3 |
| Instances per level | 2000, 1000, 500, 250 |
| Tool calling / ToolACE epochs | 2 / 2 |
| SmolTalk epochs | 1 |
| Total batch | 131,072 |
| Embedding / matrix LR | 0.072 / 0.0048 |
| Eval every / tokens | 50 / 262,144 |

The long-context data is weighted heavily (20 epochs, ~3,750 instances per
level) because the model is being taught to actually *use* a 128k window,
not merely tolerate it.

Output: `chatsft_checkpoints/utak-sft` @ 800.

---

## 10. Stage 9 — the touch-up ladder

`recipe/09_touchup.sh` · `scripts/chat_sft_long.py`

Six short, cheap SFT runs, each fixing one specific defect found by the
gates.

| Rung | Output | From | Iters | Context | What it fixes |
|---|---|---|---|---|---|
| 9a | `utak-sft-followup` | `utak-sft` @ 800 | 120 | 8,192 | continue after a tool result |
| 9b | `utak-sft-glaive` | 9a @ 120 | 80 | 8,192 | multi-turn tool-use fluency |
| 9c | `utak-sft-glaive-heavy` | 9b @ 80 | 100 | 8,192 | more multi-round weight |
| 9d | `utak-sft-a` | 9c @ 100 | 60 | 8,192 | re-anchor long-context ability; move to the v6 shape |
| 9e | `utak-sft-b` | 9d @ 60 | 240 | 32,768 | **harness basics** — the key rung |
| 9f | **`utak-sft-c`** | 9e @ 240 | 240 | 32,768 | **harness poison + the literal-`\n` data fix — SERVED** |

### The v6 shape (rung 9d)

With the tools block on the **last** user turn and 60+ turns of history
after it, the 2B model loses the block→action association and hallucinates
tool names (`get_weather`, `web_news`, `get_file_path`). With the block on
the **first** user turn, the same history+tools+question renders correctly.
The served chat template (`chat_template_v7.jinja`) places it first.

### Harness basics (rung 9e) — the key rung

Teaches the things general-purpose tool-calling corpora do not:

| Shape | Teaches |
|---|---|
| B1 social | `hello` / `how are you` / `thanks` → short greeting, **no call** |
| B2 easy-knowledge | `What is 2+2?` → direct answer, **no call** |
| B3 toolish → CALL | route to the *correct* tool |
| B4 date-from-system | `date today` → `Today is {today}.` from the system prompt, never invented |
| B5 toolish-knowledge | `What time zone is Cairo in?` → prose from knowledge |

**Why it is needed.** A census of the earlier mixture found the imbalance
that caused the original failure:

```
fresh + tools -> CALL  : ToolCalling 1,036 + ToolACE 8,957 = 9,993 rows
fresh + tools -> prose : 281 rows (144 with the harness's 8-tool set)
                          => 35:1 toward CALL
```

The model had learned "tools present → call `web_search`" as its default:
fresh `hello` → spurious `web_search` 10/10; "What is 2+2?" →
`web_search`; "what files are in the working directory?" →
`web_search {"query":"finance"}` — wrong routing too, because no row taught
which tool serves which request.

The B3 routing map:

| Request | Tool |
|---|---|
| weather / news / search a phrase | `web_search {query, type?}` |
| list files / run a command | `bash {command}` |
| read a file | `read_file {path}` |
| find where X is logged | `grep {pattern}` |
| fetch a URL | `fetch_content {url}` |
| write a file | `write_file {path, content}` |
| edit a file | `edit_file {path, old, new}` |
| search my files for X | `local_search {query}` |

Two robustness dimensions are randomized deliberately: **toolsets** (the
harness's real 8 tools 60%, random subsets 25%, the classic 3 15%) and
**system prompts** (50% the exact `baby-onit` system prompt including its
Credentials section, 50% harness-like).

### Harness poison + the literal-`\n` fix (rung 9f) — the served rung

**The bug.** `tasks/harness_basics.py` used `"\\n\\n"` — an escaped
*backslash* followed by `n` — where a real newline was intended. The
supervised **target** therefore contained the two characters `\` and `n`.
The model faithfully learned to emit a literal backslash-n.

At inference this becomes a **period-2 byte-BPE cycle**: the tokenizer
alternates two tokens, producing `\n\n\n…` until the token cap.

**The failure chain.**

```
training target has literal "\n"
  -> model emits literal backslash-n
  -> period-2 byte-BPE cycle -> 4,096 tokens of "\n\n\n..."
  -> 120-124 s per answer
  -> baby-onit SAVES it into session_history.jsonl
  -> the harness replays that history verbatim on every later turn
  -> the poison is now an in-context attractor, re-teaching the behaviour forever
```

The history is what made it persistent. A 65 k-char degenerate answer is
~16k tokens; the harness's 8k-token budget can only drop **whole messages**
from the front, so it cannot shed the poison — the oversized message rides
along in full.

**The fix.** `tasks/harness_basics.py` now uses a real newline, and
`tasks/test_no_literal_newline.py` fails if any prose target ever contains
a literal backslash-n again.

**Why both the retrain and the guard ship.** The retrain removes the
*source*; the guard's history scrub stops an existing poisoned history from
*re-infecting* the model. Measured on identical prompts:

| Condition | Result |
|---|---|
| Raw model + poisoned history | **15.7 s of garbage** (504 literal `\n` in 512 tokens) |
| Raw model + **scrubbed** history | **0.6 s, clean** |
| Model **via the guard** + poisoned history | **0.6 s, clean** |

---

## 11. Stage 10 — pack

`recipe/10_pack.sh`

Converts the nanochat checkpoint into a HuggingFace-style directory vLLM
can load with `--trust-remote-code`:

1. export the weights + config — `serving/export_hf_128k.py`
2. patch `sequence_len` to 131,072 — `serving/patch_config_131072.py`
3. copy the aux files vLLM needs — `serving/model_assets/`
4. install the served chat template — `chat_template_v7.jinja`

The aux files live in the repository, so the pack is reproducible from the
tree alone — no network, no external asset directory.

Output: `$VLLM_DIR/utak-128k/`.

---

## 12. Stage 11 — serve

`recipe/11_serve.sh`

```
client (baby-onit, OpenAI-compatible)
   |
   |  POST /v1/chat/completions
   v
:8000  guard_proxy.py          <- the PUBLIC endpoint
   |    clamp max_tokens, scrub history, always stream,
   |    abort degeneration, 45 s wall-clock cap
   v
:8010  vLLM                   <- the model, GPU 0, bf16
```

**Why the inversion.** vLLM has no server-side completion cap that a client
cannot override, so the cap has to live in front of it. The public port
must therefore belong to the proxy.

### The vLLM flags, and why each one is there

| Flag | Value | Reason |
|---|---|---|
| `--port` | 8010 | Internal; the guard owns 8000 |
| `--max-model-len` | 131072 | Full context |
| `--max-num-batched-tokens` | 131072 | One full-length request at a time |
| `--max-num-seqs` | **8** | Not 1 — with 1, one degenerate request blocks every other |
| `--enable-prefix-caching` | on | The harness replays long history every turn; without it the history is re-prefilled from scratch |
| `--no-enable-chunked-prefill` | on | Measured necessary for correct long-context behaviour with the `SSSL` window pattern |
| `--gpu-memory-utilization` | 0.94 | — |
| `--enforce-eager` | on | No CUDA graph capture |
| `--served-model-name` | `utak` | The name clients ask for |
| `--enable-auto-tool-choice` | on | Native tool calling |
| `--tool-call-parser` | `nanochat` | Custom parser |
| `--tool-parser-plugin` | `nanochat_tool_parser.py` | Plugin registration |

### The guard's six responsibilities

1. **Clamp `max_tokens`** to 4,096 (`GUARD_MAX_TOKENS`), ignoring the client's value.
2. **Always stream upstream**, so the proxy has abort power for every client shape; non-streaming clients get the chunks reassembled into a full JSON response.
3. **Abort degenerate output mid-stream**, returning a clean OpenAI-compatible finish chunk. Degenerate `tool_calls` slots are dropped so the harness never executes garbage.
4. **Wall-clock cap** of 45 s (`GUARD_MAX_SECONDS`). Calibrated against the real model: the slowest legitimate turn observed was 8.8 s, so 45 s leaves 5× headroom while still bounding a runaway well before the user's "it hangs" threshold.
5. **Pass sampling parameters through untouched**, so the gates measure the model and not the proxy.
6. **Scrub degenerate assistant turns** from the replayed history.

### The degeneracy detector

| Signal | Threshold | Rationale |
|---|---|---|
| Identical chars in a row | 48+ | The `m`/whitespace flood (period-1) |
| Short-period cycle | period 2–8, ≥ 8 reps, ≥ 32 chars | The literal-`\n` loop (period-2) |
| Literal `\n` pairs, total | ≥ 4 | Clean separation: 0 good answers had 4+ |
| Literal `\n` pairs, consecutive | ≥ 8 | The degenerate run itself |

Every threshold was calibrated against **20 real model answers** (18 good,
2 degenerate), so the numbers are measured rather than guessed.

The original detector required 48+ *identical* characters. A period-2 cycle
cannot match that — which is exactly why the bug survived the first
defense.

### The chat template

`chat_template_v7.jinja` — the tools block on the **first** user turn
(measured, not assumed; see rung 9d).

### Generation defaults

```json
{ "max_new_tokens": 1024, "repetition_penalty": 1.05 }
```

These are *defaults only*. A client sending an explicit `max_tokens`
overrides `max_new_tokens` — which is exactly the hole the guard plugs.

---

## 13. Stages 12–14 — evaluation

`recipe/12_gate.sh`, `13_e2e.sh`, `14_regression.sh`

### The v5x gate — 15 checks, must be 15/15

| ID | Probe | Expectation |
|---|---|---|
| F1 | fresh `hello` ×5 | greeting, no call |
| F2 | fresh `2+2` ×5 | `4`, no call |
| F3 | fresh files → `bash` ×5 | `CALL bash` with `ls` |
| F4 | fresh weather ×5 | `CALL web_search` |
| F5 | fresh fetch ×5 | `CALL fetch_content` |
| R1 | replay `hello` ×3 | greeting, no call |
| R2 | replay weather ×3 | `CALL web_search` |
| R3 | replay news ×3 | `CALL web_search` type=news |
| R4 | replay `2+2` ×3 | `4`, no call |
| R5 | replay files ×3 | `CALL bash ls` |
| LOOP | 4-turn OpenAI loop | call → result → answer → … |
| NEW | fresh-session check | no spurious call |

Both gates run against the **live server through the guard**, using the real
`baby-onit` package — so they measure the model *and* the guard exactly as a
client sees them.

### The HP gate — 7 checks

The self-poisoning loop: the harness replays the model's own bad answers and
the model must not copy them.

| ID | Probe | Expectation |
|---|---|---|
| HP1 | poisoned hello ×3 | greeting, no canned-date copy |
| HP2 | poisoned weather ×3 | `CALL web_search` (route beats pattern) |
| HP3 | poisoned news ×3 | `CALL web_search` type=news |
| HP4 | poisoned files ×3 | `CALL bash ls` |
| HP5 | poisoned date ×3 | date from the system prompt, no canned copy |
| HP6 | poisoned `2+2` ×3 | `4`, no call, no canned copy |
| HP7 | poison-only history ×3 | weather question routes to `web_search` |

### Results for the served checkpoint

| Gate | Result |
|---|---|
| v5x | **15/15 PASS** |
| HP | **6/7** — HP5 fails |
| End-to-end | PASS |
| Detector / scrub / segment / regression suites | 4/4 PASS, 0 false positives |
| Library tests | 58/58 PASS |

**HP5 is a known, cosmetic, pre-existing failure.** The shipped checkpoint
behaves identically. The gate reports it rather than hiding it, and treats
`6/7` as deployable while anything else is a regression to investigate.

### The offline regression suite (stage 14)

`recipe/14_regression.sh` runs five suites. Note that four of them are
**standalone scripts that report through their exit code**, not pytest
modules — they are invoked directly, each with the interpreter it needs:

| Suite | Interpreter | Needs |
|---|---|---|
| `tasks/test_no_literal_newline.py` | the training venv | — |
| `eval/test_detector.py` | the vLLM venv | `httpx` |
| `eval/test_segment_detector.py` | the vLLM venv | `httpx` |
| `eval/test_guard_regression.py` | the vLLM venv | `httpx` |
| `tests/` (pytest) | the training venv | `pytest` |

The guard suites import `serving/guard_proxy.py`, which imports `httpx`.
`httpx` lives in the vLLM virtualenv, not the training one — that split is
why the recipe carries two interpreter variables.

Current result: **58/58 library tests, 4/4 guard suites, 0 false
positives.**

### The scrub hypothesis test (stage 15)

`recipe/15_scrub_test.sh` runs `eval/test_scrub.py`, which proves the
specific claim behind the guard's history scrub: that **scrubbing the
request history** — not merely aborting the output — is what makes the
model answer cleanly and fast. It is a hypothesis test, not a unit test: it
needs a running server, and it talks to the **raw vLLM endpoint** (:8010)
rather than the guard, so the guard's own scrub cannot be what makes the
difference.

---

## 14. The file map

| Path | Contents |
|---|---|
| `recipe/config.sh` | **the single source of truth** — every path, GPU, port, tag |
| `recipe/00_data.sh` … `15_scrub_test.sh` | one script per stage |
| `recipe/run_all.sh` | the whole pipeline, with `--from` / `--only` / `--list` |
| `nanochat/gpt.py` | model definition (`GPT`, `GPTConfig`) |
| `nanochat/tokenizer.py` | BPE tokenizer, conversation rendering, `no_train` |
| `nanochat/dataset.py` | ClimbMix download and iteration |
| `nanochat/dataloader.py` | BOS-bestfit packing loaders |
| `nanochat/engine.py` | batch generation engine (RL, evaluation) |
| `nanochat/checkpoint_manager.py` | save/load with optimizer state and metadata |
| `nanochat/optim.py` | Muon + AdamW optimizers |
| `nanochat/execution.py` | the subprocess sandbox that verifies agentic rewards |
| `nanochat/core_eval.py` | the CORE benchmark harness |
| `scripts/base_train.py` | stage 1 — pre-training |
| `scripts/chat_sft.py` | stages 2–3 — 2k-context SFT |
| `scripts/chat_rl.py` | stage 4 — math RL |
| `scripts/chat_rl_agentic.py` | stage 5 — agentic RL |
| `scripts/chat_rl_curriculum.py` | stage 6 — curriculum RL |
| `scripts/base_train_long.py` | stage 7 — context extension |
| `scripts/chat_sft_long.py` | stages 8–9 — long-context SFT and the touch-ups |
| `scripts/tok_train.py`, `tok_eval.py` | tokenizer training and evaluation |
| `scripts/base_eval.py`, `chat_eval.py` | evaluation harnesses |
| `tasks/*.py` | one dataset module per mixture component |
| `tasks/test_no_literal_newline.py` | the regression test for the `\n` bug |
| `chat_template_v7.jinja` | the served chat template |
| `serving/guard_proxy.py` | the guard: the public endpoint on :8000 |
| `serving/nanochat_tool_parser.py` | the vLLM tool-call parser |
| `serving/export_hf_128k.py` | checkpoint → HF directory |
| `serving/patch_config_131072.py` | patch the exported config to 128k |
| `serving/model_assets/` | the aux files vLLM needs |
| `eval/verify_v5x.py`, `eval/verify_hp.py` | the two gates |
| `eval/e2e_harness.py` | the end-to-end acceptance test |
| `eval/test_*.py` | detector, scrub, segment and guard regression tests |
| `runs/` | upstream's GPT-2 speedrun entry points (kept for provenance) |
| `param_audit.py` | real `GPT` class on the meta device → true parameter counts |
| `mem_probe.py` | real fwd+bwd+step → measured peak GPU memory |
| `budget_real.py` | the measured feasibility table |
| `docs/MODEL_CARD.md` | the model card |
| `docs/TECHNICAL_REPORT.md` | the engineering report |
| `docs/EXECUTION_LOG.md` | the chronological log, including every failure |
| `docs/checkpoint_meta/` | the `meta_*.json` of every checkpoint on the lineage |

---

## 15. What was removed, and why

The `nanochat/` subtree was pruned to exactly what is needed to rebuild and
serve the model. Everything removed is listed here, with the reason.

| Removed | Reason |
|---|---|
| `run_v6a.sh` … `run_v6j.sh` (10 files) | superseded by `recipe/09_touchup.sh`, which reproduces the same six rungs with the correct parent steps |
| `serving/pack_v6j.sh` | superseded by `recipe/10_pack.sh` |
| `serving/serve_v6j_8000.sh` | superseded by `recipe/11_serve.sh` |
| `chat_template.jinja`, `chat_template_v5.jinja`, `chat_template_v6.jinja` | only `chat_template_v7.jinja` is served; the others are superseded drafts |
| `scripts/stage6_offline_eval.py`, `stage6b_exemplar_probe.py`, `stage6b_hint_probe.py` | one-off diagnostic probes from the stage-6 investigation, not part of the build path |
| `dev/mem_probe_128k_v2.py`, `throughput_probe_128k.py`, `find_leak_128k.py` | one-off memory and leakage diagnostics; `mem_probe.py` (the general one) is kept |
| `docs/checkpoint_meta/d28b-128k-sft/`, `…-sft-final/` | dead branches — the served path goes `utak-128k-final → utak-sft`, not through `d28b-128k-sft` |

**`runs/` was kept, deliberately.** It is upstream's canonical entry points
(`speedrun.sh`, `scaling_laws.sh`, `miniseries.sh`, `runcpu.sh`) for the
*GPT-2 speedrun* — a different training run at 2k context, not the model
`utak`. But `README.upstream.md` and `dev/LEADERBOARD.md` both link into
it, and those files are preserved unmodified for provenance, so removing
`runs/` would have broken upstream's own documentation. The `recipe/`
pipeline is the path that builds `utak`; `runs/` is kept as upstream
reference material.

### The checkpoint metadata was renamed too

`docs/checkpoint_meta/` held the `meta_*.json` of every checkpoint — the
`user_config` block is the ground truth for every hyperparameter in the
recipe. The directories were renamed to the `utak` tags, and two that were
missing were added from the live store:

| Was | Now |
|---|---|
| `d28b-v2/` | `utak-base/` |
| `d28b-128k/`, `d28b-128k-final/` | `utak-128k/`, `utak-128k-final/` |
| `d28b-128k-sft-v5/` … `-v5e/` | `utak-sft/`, `utak-sft-followup/`, `utak-sft-glaive/`, `utak-sft-glaive-heavy/`, `utak-sft-a/` |
| `d28b-128k-sft-v6a/`, `-v6j/` | `utak-sft-b/`, `utak-sft-c/` |
| *(added)* `d28b-warmup/` | `utak-warmup/` |
| *(added)* `d28b-v2/` (sft tree) | `utak-sft-tools/` |

All twelve lineage checkpoints are now represented.

**One file was added back.** `scripts/chat_rl_curriculum.py` — the stage-6
curriculum script — was present in the working fork but had never been
copied into the repository. Without it, stage 6 of the recipe could not
run at all. It is now included.

---

## 16. Provenance

Built on the `nanochat` harness by Andrej Karpathy
([github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)). The
library, the model architecture, the tokenizer and the base training scripts
are upstream's.

The long-context SFT script, the harness-shaped data tasks, the `no_train`
tokenizer flag, the chat-template experiments, the serving stack, the guard,
the tool parser, the evaluation gates, and the `recipe/` pipeline are local
work on top of that harness.

The full chronological record — including every rejected experiment and
every measured failure — is in [`docs/EXECUTION_LOG.md`](EXECUTION_LOG.md).
