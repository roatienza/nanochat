# REPRODUCE.md — rebuild the served `utak` (v6j)

Everything needed to take this tree from "no model" to "the served
`utak` on port 8000". The checkpoints themselves are **not**
in git (8 GB each); this document tells you how to regenerate them.

**Target:** `utak` — 2.019 B params, 131 072 ctx, served under
the name `utak`, root checkpoint
`utak-128k`.

---

## 0. What is in this tree

```
nanochat/
  nanochat/            the library: model, tokenizer, dataloader, optim, engine
  scripts/             the training entry points (one per stage)
  tasks/               the SFT/RL data tasks (each is a dataset + a row recipe)
  run_v6*.sh           the exact touch-up training commands (v6a ... v6j)
  chat_template*.jinja the serving chat templates (v7 is the SERVED one)
  serving/             the vLLM + guard serving stack
    guard_proxy.py       the public endpoint on :8000 (safety policy)
    nanochat_tool_parser.py  the vLLM tool-call parser plugin
    export_hf_128k.py    checkpoint -> safetensors + config.json
    pack_v6j.sh          ckpt -> vLLM model dir
    serve_v6j_8000.sh    start vLLM :8010 + guard :8000
    model_assets/        the aux files (arch code, tokenizer, generation cfg)
  eval/                the gates and the end-to-end checks
    verify_v5x.py        the 15-point real-harness gate
    verify_hp.py         the 7-point poisoned-history gate
    e2e_harness.py       the live agent-loop acceptance test
    rate_check.py        slow/incomplete rate on the reported sequence
    test_detector.py     guard degeneracy detector (regression)
    test_guard_regression.py / test_segment_detector.py / test_scrub.py
  docs/
    MODEL_CARD.md        what the model is
    TECHNICAL_REPORT.md  how it was built, in depth
    EXECUTION_LOG.md     the dated experiment log
    checkpoint_meta/     every checkpoint's meta JSON (the lineage of record)
```

### Dependencies

| Component | Version | Needed for |
|---|---|---|
| Python | 3.10+ | everything |
| PyTorch | 2.7.x (CUDA 12.6) | training |
| `rustbpe` | from `pyproject.toml` | tokenizer training |
| `tiktoken` | | tokenizer inference |
| `vllm` | ≥ 0.31 | serving (the tool-parser plugin imports `vllm.entrypoints.*`) |
| `fastapi` + `uvicorn` | | the guard proxy |
| `wandb` | optional | run logging (`--run dummy` disables it) |

```bash
cd nanochat
uv venv && source .venv/bin/activate
uv pip install -e . -U
# serving extras
uv pip install "vllm>=0.31" fastapi uvicorn
```

Set the data root once:

```bash
export NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2
```

Everything (checkpoints, base data, tokenizer, exports) lives under it.

---

## 1. The lineage

Read from the `meta_*.json` in `docs/checkpoint_meta/` — these are the
authoritative record of every run, and the only record of v5b–v5e (no run
script for those was ever written).

| # | Stage | Output tag | Source | ctx | iters | val bpb |
|---|---|---|---|---|---|---|
| 1 | Pretrain | `utak-base` | — | 2 048 | 277 760 | 0.715437 |
| 2 | Context extension | `utak-128k` | `rl:utak-rl@177` | 131 072 | 1 750 | 0.902946 |
| 2b | Context ext. (final) | `utak-128k-final` | `base:utak-128k@1750` | 131 072 | 250 | 0.893391 |
| 3 | SFT (128k) | `utak-128k-sft` | `base:utak-128k-final@250` | 131 072 | 600 | 0.361935 |
| 3b | SFT (final) | `utak-128k-sft-final` | `sft:utak-128k-sft@500` | 131 072 | 100 | 0.355233 |
| 4 | Touch-up v5 | `utak-sft` | `base:utak-128k-final@250` | 131 072 | 800 | 0.524880 |
| 5 | Touch-up v5b | `utak-sft-a` chain | `sft:…-v5@800` | 8 192 | 120 | 0.421962 |
| 6 | Touch-up v5c | ″ | `sft:…-v5b@120` | 8 192 | 80 | 0.407229 |
| 7 | Touch-up v5d | ″ | `sft:…-v5c@80` | 8 192 | 100 | 0.399349 |
| 8 | Touch-up v5e | ″ | `sft:…-v5d@100` | 8 192 | 60 | 0.397745 |
| 9 | Touch-up v6a | `utak-sft-b` | `sft:…-v5e@60` | 32 768 | 240 | 0.391964 |
| 10 | **Touch-up v6j** | **`utak-sft-c`** | `sft:…-v6a@240` | 32 768 | 240 | **0.393419** |

The RL branches (GSM8K REINFORCE → `utak-rl`, agentic RL → the same tag)
are part of the upstream tree but **not on the served model's path**: the
served v6j is a pure SFT touch-up chain rooted at `utak-128k-final`. They
are included because `scripts/chat_rl.py` and `scripts/chat_rl_agentic.py`
are part of the project and `utak-128k` was extended *from* the RL
checkpoint.

> ⚠️ **v5b–v5e have no run script.** Their exact hyperparameters survive
> only in `docs/checkpoint_meta/utak-sft{,b,c,d,e}/meta_*.json`.
> If you re-run them, read the flags from there, not from a script.

---

## 2. Data preparation

### 2.1 Pretraining corpus

ClimbMix-400B (NVIDIA Nemotron), downloaded shard-by-shard on demand:

```bash
python -m nanochat.dataset -n 170        # ~170 shards; 171 incl. the validation shard
python -m scripts.tok_train              # train the BPE tokenizer on it
```

The tokenizer lands in `$NANOCHAT_BASE_DIR/tokenizer/`. The **last shard is
pinned as validation** — do not train on it.

### 2.2 SFT / RL corpora

The touch-up tasks pull from the HF hub (cached under
`$NANOCHAT_BASE_DIR/task_data/`) and from three local corpora:

| Corpus | Path | Used by |
|---|---|---|
| hermes function-calling | `/data/rowel/data/hermes/func-calling.json` | `tasks/toolcalling.py`, `tasks/toolfollowup.py` |
| ToolACE | `/data/rowel/data/toolace/data.json` | `tasks/toolace.py` |
| glaive-v2 | `/data/rowel/data/glaive/glaive-function-calling-v2.json` | `tasks/glaive_multiround.py` |

Override with `TOOLCALL_DATA` / `TOOLACE_DATA` if your paths differ.

### 2.3 The v6j mixture

`run_v6j.sh` is the recipe. The mixture (from `--*` flags):

| Task | Epochs | Notes |
|---|---|---|
| `LongContextSFT` | 2 | families `niah,doc_qa,multi_tool`, levels 0–1, 40/level |
| `ToolCalling` | 1 | hermes |
| `ToolACE` | 1 | |
| `ToolFollowUp` | 1 | hermes result→prose |
| `GlaiveMultiRound` | 2 | cap 20 000 |
| `ToolStop` | 6 | 6 000 |
| `HarnessReplay` | 2 | 6 000 |
| `HarnessBasics` | 3 | 8 000 |
| `HarnessPoison` | 3 | 6 000 |
| `SmolTalk` | 1 | cap 20 000 |

Learning rates: emb 0.0036 · unemb 0.0001 · matrix 0.00024 · scalar 0.006 ·
wd 0.028 · warmup 0.02 · warmdown 0.5 · final_lr_frac 0.
`--device-batch-size 1 --total-batch-size 262144 --max-seq-len 32768`.

---

## 3. Training

### 3.1 Pretrain (stage 1)

```bash
python -m scripts.base_train --run headline-d28-v2 \
  --depth 28 --aspect-ratio 64 --head-dim 128 --max-seq-len 2048 \
  --window-pattern SSSL --device-batch-size 4 --total-batch-size 32768 \
  --embedding-lr 0.3 --unembedding-lr 0.008 --matrix-lr 0.02 --scalar-lr 0.5 \
  --weight-decay 0.28 --warmup-steps 40 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --eval-every 500 --eval-tokens 41943040 --save-every 5000 --model-tag utak-base
```

### 3.2 Context extension (stage 2)

`scripts/base_train_long.py`, 1 750 steps to `utak-128k` then 250 to
`utak-128k-final`. The full flag block is in
`docs/checkpoint_meta/base_checkpoints/utak-128k-final/meta_000250.json`
(`user_config`).

### 3.3 SFT (stage 3)

`scripts/chat_sft.py` for the 131 072-ctx SFT, `scripts/chat_sft_long.py`
for the touch-ups. Flags are in
`docs/checkpoint_meta/chatsft_checkpoints/utak-128k-sft/meta_000500.json`.

### 3.4 The touch-up chain (the served model's path)

```bash
./run_v6a.sh   # v6a: tools-block placement rows  (240 it, ctx 32768)
./run_v6j.sh   # v6j: the literal-\n data fix   (240 it, ctx 32768)  <- SERVED
```

Both read their source checkpoint by `--model-tag` / `--model-step` and
write `--output-tag`. The script now `cd`s to its own directory, so it can
be run from anywhere.

**The two bugs that shaped v6a and v6j — do not reintroduce them:**

- **v6h → v6j (the literal `\n` bug).** `tasks/harness_basics.py` had
  `"\\n\\n"` (an escaped *backslash*) where a real newline was meant. The
  supervised *target* therefore contained the two characters `\n`, the model
  learned to emit them, and at inference it degenerated into a period-2
  byte-BPE cycle — 4 096 tokens of literal `\n`, 120 s, saved to the
  harness's session history, and replayed forever as an in-context
  attractor. Fixed in `tasks/harness_basics.py`; guarded by
  `tasks/test_no_literal_newline.py`, which **fails if it ever returns**.
- **v6h (the anchor-stop bug).** A P7-anchor row made the model's *own*
  round-2 tool call a supervised target, so the model learned to re-call
  after the result. Fixed by making the row context-only via the `no_train`
  flag in `nanochat/tokenizer.py`.

`run_v6b.sh` … `run_v6h.sh`, `run_v6i.sh` are the other attempts kept for
the record. **v6i was rejected**: it scored 15/15 on the v5x gate but
*regressed* the poison gate (it had learned "long history ⇒ any pool date
is acceptable").

### 3.5 RL (off the served path, but part of the project)

```bash
python -m scripts.chat_rl        --run stage5-gsm8k   ...   # GSM8K REINFORCE
python -m scripts.chat_rl_agentic --run stage6-pilot   ...   # agentic sandbox
```

The RL algorithm is REINFORCE with DAPO-style token normalisation;
advantage = `r − μ` (mean over the sampled group). Output tag
`utak-rl`, step 177.

---

## 4. Pack to a vLLM model dir

```bash
cd nanochat/serving
NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2 \
OUT=/data/rowel/nanochat_store/v2/vllm_d28b/utak-128k \
./pack_v6j.sh
```

This runs `export_hf_128k.py` (checkpoint → `model.safetensors` +
`config.json`), patches the config to 131 072, copies the aux files from
`serving/model_assets/`, and installs **`chat_template_v7.jinja`** as
`chat_template.jinja`.

The resulting directory:

```
utak-128k/
  model.safetensors            8.07 GB, bf16
  config.json                  nanochat_d28, seq 131072, n_embd 1792
  configuration_nanochat.py    auto_map entry
  modeling_nanochat_d28.py     the vLLM architecture implementation
  tokenizer.json  tokenizer_config.json
  generation_config.json       max_new_tokens 1024, repetition_penalty 1.05
  chat_template.jinja          template v7
```

> **The chat template is part of the model.** v7 puts the tools block on the
> **first** user turn. The v6 experiment (tools block on the *last* user
> turn) broke the live tool loop, because when the model answers from a tool
> result the last user turn *is* the "Tool result:" turn — so the block
> landed right before the answer, a shape zero training rows teach. v5
> placement matches ToolStop / GlaiveMR / ToolFollowUp exactly.

---

## 5. Serve

```bash
cd nanochat/serving
MODEL_DIR=/data/rowel/nanochat_store/v2/vllm_d28b/utak-128k \
VLLM_BIN=/path/to/venv/bin/vllm \
PY=/path/to/venv/bin/python \
GPU=0 \
./serve_v6j_8000.sh
```

### Topology

```
client ──▶ :8000  guard_proxy.py    public endpoint, enforces the safety policy
                │
                └─▶ :8010  vLLM    the model (internal)
```

**vLLM flags and why each one is there:**

| Flag | Why |
|---|---|
| `--max-model-len 131072` | the model's trained context |
| `--max-num-batched-tokens 131072` | match the window; no chunked prefill |
| `--max-num-seqs 8` | **not 1** — with 1, one runaway request queues everything behind it |
| `--enable-prefix-caching` | the harness replays long histories; the prefix is reused |
| `--no-enable-chunked-prefill` | long single prompts stay in one prefill |
| `--enable-auto-tool-choice` | OpenAI-style tool calling |
| `--tool-call-parser nanochat` | the model's `CALL name {args}` format is not JSON |
| `--tool-parser-plugin …/nanochat_tool_parser.py` | the parser that understands it |
| `--served-model-name utak` | the name clients ask for |
| `--enforce-eager` | no CUDA-graph capture; predictable startup |
| `--gpu-memory-utilization 0.94` | fits 8 GB of weights at 131 072 ctx |

### The guard (`serving/guard_proxy.py`)

The guard owns the public port and enforces six things:

1. **Clamps `max_tokens` ≤ 4096**, ignoring a client's 32 768.
2. **Always streams** upstream, and reassembles for non-streaming clients.
3. **45 s wall-clock cap** per request.
4. **Aborts degenerate output mid-stream** and returns a clean SSE finish.
5. **Drops degenerate tool-call slots** (the garbage often arrives as tool
   *arguments*, invisible to a content-only check).
6. **Scrubs degenerate assistant turns out of the replayed history** — this
   is the load-bearing one (see below).

### The degeneracy detector

| Signal | Threshold |
|---|---|
| identical-char run | 48+ |
| short-period cycle | periods 2–8, 8 reps, ≥ 32 chars |
| literal `\n` pairs | ≥ 4 total, ≥ 3 consecutive |
| heading density | ≥ 1.5 `##` per 100 chars |
| repeated segment | ≥ 3 occurrences |
| answer length | > 2 000 chars |

The short-period detector exists because the original one required 48+
*identical* characters — which a 2-character cycle can never match. Every
threshold was calibrated against 20 real model answers: **0 false
positives**.

### Why both the retrain and the scrub are needed

Measured, not inferred:

| | result |
|---|---|
| raw v6j + poisoned history | 15.7 s of garbage (504 literal `\n`) |
| raw v6j + **scrubbed** history | **0.6 s, clean** |
| v6j **via the guard** + poisoned history | **0.6 s, clean** |

The retrain removes the *source*; the guard's scrub stops an already-poisoned
history from re-teaching it in context. Either one alone is insufficient.

---

## 6. Evaluate

### The v5x gate — 15/15 PASS

The real-harness gate: it drives the actual `baby-onit` agent loop against
the live endpoint.

```bash
cd nanochat/eval
BABY_ONIT_DIR=/data/rowel/tmp/baby-onit \
GATE_DATA_DIR=/data/rowel \
python verify_v5x.py 8000
```

### The HP gate — 6/7 (HP5 fails)

Poisoned-history gate. Same invocation, `verify_hp.py`.

**HP5 is a known, pre-existing failure**: asking for the date after a long
replayed history sometimes returns a training-pool date instead of the
header date. It is cosmetic, it does not affect tool calling, and it
pre-dates this work (v6h had the same). A real fix needs training rows
built from the actual session text rather than synthetic stand-ins.

### The end-to-end acceptance test

```bash
BABY_ONIT_DIR=/data/rowel/tmp/baby-onit python e2e_harness.py   # 8 turns
BABY_ONIT_DIR=/data/rowel/tmp/baby-onit python rate_check.py    # N repeats
```

### Regression tests

```bash
python nanochat/tasks/test_no_literal_newline.py   # the v6j bug cannot return
cd nanochat/eval
python test_detector.py && python test_guard_regression.py \
  && python test_segment_detector.py && python test_scrub.py
```

---

## 7. The served configuration, as measured

```
host          http://<host>:8000/v1
model         utak
root          utak-128k
max_model_len 131072
params        2,018,511,234
architecture  28 layers · 14 heads · 14 kv heads (MHA) · n_embd 1792
              vocab 32768 · head_dim 128 · window SSSL/512 · rope 1e5
              QK-norm, qk_scale 1.2, logit softcap 15, relu² MLP, FA3
val bpb       0.393419
gates         v5x 15/15 · HP 6/7
```

---

## 8. Provenance

Upstream: [karpathy/nanochat](https://github.com/karpathy/nanochat) (the
library, the model architecture, the tokenizer, the base training scripts).
The fork adds the long-context SFT script, the harness-shaped data tasks,
the `no_train` tokenizer flag, the chat-template experiments, the vLLM
serving stack, the guard, and the evaluation gates.

Deep background: `docs/MODEL_CARD.md` (what the model is) and
`docs/TECHNICAL_REPORT.md` (how it was built, with the root-cause analyses).
The dated experiment history is `docs/EXECUTION_LOG.md`.
