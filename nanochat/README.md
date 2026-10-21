# utak — the training and serving code

This is the code that builds, trains, serves and verifies `utak`.

| Path | What it is |
|---|---|
| `recipe/` | **the build pipeline** — one script per stage, `run_all.sh` drives it |
| `nanochat/` | the library: model, tokenizer, dataloader, optim, engine, sandbox |
| `scripts/` | training entry points, one per stage |
| `tasks/` | the SFT/RL data tasks (each is a dataset + a row recipe) |
| `chat_template_v7.jinja` | the serving chat template — the only one that is served |
| `serving/` | the vLLM stack + the guard proxy that owns the public port |
| `eval/` | the v5x gate, the HP gate, the end-to-end test, the detector tests |
| `docs/` | UTAK_RECIPE.md, MODEL_CARD.md, TECHNICAL_REPORT.md, EXECUTION_LOG.md, checkpoint_meta/ |
| `param_audit.py` | real `GPT` class on the meta device → true parameter counts |
| `mem_probe.py` | real fwd+bwd+step → measured peak GPU memory |
| `budget_real.py` | the measured feasibility table (supersedes the plan's derived one) |
| `dev/` | how the pretraining corpus was prepared; the upstream leaderboard |
| `README.upstream.md` | nanochat's upstream README, unmodified (for provenance) |

## The model

```
model   utak                root utak-128k
params  2,018,511,234       28L · 14H · n_embd 1792 · vocab 32768
ctx     131072              window SSSL/512 · rope 1e5 · FA3
gates   v5x 15/15 · HP 6/7 (HP5 fails; cosmetic, pre-existing)
speed   445 ms/step · 73,655 tok/s · 43.5% BF16 MFU on 4x A100-40GB
        (40% of the original wall clock was evaluation; see "Make it fast")
```

## Build it

```bash
export NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2
./recipe/run_all.sh                 # everything, from scratch
./recipe/run_all.sh --list          # the stages
./recipe/run_all.sh --from 7        # resume at the context extension
./recipe/run_all.sh --only 9        # one stage
```

Every path, GPU, port, model name and checkpoint tag lives in
**`recipe/config.sh`**. Nothing else hardcodes one.

The stages, in the order they must run:

```
 0  data          the tokenizer + the ClimbMix corpus
 1  pretrain      base model, 2k ctx
 2  sft_tools     tool-calling SFT, 2k ctx
 3  warmup        GSM8K warmup SFT, fresh optimizer
 4  rl_math       math RL on GSM8K
 5  rl_agentic    agentic RL on sandbox-verified environments
 6  rl_curriculum weighted-block + exemplar curriculum
 7  longctx       context extension 2k -> 128k
 8  sft_long      long-context SFT
 9  touchup       the six-rung touch-up ladder  <-- the served checkpoint
10  pack          checkpoint -> vLLM model dir
11  serve         guard on :8000 -> vLLM on :8010
12  gate          the v5x and HP gates
13  e2e           the end-to-end acceptance test
14  regression    the offline unit tests
15  scrub_test    the live-server hypothesis test (the guard's history scrub)
```

The RL chain (4-6) runs **before** the context extension (7): the
131,072-context model is initialized from the RL weights.

## Serve it

```bash
./recipe/10_pack.sh     # checkpoint -> $VLLM_DIR/utak-128k/
./recipe/11_serve.sh    # guard on :8000 -> vLLM on :8010
```

The guard is not optional. It is the load-bearing half of the fix for the
`\n` degeneration: the retrain removes the source, the guard's history
scrub stops an already-poisoned history from re-infecting the model.

## Make it fast

**The GPU was never the bottleneck.** The run that produced the served
model did 445 ms/step at **73,655 tok/s and 43.5% BF16 MFU** on 4x
A100-40GB, peaking at 34,071 / 40,960 MiB (83% of the card). That is a
respectable number for a 2B model on A100.

The waste was elsewhere. The same run took **57.35 h of wall clock for
34.33 h of training** — 40% of it was overhead:

| Where the time went | Cost |
|---|---|
| evaluation (556 passes x 147 s) | **25.4 h** |
| the training steps themselves | 34.3 h |

The evaluation was the single biggest item in the entire pretrain, and it
was **more precise than it needed to be**. From the same log, the standard
deviation of one bpb estimate at 41,943,040 tokens is 0.000065, while the
real improvement over 500 steps at the end of training is 0.00021. The
original evaluation was resolving a signal 3x smaller than it needed to.

### What changed

| Change | Effect | Where |
|---|---|---|
| Evaluation budget: 41.9M -> 4.2M tokens, every 500 -> every 1000 steps | ~10x less eval time; noise (0.000204) still matches the signal (0.00021) | `recipe/config.sh` |
| `--core-metric-every` set explicitly | stops the default 2000 from firing 138x (4.3 h) | `recipe/01_pretrain.sh` |
| `PrefetchLoader` | loader stall 6.11 ms -> **0.03 ms** per step | `nanochat/dataloader.py` |
| Chunked cross-entropy (the `TODO` in `gpt.py`) | ~500-900 MiB less peak memory, same speed | `nanochat/gpt.py` |
| `torch.compile` in the 5 trainers that lacked it | fuses the attention and MLP matmuls | `scripts/` |
| Threading / allocator / NCCL env | stops the ranks fighting over cores | `recipe/config.sh` |

### What did NOT change (measured, and deliberately left alone)

- **Flash Attention 3 is already the fastest option on A100.** At the real
  d28 shapes (B=4, T=2048, 14 heads, head_dim 128): FA3 **0.507 ms**,
  SDPA-flash 0.611 ms, SDPA-cudnn 1.020 ms, SDPA-efficient 1.175 ms.
  Switching away from FA3 would be a regression.
- **The batch size is not the lever.** d28 at bs=4 already peaks at
  34,071 MiB and bs=5 does not fit. The memory is bound by the fp32 logits
  block, which is exactly what the chunked CE attacks.
- **The RL evaluation size is not reduced.** The measured confidence
  interval on pass@1 is +/- 3.7 pt at n=400, and the effect the RL stages
  chase is of that same order. Halving n would hide the signal.

### The knobs

Everything is in `recipe/config.sh`. The fast settings are the defaults;
the original settings are one override away.

```bash
# reproduce the original run exactly
PRETRAIN_EVAL_TOKENS=41943040 PRETRAIN_EVAL_EVERY=500 ./recipe/01_pretrain.sh

# a bit-exact reproduction of the original (single-shot cross-entropy)
./recipe/01_pretrain.sh   # with --ce-chunk 0
```

| Variable | Default | What it controls |
|---|---|---|
| `PRETRAIN_EVAL_TOKENS` | 4194304 | validation tokens per eval, stage 1 |
| `PRETRAIN_EVAL_EVERY` | 1000 | steps between evals, stage 1 |
| `SFT_EVAL_TOKENS` / `SFT_EVAL_EVERY` | 2097152 / 500 | stage 2 |
| `CORE_METRIC_EVERY` | 999999 | the CORE benchmark (111 s per pass) |
| `RL_EVAL_EVERY` | 20 | steps between RL evals |
| `RL_EVAL_EXAMPLES_04/05/06` | 400 / 150 / 150 | RL eval size per stage |
| `COMPILE_BASE` | 1 | `torch.compile` on/off |
| `--ce-chunk` | 1024 | cross-entropy chunk (0 = single shot) |
| `NANOCHAT_COMPILE=0` | — | env override for the compile flag |
| `UTAK_PREFETCH=0` | — | env override for the prefetching loader |

### Correctness

Every change above was checked against the original, not just for speed:

- **The prefetch is bit-identical.** Running the same training with and
  without it gives the same losses to the last printed digit and the same
  validation bpb (`tests/test_prefetch.py`).
- **The chunked CE is exact to 8e-8 relative.** The logits are
  *bit-identical*; only the order of the fp32 summation over tokens
  differs. Use `--ce-chunk 0` for a bit-exact reproduction
  (`tests/test_chunked_ce.py`).
- **`torch.compile` is exact.** Same losses, same validation bpb.

## Verify it

```bash
./recipe/12_gate.sh      # v5x 15/15, HP 6/7
./recipe/13_e2e.sh       # the real baby-onit loop, against the live server
./recipe/14_regression.sh
./recipe/15_scrub_test.sh
```

## The documentation

- **[docs/UTAK_RECIPE.md](docs/UTAK_RECIPE.md)** — the step-by-step build
  and training recipe: every stage, every hyperparameter, and the measured
  results. Start here.
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — what the model is.
- [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) — the engineering
  report: the failures, the measurements, the rejected experiments.
- [docs/EXECUTION_LOG.md](docs/EXECUTION_LOG.md) — the chronological
  record, including every experiment that failed.

## Upstream

Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat). The
library, the model architecture, the tokenizer, and the base training
scripts are upstream's; the long-context SFT script, the harness-shaped
data tasks, the `no_train` tokenizer flag, the chat-template experiments,
the serving stack, the guard, the gates, and the `recipe/` pipeline are
this fork's.
