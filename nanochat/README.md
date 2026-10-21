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
