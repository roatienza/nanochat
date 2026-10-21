# nanochat (fork) — the code behind `nanochat-d28b-rl100` (v6j)

This is the training + serving code for the served model. The checkpoints
are not in git; **[docs/REPRODUCE.md](docs/REPRODUCE.md)** is the guide to
regenerating them.

| Path | What it is |
|---|---|
| `nanochat/` | the library: model, tokenizer, dataloader, optim, engine |
| `scripts/` | training entry points, one per stage |
| `tasks/` | the SFT/RL data tasks (each is a dataset + a row recipe) |
| `run_v6*.sh` | the exact touch-up training commands (v6a … v6j) |
| `chat_template*.jinja` | the serving chat templates; **v7 is the served one** |
| `serving/` | the vLLM stack + the guard proxy that owns port 8000 |
| `eval/` | the v5x gate, the HP gate, and the end-to-end checks |
| `docs/` | REPRODUCE.md, MODEL_CARD.md, TECHNICAL_REPORT.md, EXECUTION_LOG.md, checkpoint_meta/ |
| `param_audit.py` | real `GPT` class on meta device → true parameter counts |
| `mem_probe.py` | real fwd+bwd+step → measured peak GPU memory |
| `budget_real.py` | the measured feasibility table (supersedes the plan's derived one) |
| `runs/speedrun.sh` | upstream's canonical entry point (cited by PLAN.md) |
| `dev/` | how the pretraining corpus was prepared; the upstream leaderboard |
| `README.upstream.md` | nanochat's upstream README, unmodified (for provenance) |

## The served model

```
model   nanochat-d28b-rl100        root nanochat-d28b-rl100-128k-v6j
params  2,018,511,234              28L · 14H · n_embd 1792 · vocab 32768
ctx     131072                     window SSSL/512 · rope 1e5 · FA3
gates   v5x 15/15 · HP 6/7 (HP5 fails; cosmetic, pre-existing)
```

## Train, pack, serve

```bash
export NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2
./run_v6j.sh                                  # the served touch-up (240 it)
cd serving && ./pack_v6j.sh && ./serve_v6j_8000.sh
```

Full details, including the data prep and the earlier stages, are in
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## Upstream

Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat). The
library, the model architecture, the tokenizer, and the base training
scripts are upstream's; the long-context SFT script, the harness-shaped
data tasks, the `no_train` tokenizer flag, the chat-template experiments,
the serving stack, the guard, and the gates are this fork's.
