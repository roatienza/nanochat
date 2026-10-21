# nanochat-d28b-rl100 — Model Card

**Model ID (served):** `nanochat-d28b-rl100`
**Latest checkpoint:** `nanochat-d28b-rl100-128k-v6j` (`chatsft_checkpoints/d28b-128k-sft-v6j`, step 240)
**Parameters:** 2,018,511,234 (2.02 B) — 58.7 M embedding, 1,959.8 M non-embedding
**Architecture:** 28-layer decoder, 14 heads, 14 KV heads (MHA, no GQA), `n_embd` 1792, head dim 128
**Context:** 131,072 tokens (trained at 32,768, extended by Stage 7a)
**Vocab:** 32,768 (custom byte-level BPE), bf16 weights
**Final val bpb:** 0.393419
**Status:** deployed on `http://202.92.132.48:8000/v1` behind a guard proxy

---

## 1. What this model is

A 2.02 B parameter chat + tool-calling model, trained end to end on a single 8×GPU node
using the `nanochat` research harness. It is optimized for **agentic tool use on a real
harness** (the `baby-onit` OpenAI-compatible client) rather than for general knowledge
benchmarks.

Its defining characteristics:

- **One tool call per reply.** The model emits a single line of the form
  `CALL <name> <json-args>` inside `<|python_start|>…<|python_end|>` and ends its turn.
  It does **not** emit parallel calls in one turn.
- **Tool results arrive as user turns.** The serving template renders an OpenAI `tool`
  message as a user turn beginning `Tool result: {content}`, which is the exact shape the
  model was supervised on.
- **It knows when *not* to call.** A large fraction of the SFT mixture exists specifically
  to teach the boundary between "answer this" and "call a tool for this" (see §4.3).

## 2. Intended use and limitations

**Intended use.** Local/single-tenant agentic assistant: chat, answer arithmetic and
factual questions, and call a small set of tools (`get_weather`, `web_news`,
`get_file_path`, `bash`, …) when the request genuinely requires them.

**Out of scope.** This is a 2 B model trained on a research budget. It is not a
replacement for a frontier assistant, it has no reliable long-tail factual knowledge, it
has no multi-turn parallel tool calls, and its tool-argument accuracy degrades on
unfamiliar schemas.

**Known limitations, measured:**

| Limitation | Status |
|---|---|
| Answering a *date* question after a long replayed history returns a training-pool date instead of the header date | **Open** — gate `HP5` fails 0/3. Cosmetic; does not affect tool calling. |
| Parallel tool calls in one turn | Not supported by design; emits one call per turn. |
| `tool_choice="none"` may leak a `CALL` line as plain text | Pre-existing; not a regression. |
| NIAH beyond ~105 k tokens | Hits the harness's 131,072-token window limit. |

## 3. Model architecture

| Component | Value |
|---|---|
| Layers | 28 |
| Attention heads / KV heads | 14 / 14 (multi-head attention; GQA supported by the code but unused here) |
| Model dimension | 1792 |
| Head dimension | 128 |
| Feed-forward | `relu²`, hidden multiplier 4 |
| Attention window pattern | `SSSL` — sliding window tiled across layers, final layer always `L` (long) |
| Short window | 512 tokens |
| RoPE base | 100,000 |
| QK scale | 1.2 |
| Logit softcap | 15.0 |
| Weight tying | Untied (`wte` and `lm_head` are separate) |
| Normalization | RMSNorm with no learnable parameters, applied after token embedding |
| Attention kernel | Flash Attention 3 when available, SDPA fallback |
| Value-embed gating | 12 channels, `backout_layer_div` 2 |
| Position encoding | Rotary only — no learned positional embeddings |

Architectural notes carried from the `nanochat` base design: rotary embeddings, QK norm,
untied input/output embeddings, `relu²` MLP activation, no biases in linear layers, and
parameter-free RMSNorm. The `SSSL` window pattern is the mechanism that makes 128 k
context affordable: most layers only attend over a short window, with one full-context
layer per group.

The parameter count was verified by instantiating the real `nanochat.gpt.GPT` class on
the `meta` device, not by an analytic formula.

## 4. Training stages

The full pipeline is six stages. Each is a separate script under `scripts/`.

### 4.1 Stage 1 — Pre-training (base model)

`scripts/base_train.py` → checkpoint `d28b-v2`.

| Setting | Value |
|---|---|
| Steps | 40,000 |
| Context | 2,048 |
| Device batch / total batch | 4 / 32,768 |
| Embedding LR / matrix LR | 0.3 / 0.02 |
| Weight decay | 0.28 |
| Target param:data ratio | 8.0 |
| Warmup / warmdown | 40 steps / 0.65 ratio |
| Final val bpb | **0.840180** |

### 4.2 Stage 2 — Context extension (2 k → 128 k)

`scripts/base_train_long.py` → `d28b-128k` (step 1750) → `d28b-128k-final` (step 250).

| Setting | Value |
|---|---|
| Context | 131,072 |
| Short window | 512 |
| RoPE base | 100,000 |
| Total batch | 262,144 tokens |
| Embedding / matrix LR | 0.09 / 0.006 |
| Warmdown ratio | 0.9 |
| Final val bpb | **0.893391** |

The bpb rises from 0.840 to 0.893 because the validation mixture changed: it is now
evaluated at 128 k context on long packed rows, a strictly harder distribution. The two
numbers are not comparable.

### 4.3 Stages 3–6 — SFT, RL, agentic RL, and the Stage-7b touch-up ladder

See [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) for the full breakdown of the
SFT mixture, the RL algorithms, the v5x→v6j touch-up ladder, and the two root-cause
analyses that produced v6h and v6j.

Summary of the supervised stages:

| Stage | Script | Output |
|---|---|---|
| SFT (long) | `scripts/chat_sft_long.py` | `d28b-128k-sft-*` |
| RL (math) | `scripts/chat_rl.py` | `d28b-rl100` (step 177) |
| Agentic RL | `scripts/chat_rl_agentic.py` | stage 6 checkpoints |
| Stage-7b SFT touch-ups | `scripts/chat_sft_long.py` | `d28b-128k-sft-v6j` ← **current** |

**RL algorithm.** `chat_rl.py` implements what the upstream source calls "GRPO in
quotes", which is really REINFORCE with four deliberate simplifications:

1. No trust region — no KL regularization against a reference model.
2. On-policy, so no PPO probability ratio or clipping.
3. DAPO-style **token-level** normalization rather than sequence-level.
4. Advantage is `(r − μ)` — no z-score division by `σ`.

`chat_rl_agentic.py` (Stage 6) keeps the optimization core byte-identical and changes
only the reward source: `tasks/agentic.py` generates fresh instances per index so the
rollout set cannot be memorized. This was the fix for Stage 5's failure mode, where
train reward saturated at 1.0, advantage went to zero, and gradients died at step ~40.

**The current checkpoint (v6j)** is a Stage-7b SFT touch-up: 240 iterations from
`d28b-128k-sft-v6a@240`, total batch 262,144 tokens, max sequence length 32,768.

## 5. Datasets

### 5.1 Pre-training corpus

`base_data_climbmix` — 171 parquet shards, 15 GB on disk, downloaded from
`karpathy/climbmix-400b-shuffle` (NVIDIA Nemotron ClimbMix, 400 B tokens: a blend of
high-quality web text, code, and math). The **last shard is pinned as the validation
split**; all others are training. Documents are tokenized with the custom BPE and packed
into rows with BOS-bestfit.

Upstream's own log records the switch from FineWeb-Edu-100B to ClimbMix-400B as the
single largest pretraining improvement — a 27% reduction in GPT-2-speedrun time.

### 5.2 SFT mixture (v6j)

| Task | Module | Epochs | Rows/instance |
|---|---|---|---|
| Long context SFT | `tasks/longctx_sft.py` | 2 | levels 0–1, 40 instances/level |
| Tool calling | `tasks/toolcalling.py` | 1 | full |
| ToolACE | `tasks/toolace.py` | 1 | full |
| Tool follow-up | `tasks/toolfollowup.py` | 1 | full |
| Glaive multi-round | `tasks/glaive_multiround.py` | 2 | capped at 20,000 |
| Tool stop | `tasks/toolstop.py` | 6 | 6,000 |
| Harness replay | `tasks/harness_replay.py` | 2 | 6,000 |
| Harness basics | `tasks/harness_basics.py` | 3 | 8,000 |
| Harness poison | `tasks/harness_poison.py` | 3 | 6,000 |
| SmolTalk | `tasks/smoltalk.py` | 1 | capped at 20,000 |

Long-context families: `niah` (needle-in-a-haystack), `doc_qa` (structured report QA),
`multi_tool` (tool schemas + long document). `long_summary` and `multi_tool_weather`
exist in the code but are not in the v6j mixture.

The three `harness_*` tasks are **synthetic tasks generated to mirror the real
`baby-onit` harness's exact prompt shapes** — system prompt, tool block, history replay,
tool-result turns. They are the highest-epoch part of the mixture (2–3 epochs each) and
are what made the difference between a model that works on the harness and one that
does not.

### 5.3 The `no_train` mechanism

`nanochat/tokenizer.py` supports a per-message `no_train` flag. A message marked
`no_train` is rendered into the context with its loss mask set to 0, so it is **visible
but never supervised**.

This is load-bearing in two places:

- `tasks/harness_poison.py` puts the model's own *broken* answers into history. Those
  must be seen as context so the model learns to act on the current turn anyway, but
  must never be imitated as a target.
- `tasks/harness_basics.py` (v6h fix) puts the P7 anchor's spurious round-2 tool call
  into history as context only. Supervising it taught the model to re-issue the call.

## 6. Serving

### 6.1 Architecture

```
client (baby-onit, OpenAI-compatible)
   │
   ▼  :8000
guard_proxy.py            ← clamps max_tokens, aborts degeneration, scrubs history
   │
   ▼  :8010
vLLM  (nanochat-d28b-rl100-128k-v6j, GPU 0, bf16)
```

The vLLM server owns **port 8010**; the **guard proxy owns port 8000**, which is the
public endpoint. This inversion is intentional: vLLM has no server-side completion cap
that a client cannot override, so the cap has to live in front of it.

### 6.2 vLLM flags

```
--port 8010
--trust-remote-code
--enforce-eager
--gpu-memory-utilization 0.94
--max-model-len 131072
--max-num-batched-tokens 131072
--max-num-seqs 8
--enable-prefix-caching
--no-enable-chunked-prefill
--served-model-name nanochat-d28b-rl100
--enable-auto-tool-choice
--tool-call-parser nanochat
--tool-parser-plugin .../nanochat_tool_parser.py
```

`--max-num-seqs 8` (not 1) and `--enable-prefix-caching` are both required for the
harness: with a single sequence slot, one degenerate request blocks every subsequent
one, and without prefix caching a long replayed history is re-prefilled on every turn.

`--no-enable-chunked-prefill` is deliberate — it was measured to be required for
correct long-context behaviour with the `SSSL` window pattern.

### 6.3 Generation defaults

`generation_config.json` in the checkpoint directory:

```json
{ "max_new_tokens": 1024, "repetition_penalty": 1.05 }
```

These are *defaults*. A client that sends an explicit `max_tokens` overrides
`max_new_tokens`, which is exactly why the guard proxy exists.

### 6.4 Guard proxy

`guard_proxy.py` (807 lines) is a FastAPI/httpx transparent proxy with six
responsibilities:

1. **Clamps `max_tokens`** to 4096 regardless of what the client asks for.
2. **Always streams upstream**, so it has abort power for every client shape;
   non-streaming clients get the chunks reassembled into a full JSON response.
3. **Aborts degenerate output mid-stream** and returns a clean OpenAI-compatible
   finish chunk. Degenerate `tool_calls` slots are dropped so the harness never
   executes garbage.
4. **Wall-clock cap** of 45 s. Calibrated against the real model: the slowest
   legitimate turn observed was 8.8 s, so 45 s leaves 5× headroom.
5. **Passes sampling parameters through untouched**, so the evaluation gates measure
   the model, not the proxy.
6. **Scrubs degenerate assistant turns out of the replayed history** (see §7).

Degeneration detection, with every threshold calibrated against 20 real model answers
(18 good, 2 degenerate):

| Signal | Threshold |
|---|---|
| Identical characters in a row | 48+ |
| Short-period cycle (period 2–8) | ≥ 8 reps, ≥ 32 chars, pathological block |
| Literal `\n` pairs, total | ≥ 4 |
| Literal `\n` pairs, consecutive | ≥ 3 |
| `##` heading density | ≥ 1.5 per 100 chars |
| Most-repeated segment | ≥ 3 occurrences |
| Total answer length | ≥ 2,000 chars |

The short-period-cycle detector is the important one: the `\n\n\n` pathology is a
period-2 cycle, which a "48 identical characters in a row" test can never see.

### 6.5 Chat template

`chat_template_v7.jinja`, shipped inside the checkpoint directory.

The tools block is placed on the **first** user turn. This was measured, not assumed:
the v6 template placed it on the *last* user turn, which broke the live tool loop —
when the model was about to answer from a tool result, the last user turn *is* the
"Tool result:" turn, so the block landed immediately before the answer. That is a shape
zero training rows teach, and it caused `P1`/`P13`/`LOOP-T2` failures.

The template also renders each tool compactly (`separators=(",", ":")`) and
insertion-ordered, matching the Stage-4 training recipe exactly. Spaced or sorted JSON
is out of distribution and makes the model emit garbage arguments.

### 6.6 Tool-call parser

`nanochat_tool_parser.py` (236 lines) is registered with vLLM as a `--tool-parser-plugin`.
The model emits calls inside python tags; the tokenizer's `skip_special_tokens=True`
default strips the special tokens, so the parser matches the text form
`CALL <name> <json-args>`. Both the special-token and text spellings are accepted.

## 7. The two root causes behind v6h and v6j

These are documented in full in [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md);
the short form, because they are the most instructive results in the project:

**v6h — a supervised target that taught the wrong thing.** The P7 anchor rows taught
`[get_weather result → ANOTHER get_weather call for city2]` as a *supervised target*.
The `P1` probe is exactly the anchor's mid-row state, so the model completed the row
with the anchor's second call and produced a spurious re-call. The fix was to demote
the round-2 call to context-only (`no_train`).

**v6j — a two-character escape in the training data.** `tasks/harness_basics.py` used
`"\\n\\n"` (an escaped *backslash*) where a real newline was meant. The supervised
target therefore contained the two characters `\` `n`, and the model learned to emit a
literal backslash-n. At inference this degenerates into a period-2 byte-BPE cycle:
thousands of tokens of `\n\n\n…` for the full token cap — 120 s per answer, saved into
`session_history.jsonl`, and replayed on every later turn as an in-context attractor.

The retrain removes the *source* of the behaviour, but a poisoned history re-teaches it
in context. Both fixes are required, and the measurement proves it:

| Condition | Result |
|---|---|
| Raw v6j + poisoned history | 15.7 s of garbage (504 literal `\n` in 512 tokens) |
| Raw v6j + **scrubbed** history | 0.6 s, clean |
| v6j **via the guard** + poisoned history | 0.6 s, clean |

## 8. Evaluation

### 8.1 Two gates, both run against the live server

Both gates import `baby_onit` itself, so the system prompt, tool schemas, history
replay, and request shape are byte-identical to what the real harness sends.

**`verify_v5x.py` — the real-harness gate (15 checks).** Temperature 0.6, top_p 0.95, the
real 8-tool schema list, streaming, multiple seeds.

| ID | Probe | Expectation |
|---|---|---|
| F1 | fresh `hello` ×5 | greeting, no call |
| F2 | fresh `2+2` ×5 | `4`, no call |
| F3 | fresh files → `bash` ×5 | `CALL bash` with `ls` |
| F4 | fresh weather ×5 | `CALL web_search` |
| F5 | fresh fetch ×5 | `CALL fetch_content` |
| R1–R5 | replayed history, 5 variants ×3 | same expectations as fresh |
| LOOP | 4-turn OpenAI loop | call → result → answer → new call |
| P7 | bare 2-round chain ×10 | **0** over-calls |
| P13 | question already answered by round 1 ×5 | **0** over-calls |
| P1 | 1-round answer ×5 | answers from the result |
| MT24 | `multi_tool` at ~24 k | `CALL get_weather {"city":"Oslo",…}` |

**`verify_hp.py` — the poison gate (7 checks).** Same probes, but with a deliberately
poisoned history replayed.

### 8.2 Results for the current checkpoint

Measured 2026-10-20 23:02–23:03 against v6j, both on the gate server (8016) and on the
live deployment (8000).

```
v5x gate: 15/15 PASS
  F1 true  F2 true  F3 true  F4 true  F5 true
  R1 true  R2 true  R3 true  R4 true  R5 true
  LOOP true  P7 true  P13 true  P1 true  MT24 true

HP gate: 6/7 PASS
  HP1 true  HP2 true  HP3 true  HP4 true
  HP5 false  <-- poisoned date (known, open)
  HP6 true  HP7 true
```

### 8.3 Touch-up ladder

| Checkpoint | v5x gate | HP gate | Verdict |
|---|---|---|---|
| v6g | 12/15 | 6/7 | reference |
| v6h | 14/15 | 6/7 | anchor-stop fix |
| v6i | 15/15 | regressed | **rejected** — taught that any pool date is acceptable |
| **v6j** | **15/15** | **6/7** | **current** — literal-`\n` data fix |

v6i is the cautionary case: it hit 15/15 on the real-harness gate but regressed the
poison gate, teaching the wrong association (a long history makes *any* pool date
acceptable). It was rejected and v6h was kept. A gate that improves one behavior and
degrades another is not an improvement.

### 8.4 Base model benchmark (CORE)

For reference, the pre-trained base model at step 10,080:

| Task | Accuracy | Centered |
|---|---|---|
| hellaswag_zeroshot | 0.370544 | 0.160725 |
| arc_easy | 0.555556 | 0.407407 |
| arc_challenge | 0.283276 | 0.044369 |
| copa | 0.560000 | 0.120000 |
| piqa | 0.674102 | 0.348204 |
| openbook_qa | 0.332000 | 0.109333 |
| winograd | 0.593407 | 0.186813 |
| winogrande | 0.512234 | 0.024467 |
| boolq | 0.524465 | −0.251408 |
| squad | 0.163292 | 0.163292 |
| **CORE (centered)** | | **0.138977** |

### 8.5 End-to-end checks

Beyond the gates, the deployment was verified with the real `baby-onit` client:

- The user's exact turn sequence (`hello` / `who are you` / `date today` /
  `weather today`) × 10 → 40 turns, 0 slow, 0 degenerate, worst turn 21.4 s.
- Real `baby-onit agent_loop` end-to-end × 6 → 48/48 turns clean.
- Poisoned replay of `date today`: **120.4 s → 0.6 s**.
- Fresh `hello`: **0.37 s**. 4 parallel `hello` requests: all 200 OK in ~0.4 s each.

## 9. Reproducing the current checkpoint

```bash
# 1. Train (GPU 7, ~2 h, 240 iterations) — run script lives in the repo
cd /data/rowel/nanochat && ./run_v6j.sh

# 2. Pack to a vLLM-ready directory — pack/serve scripts live in vllm_d28b
cd /data/rowel/nanochat_store/v2/vllm_d28b && ./pack_v6j.sh
#    -> nanochat-d28b-rl100-128k-v6j/

# 3. Gate before deploying (serve on 8016)
cd /data/rowel/nanochat_store/v2/vllm_d28b && ./serve_v6j.sh
python verify_v5x.py 8016
python verify_hp.py  8016

# 4. Deploy (guard on 8000 -> vllm on 8010)
cd /data/rowel/nanochat_store/v2/vllm_d28b && ./serve_v6j_8000.sh
```

`serve_v6j_8000.sh` is restart-safe: it kills stale listeners on 8000/8010 first,
respawns the guard if it dies, and traps `EXIT` so vLLM never outlives the script.

## 10. File map

| Path | Contents |
|---|---|
| `nanochat/gpt.py` | Model definition (`GPT`, `GPTConfig`) |
| `nanochat/tokenizer.py` | BPE tokenizer, conversation rendering, `no_train` support |
| `nanochat/engine.py` | Batch generation engine used by RL and evaluation |
| `nanochat/checkpoint_manager.py` | Save/load with optimizer state and metadata |
| `scripts/base_train.py` | Stage 1 pre-training |
| `scripts/base_train_long.py` | Stage 2 context extension |
| `scripts/chat_sft_long.py` | Stages 3 and 6 — long-context SFT and touch-ups |
| `scripts/chat_rl.py` | Stage 4 — math RL |
| `scripts/chat_rl_agentic.py` | Stage 5 — agentic RL |
| `scripts/chat_eval.py`, `scripts/base_eval.py` | Evaluation harnesses |
| `tasks/*.py` | Dataset tasks, one module per mixture component |
| `chat_template_v7.jinja` | Serving template (shipped in the checkpoint dir) |
| `tasks/test_no_literal_newline.py` | Regression test for the v6j bug |
| `run_v6*.sh` | One script per touch-up run, each documenting its own fix |
| `EXECUTION_LOG.md` | Chronological log of every experiment and failure |
| `docs/TECHNICAL_REPORT.md` | Full technical report |

Serving artifacts live outside the repo, under
`/data/rowel/nanochat_store/v2/vllm_d28b/`: `pack_v6*.sh` (checkpoint → vLLM export),
`serve_v6*.sh` (deployment), `guard_proxy.py`, `nanochat_tool_parser.py`,
`verify_v5x.py`, and `verify_hp.py`.

## 11. Ethics and limitations

This model is a research artifact trained on web-derived text. It inherits the biases,
inaccuracies, and potentially harmful content of that corpus. It performs real tool calls
when a request matches a tool's purpose, so a deployed copy should be sandboxed and its
tool execution subject to the caller's own permission model. The model's tool-argument
accuracy on schemas it has not seen during training is low, and it should not be
trusted with destructive operations without a human in the loop.

## 12. Citation and provenance

Built on the `nanochat` harness by Andrej Karpathy. This checkpoint, the dataset
mixture, the guard proxy, the tool parser, and the evaluation gates are local work on top
of that harness. The full chronological record of how this checkpoint was reached,
including the rejected experiments, is in `EXECUTION_LOG.md`.
