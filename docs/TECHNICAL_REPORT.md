# nanochat-d28b-rl100 — Technical Report

**Checkpoint:** `nanochat-d28b-rl100-128k-v6j` (`chatsft_checkpoints/d28b-128k-sft-v6j`, step 240)
**Served as:** `nanochat-d28b-rl100` on `http://202.92.132.48:8000/v1`
**Final val bpb:** 0.393419 · **Params:** 2,018,511,234 (2.02 B) · **Context:** 131,072

This document is the engineering companion to [`../MODEL_CARD.md`](../MODEL_CARD.md). The
model card describes *what* the model is; this report describes *how it was built*,
including the failures, the measurements, and the rejected experiments. Every number
here was read from a checkpoint, a log, or a live measurement — not recalled.

---

## 1. Model architecture

Defined in `nanochat/gpt.py` (`GPT`, `GPTConfig`).

| Component | Value |
|---|---|
| Layers | 28 |
| Attention heads / KV heads | 14 / 14 |
| Model dimension | 1792 |
| Head dimension | 128 |
| Vocabulary | 32,768 |
| Feed-forward | `relu²`, hidden multiplier 4 |
| Window pattern | `SSSL` (tiled across layers; final layer always `L`) |
| Short window | 512 |
| RoPE base | 100,000 |
| QK scale | 1.2 |
| Logit softcap | 15.0 |
| Weight tying | Untied |
| Norm | RMSNorm, no learnable parameters, applied after token embedding |
| Attention kernel | Flash Attention 3, SDPA fallback |
| Value-embed gating | 12 channels, `backout_layer_div` 2 |
| Position encoding | Rotary only |

**Parameter count** (verified by instantiating the real class on the `meta` device):

```
TOTAL params: 2,018,511,234
total (B): 2.019
embedding (M): 58.7
non-emb (M): 1959.8
```

**Why `SSSL` matters.** The window pattern is what makes 128 k context affordable on one
node: three of every four layers attend over a short window, and one layer per group
attends over the full context. This is a *serving-relevant* architectural choice — it
also required `--no-enable-chunked-prefill` in vLLM, which was measured to be necessary
for correct behaviour with this pattern.

**Tokenizer.** Custom byte-level BPE (`nanochat/tokenizer.py`), vocab 32,768, trained on
ClimbMix data via `scripts/tok_train.py`. The split pattern uses `\p{N}{1,2}` rather
than GPT-4's `\p{N}{1,3}` — deliberately, so numbers are not over-tokenized at a 32 K
vocab size. Special tokens:

```
<|bos|>, <|user_start|>, <|user_end|>, <|assistant_start|>, <|assistant_end|>,
<|python_start|>, <|python_end|>, <|output_start|>, <|output_end|>
```

The `python_*` tokens delimit a tool call; `output_*` delimit a tool result.

---

## 2. Dataset preparation

### 2.1 Pre-training corpus

| Property | Value |
|---|---|
| Source | `karpathy/climbmix-400b-shuffle` (NVIDIA Nemotron ClimbMix) |
| Nominal size | 400 B tokens, 6,543 shards available upstream |
| Downloaded | 171 shards, 15 GB on disk |
| Local path | `base_data_climbmix/` (symlinked from the cache dir) |
| Validation | The **last** shard is pinned as validation; all others are train |
| Packing | BOS-bestfit into fixed-length rows |

`nanochat/dataset.py` handles download-on-demand and iteration by parquet row-group for
throughput. The training loader (`nanochat/dataloader.py`) is
`tokenizing_distributed_data_loader_bos_bestfit` — tokenize, then pack with a
best-fit bin-packing so rows are dense and few documents straddle a row boundary.

### 2.2 SFT mixture construction

`scripts/chat_sft_long.py` assembles a `TaskMixture` from per-task modules in
`tasks/`, each a dataset of conversations plus optional evaluation criteria. Rows are
rendered with the **full** tokenizer conversation format, then packed BOS-bestfit into
32,768-token rows.

The v6j mixture:

| Task | Module | Epochs | Rows |
|---|---|---|---|
| Long context SFT | `tasks/longctx_sft.py` | 2 | levels 0–1, 40 instances/level |
| Tool calling | `tasks/toolcalling.py` | 1 | full |
| ToolACE | `tasks/toolace.py` | 1 | full |
| Tool follow-up | `tasks/toolfollowup.py` | 1 | full |
| Glaive multi-round | `tasks/glaive_multiround.py` | 2 | capped 20,000 |
| Tool stop | `tasks/toolstop.py` | 6 | 6,000 |
| Harness replay | `tasks/harness_replay.py` | 2 | 6,000 |
| Harness basics | `tasks/harness_basics.py` | 3 | 8,000 |
| Harness poison | `tasks/harness_poison.py` | 3 | 6,000 |
| SmolTalk | `tasks/smoltalk.py` | 1 | capped 20,000 |

Long-context families in use: `niah` (needle-in-a-haystack), `doc_qa` (structured report
QA), `multi_tool` (tool schemas + long document). `long_summary` and `multi_tool_weather`
exist in the code but are not in the v6j mixture.

The epoch counts are the story: the three `harness_*` tasks and `toolstop` carry 2–3× and
6× weight respectively, because they encode the *harness-specific* behaviour that
general-purpose tool-calling corpora do not teach.

### 2.3 What each harness task actually teaches

These three tasks are synthetic generators built to reproduce the `baby-onit` harness's
exact prompt shapes. Their docstrings are the specification.

**`tasks/harness_basics.py` — the fresh-session half (v5x).**

A census of the earlier (v5v) mixture showed the imbalance that caused the original
failure:

```
fresh + tools -> CALL  : ToolCalling 1,036 + ToolACE 8,957 = 9,993 rows
fresh + tools -> prose : 281 rows (144 with the harness's 8-tool set)
                          => 35:1 toward CALL
```

The model learned "tools present → call `web_search`" as its default: fresh `hello`
→ spurious `web_search` 10/10; "What is 2+2?" → `web_search`; "what files are in the
working directory?" → `web_search {"query":"finance"}` — wrong routing too, because no
row taught which tool serves which request.

`harness_basics` closes both gaps with five row shapes:

| Shape | Teaches |
|---|---|
| B1 social | `hello` / `how are you` / `thanks` → short greeting, **no call** |
| B2 easy-knowledge | `What is 2+2?` → direct answer, **no call** |
| B3 toolish → CALL | correct routing to the right tool (see table below) |
| B4 date-from-system | `date today` → `Today is {today}.` from the system prompt, never invented |
| B5 toolish-knowledge | `What time zone is Cairo in?` → prose from knowledge |

Routing map in B3:

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

Two robustness dimensions are randomized deliberately:

- **Toolsets:** the harness's real 8 (60%), random subsets (25%), the classic 3 (15%).
  The no-call boundary must hold for *any* toolset; and when the routed tool is absent
  from the row's toolset, the final turn falls back to prose — "call what exists, answer
  when nothing fits".
- **System prompts:** 50% the exact `baby-onit` system prompt (including its
  Credentials section), 50% harness-like/short variants. The model must not key on one
  wording.

Answers have 4–8 variants per shape, because the harness samples at temperature 0.6 —
single-canned-string rows pass a temperature-0 gate and still fail live.

35% of rows carry a replay variant (short technical history, 800–3,000 estimated tokens)
so the boundary also holds in the resumed-session shape.

**`tasks/toolstop.py` — when to stop calling (v5f).**

Live probes on the served v5e found the exact trigger of the agent-harness over-call:
the **bare chain** `[CALL] [result] [CALL] [result] → ?` — no prose, no user turn
between rounds. That shape is out of distribution: no v4-shape training row has
`[user "Tool result: …"]` followed by `[assistant CALL]`. The model chained 10/10 at
temperature 0, even when round 1 had already answered the question (5/5).

The data census explaining why no existing row covered it:

- `toolfollowup` (hermes): every `[user Tool result]` is followed by prose. **0** rows of
  result→CALL.
- `glaive_multiround`: calls only follow a real user question; results always followed
  by prose. **0** rows of result→CALL.
- `toolcalling`/`toolace` (Stage-4 shape): results ride *inside* the assistant turn as
  `python_output` parts, and result→CALL *is* supervised there (612 hermes + 650 toolace
  rows) — but in the old rendering, not the v4 serving shape.
- glaive prose-only rounds (tools present, no call): 1,650 of 33,484 rows (4.9%) —
  swamped by the call rows, hence the over-trigger on easy questions.

`toolstop` generates the four missing transitions in the exact serving shape:

| Shape | Sequence |
|---|---|
| A | `q → CALL → [result] → CALL → [result] → PROSE` (multi-part question; answer covers every round) |
| B | `q (single-part) → CALL → [result] → CALL (unprompted) → [result] → PROSE answering only the asked city` |
| C | `easy q + tools present → PROSE` directly (fixes the P5 over-trigger) |
| D | `q → CALL → [result] → PROSE` (reinforces the working P1/P3 behaviour) |

**`tasks/harness_poison.py` — act on the current turn (v6c).**

The harness replays session history verbatim on every turn. The user's real 284-turn
session from the broken v5-era model contains pairs like:

```
user: "weather today"  -> asst: "Today's date is December 7, 2026."
user: "hello"          -> asst: "I'm sorry for the inconvenience. ..."
```

The model (a) hallucinated a date on the first turn, (b) saved it to session history,
(c) then copy-pasted that date as the answer to *every* subsequent question. The tool
code never got a chance to run. Live repro against deployed v6a: HP gate 2/7.

Root cause: every training row shows history assistant answers as **competent**. No row
ever shows a broken assistant answer in history, so in-context pattern matching on the
model's own bad output beats routing. The naive fix — putting broken answers into rows
as-is — would **supervise** them, training the model to produce them.

This is where the `no_train` flag earns its keep. Row recipe (6,000 rows, seed 1111):

- Tools: harness8 60% / random subset 25% / classic 15%
- System: exact `baby-onit` 50% / harness 25% / short 15% / none 10%
- History blocks (strict alternation, every block starts user and ends assistant):
  - 0–6 clean technical pairs
  - 0–2 **working** tool rounds — history containing tool rounds must not itself be a
    poison signal
  - 2–6 poison pairs; placement 30% last (hardest — immediately before the final
    turn), 40% middle, 30% first
- Final turn (**supervised**): 30% redo-poison (a poisoned user turn asked *again* →
  the correct behaviour), 20% social → greeting, 20% toolish → CALL with correct
  routing, 10% date → from the system prompt, 10% easy → direct answer, 10%
  toolish-knowledge → prose.

### 2.4 The `no_train` mechanism

`nanochat/tokenizer.py` supports a per-message `no_train` flag. A message marked
`no_train` is rendered into the context with its loss mask set to **0** — visible,
never imitated:

```python
no_train = bool(message.get("no_train", False))
...
add_tokens(value_ids, 0 if no_train else 1)
```

Two places depend on it:

1. `tasks/harness_poison.py` — broken answers ride in history as context; only the
   correct final turn is a target.
2. `tasks/harness_basics.py` (the v6h fix) — the P7 anchor's spurious round-2 call
   rides as context only.

---

## 3. Training stages

### 3.1 Stage 1 — Pre-training

`scripts/base_train.py` → `d28b-v2`.

| Setting | Value |
|---|---|
| Steps | 40,000 |
| Context | 2,048 |
| Device batch / total batch | 4 / 32,768 |
| Embedding LR | 0.3 |
| Matrix LR | 0.02 |
| Weight decay | 0.28 |
| Target param:data ratio | 8.0 |
| Warmup / warmdown | 40 steps / 0.65 |
| Eval tokens | 41,943,040 |
| Save every | 5,000 |
| **Final val bpb** | **0.840180** |

### 3.2 Stage 2 — Context extension (2 k → 128 k)

`scripts/base_train_long.py` → `d28b-128k` (step 1750) → `d28b-128k-final` (step 250).

| Setting | Value |
|---|---|
| Context | 131,072 |
| Short window | 512 |
| RoPE base | 100,000 |
| Total batch | 262,144 tokens |
| Embedding / matrix LR | 0.09 / 0.006 |
| Warmdown ratio | 0.9 |
| Final LR fraction | 0.05 |
| **Final val bpb** | **0.893391** |

The bpb rises from 0.840 to 0.893 because the validation mixture changed — now evaluated
at 128 k on long packed rows, a strictly harder distribution. **The two numbers are not
comparable**; the same caveat recurs at every SFT stage below.

### 3.3 Stage 3 — Long-context SFT

`scripts/chat_sft_long.py`. This script is used both for the initial SFT and for all
Stage-7b touch-ups. It differs from `chat_sft.py` in rendering with the full tokenizer
conversation format and packing at the full `max_seq_len`.

### 3.4 Stage 4 — Math RL

`scripts/chat_rl.py` → `d28b-rl100` (step 177, checkpoint 100 also retained).

The upstream header calls this "GRPO in quotes", which is accurate: it is REINFORCE
with four deliberate simplifications.

1. **No trust region** — no KL regularization against a reference model.
2. **On-policy**, so no PPO probability ratio and no clipping.
3. **DAPO-style token-level normalization**, not sequence-level.
4. **Advantage is `(r − μ)`** — no z-score division by `σ`.

```python
parser.add_argument("--num-epochs", type=int, default=1)
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--examples-per-step", type=int, default=16)
parser.add_argument("--num-samples", type=int, default=16)
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--embedding-lr", type=float, default=0.2)   # Adam
parser.add_argument("--unembedding-lr", type=float, default=0.004)  # Adam
parser.add_argument("--matrix-lr", type=float, default=0.02)     # Muon
parser.add_argument("--init-lr-frac", type=float, default=0.05)
```

### 3.5 Stage 5 → 6 — Agentic RL

`scripts/chat_rl_agentic.py`. The optimization core is **byte-identical** to
`chat_rl.py`; only the reward source changes. `tasks/agentic.py` generates fresh
instances per index so the rollout set can never be memorized.

This was the fix for Stage 5's failure mode: train reward saturated at 1.0, advantage
went to zero, and gradients died at ~step 40. The fix was to make the training
distribution non-memorizable rather than to change the optimizer. Stage 6 also logs
reward broken out by family and difficulty level, and evaluates pass@k on held-out
instances of the same families.

### 3.6 Stage 7b — The touch-up ladder (current work)

Every v5x–v6j checkpoint is a 240-iteration SFT touch-up from `d28b-128k-sft-v6a@240`.
The recipe is fixed; only the data and specific fixes change between runs.

**v6j run configuration** (`run_v6j.sh`):

```
--run stage7b-v6j-literalnl
--source-phase sft --model-tag d28b-128k-sft-v6a --model-step 240
--output-tag d28b-128k-sft-v6j
--max-seq-len 32768 --short-window 512
--device-batch-size 1 --total-batch-size 262144
--num-iterations 240
--longctx-epochs 2 --levels 0,1 --instances-per-level 40
--longctx-families multi_tool,niah,doc_qa
--toolcall-epochs 1 --toolace-epochs 1 --followup-epochs 1
--glaive-mr-epochs 2 --toolstop-epochs 6
--harness-epochs 2 --harness-basics-epochs 3 --harness-poison-epochs 3
--smoltalk-epochs 1 --smoltalk-cap 20000
--embedding-lr 0.0036 --unembedding-lr 0.0001
--matrix-lr 0.00024 --scalar-lr 0.006
--weight-decay 0.028 --warmup-ratio 0.02 --warmdown-ratio 0.5 --final-lr-frac 0.0
--eval-every 40 --eval-tokens 524288 --save-every 60 --ce-chunk 4096
```

**Training curve** (GPU 7, world size 1, ~10 k tok/s, 26 s/step, 25.7 GB peak):

| Step | Train loss | Val bpb |
|---|---|---|
| 0 | — | 0.393432 |
| 40 | — | 0.397926 |
| 80 | — | 0.396480 |
| 115 | 0.3052 | — |
| 120 | 0.2919 | 0.397068 |
| 160 | 0.3280 | 0.394449 |
| 200 | — | 0.394106 |
| 240 | — | **0.393419** |

Total training time 6,211 s (~1 h 44 m).

### 3.7 The touch-up ladder — what each run fixed

| Run | Change | v5x | HP | Verdict |
|---|---|---|---|---|
| v6g | baseline | 12/15 | 6/7 | reference |
| v6h | anchor rows demoted to context-only (`no_train`) | 14/15 | 6/7 | kept |
| v6i | chased the last 2 fails | 15/15 | **regressed** | **rejected** |
| **v6j** | literal-`\n` training-data fix | **15/15** | **6/7** | **current** |

**v6i is the cautionary case.** It reached 15/15 on the real-harness gate but regressed
the poison gate, because the way it closed the HP5 gap taught the wrong association — a
long history makes *any* pool date acceptable, rather than "the header date is
authoritative". It was rejected and v6h was kept. A change that improves one gate and
degrades another is not an improvement.

**v6h — the spurious second call.** The v6g gates failed `P1`, `P13`, and `LOOP-T2` with
a hard `get_weather Toronto` re-call attractor. The census traced it to the
`harness_basics` P7-anchor rows: each taught `[get_weather result → ANOTHER get_weather
call for city2]` as a **supervised target** — the spurious round that the final prose
must ignore. The `P1` probe is exactly the anchor's mid-row state, so the model
completed the row with the anchor's second call. The v6h fix: the round-2 call rides as
context only, never a target.

**v6j — a two-character escape.** See §4.

---

## 4. Root cause analysis: the `\n` degeneration

This is the most instructive result in the project, because the symptom pointed at the
model, the server, and the client, and the actual cause was none of them.

### 4.1 The symptom

```
date today
Today's date is 2026-10-19, and
I'm\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n
\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n
...
 model nanochat-d28b-rl100  provider openai-compat  dir /home/rowel/baby-sandbox
 7,246 tok (3,150 in / 4,096 out)  124.1s
```

A one-line answer became 4,096 tokens of literal `\n` and took 124 s.

### 4.2 The cause

`tasks/harness_basics.py` used `"\\n\\n"` — an escaped **backslash** followed by `n` —
where a real newline was intended. The supervised *target* therefore contained the two
characters `\` and `n`.

The model faithfully learned to emit a literal backslash-n. At inference this becomes a
**period-2 byte-BPE cycle**: the tokenizer alternates two tokens, producing `\n\n\n…`
until the token cap. It is not a "same character repeated" degeneration, so the
original guard — which required 48+ *identical* characters — could not see it.

The full failure chain:

```
training target has literal "\n"
   → model emits literal backslash-n
   → period-2 byte-BPE cycle → 4,096 tokens of "\n\n\n…"
   → 120–124 s per answer
   → baby-onit SAVES it into session_history.jsonl
   → harness replays that history verbatim on every later turn
   → the poison is now an in-context attractor, re-teaching the behaviour forever
```

The history is what made it persistent. A 65 k-char degenerate answer is ~16 k tokens;
the harness's 8 k-token budget can only drop **whole messages** from the front, so it
cannot shed the poison — the oversized message rides along in full.

### 4.3 Why both fixes are needed

The obvious fix — retrain without the bad target — is necessary but **not sufficient**.
A poisoned history re-teaches the behaviour in context regardless of the weights. The
measurement, on identical prompts:

| Condition | Result |
|---|---|
| Raw v6j + poisoned history | **15.7 s of garbage** (504 literal `\n` in 512 tokens) |
| Raw v6j + **scrubbed** history | **0.6 s, clean** |
| v6j **via the guard** + poisoned history | **0.6 s, clean** |

Therefore: the retrain removes the **source**; the guard's history scrub stops an
existing poisoned history from **re-infecting** the model. Both ship.

### 4.4 The fix

- **Data:** `tasks/harness_basics.py` now uses a real newline.
- **Regression test:** `tasks/test_no_literal_newline.py` fails if any prose target ever
  contains a literal backslash-n again.
- **Guard:** the detector was rebuilt around *period-aware* detection. A short-period
  cycle (period 2–8, ≥ 8 reps, ≥ 32 chars) is now caught, which is exactly the class
  the old detector was blind to. Every threshold was calibrated against **20 real model
  answers** (18 good, 2 degenerate) so the numbers are measured rather than guessed.
- **History scrub:** the guard removes degenerate assistant turns from the replayed
  history before forwarding, replacing each with `[earlier answer truncated]`.
- **Session file:** the poisoned `session_history.jsonl` was sanitized in place
  (39.8 kB → 1.45 kB, 6 degenerate turns truncated, backup kept).

### 4.5 The detector, in full

| Signal | Threshold | Rationale |
|---|---|---|
| Identical chars in a row | 48+ | The `m`/whitespace flood (period-1) |
| Short-period cycle | period 2–8, ≥ 8 reps, ≥ 32 chars | The literal-`\n` loop (period-2) |
| Literal `\n` pairs, total | ≥ 4 | Clean separation: 0 good answers had 4+ |
| Literal `\n` pairs, consecutive | ≥ 3 | All 3 degenerate answers did |
| `##` heading density | ≥ 1.5 per 100 chars | Heading-spam loop with drifting text |
| Most-repeated segment | ≥ 3 | A list looping over a few entries |
| Total answer length | ≥ 2,000 chars | Runaway enumeration with unique lines |
| Trailing `m`/whitespace run | ≥ 200 chars | Whitespace flood |

The cycle detector only fires when the repeated block is itself *pathological*
(backslash, or only `m`/whitespace/dot) — so legitimate repeated text (bullet lists,
table rows, code) is not flagged.

### 4.6 The other failure shape: the 16-minute hang

Earlier in the same investigation, a "simple hello" hung for 16–21 minutes. That was a
different bug with the same smell:

- The serve script ran with `--max-num-seqs 1` — strictly one request at a time.
- The harness client sends `max_tokens: 32768` with no repetition penalty.
- One request degenerated and generated continuously at 34 tok/s → ~16 minutes of GPU.
- Every subsequent request, including `hello`, sat in the queue (`Waiting: 1`).

Fixes: `--max-num-seqs 1` → **8**; `--enable-prefix-caching` (a long replayed history
was re-prefilled every turn); `generation_config.json` `max_new_tokens: 1024` and
`repetition_penalty: 1.05`; and in the guard, `max_tokens` is clamped to 4096
regardless of what the client asks for.

**A note on a failed fix.** The first guard only detected 48+ identical characters, and
a missing `break` in the runaway-length branch set an `aborted` flag but never stopped
the stream. Both were found and fixed. The lesson, recorded because it generalizes: a
detector calibrated on one observed failure shape will miss the next one. The
period-aware rewrite and the 20-answer calibration were the response.

---

## 5. Serving

### 5.1 Topology

```
client (baby-onit, OpenAI-compatible)
   │  POST /v1/chat/completions
   ▼
:8000  guard_proxy.py          ← public endpoint
   │  clamp max_tokens, scrub history, always stream, abort degeneration
   ▼
:8010  vLLM                    ← nanochat-d28b-rl100-128k-v6j, GPU 0, bf16
```

**Why the inversion.** vLLM has no server-side completion cap that a client cannot
override, so the cap has to live in front of it. The public port must therefore belong
to the proxy.

### 5.2 vLLM flags and why each one is there

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
| `--served-model-name` | `nanochat-d28b-rl100` | The name clients ask for |
| `--enable-auto-tool-choice` | on | Native tool calling |
| `--tool-call-parser` | `nanochat` | Custom parser |
| `--tool-parser-plugin` | `nanochat_tool_parser.py` | Plugin registration |

### 5.3 Generation defaults

```json
{ "max_new_tokens": 1024, "repetition_penalty": 1.05 }
```

These are *defaults only*. A client sending an explicit `max_tokens` overrides
`max_new_tokens` — which is exactly the hole the guard plugs.

### 5.4 Guard proxy (`guard_proxy.py`, 807 lines)

Six responsibilities:

1. **Clamp `max_tokens`** to 4096 (`GUARD_MAX_TOKENS`), ignoring the client's value.
2. **Always stream upstream**, so the proxy has abort power for every client shape;
   non-streaming clients get the chunks reassembled into a full JSON response.
3. **Abort degenerate output mid-stream**, returning a clean OpenAI-compatible finish
   chunk. Degenerate `tool_calls` slots are dropped so the harness never executes
   garbage.
4. **Wall-clock cap** of 45 s (`GUARD_MAX_SECONDS`). Calibrated against the real model:
   the slowest legitimate turn observed was 8.8 s, so 45 s leaves 5× headroom while
   still bounding a runaway well before the user's "it hangs" threshold.
5. **Pass sampling parameters through untouched**, so the gates measure the model and
   not the proxy.
6. **Scrub degenerate assistant turns** from the replayed history (§4.4).

Aborts are logged to `guard_aborts.log`. Per-request detail is available via
`GUARD_TRACE=1`.

### 5.5 Chat template (`chat_template_v7.jinja`)

The tools block is placed on the **first** user turn. This was measured, not assumed.

The v6 template placed it on the *last* user turn, and it broke the live tool loop: when
the model is about to answer from a tool result, the last user turn **is** the
"Tool result:" turn — so the block landed immediately before the answer. That is a shape
zero training rows teach, and it caused `P1`, `P13`, and `LOOP-T2` failures. The v5
placement matches the `ToolStop` / `GlaiveMR` / `ToolFollowUp` rows exactly.

The template also:

- Renders the system message merged into the following user turn (matching the
  renderer's behaviour).
- Renders each tool **compactly** (`separators=(",", ":")`) and insertion-ordered —
  the exact Stage-4 training format. Spaced or sorted JSON is out of distribution and
  makes the model emit garbage arguments.
- Renders an OpenAI `tool` message as a **user** turn beginning `Tool result: …`, and
  merges consecutive tool messages into one turn (matching the bundled hermes results).
- Appends the instruction: `To call a tool, reply with a single line of the form CALL
  <name> <json-args>. Call exactly one tool per reply.`

### 5.6 Tool-call parser (`nanochat_tool_parser.py`, 236 lines)

The model emits calls in its Stage-4 SFT format inside python tags:

```
<|python_start|>CALL <name> <json-args><|python_end|>
```

The tokenizer's `skip_special_tokens=True` default drops the special tokens, so the
parser matches the text form `CALL <name> <json-args>`. Both spellings are accepted.

`supports_required_and_named = False` routes `tool_choice="required"` / named through
the auto path, because the model cannot be grammar-constrained into bare-JSON output.

### 5.7 Deployment hygiene

`serve_v6j_8000.sh` is restart-safe: it kills stale listeners on 8000/8010 first,
respawns the guard if it dies, and traps `EXIT` so vLLM never outlives the script.

---

## 6. Performance evaluation

### 6.1 Methodology

The earlier gates were **blind** — they used temperature 0 and 2 classic tools, while
the real harness uses temperature 0.6 and 8 real tools. The current gates import
`baby_onit` itself, so the system prompt, tool schemas, history replay, and request
shape are byte-identical to what the harness sends:

```python
import baby_onit as bo
cfg = {"serving": {"host": ..., "model": "nanochat-d28b-rl100",
       "history_budget_tokens": 8000, "max_context_tokens": 262144,
       "temperature": 0.6, "top_p": 0.95, "max_tokens": 32768,
       "max_chat_iterations": -1, "think": True}}
SYSMSG = bo.build_system_prompt(cfg)
TOOLS  = [s for s, _ in bo.TOOLS.values()]
REPLAY = bo.build_session_messages(bo.load_session_history(cfg), 8000)
```

### 6.2 Gate A — `verify_v5x.py`, the real-harness gate (15 checks)

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
| LOOP | 4-turn OpenAI loop | call → result → answer → new call |
| P7 | bare 2-round chain ×10 | **0** over-calls |
| P13 | question already answered by round 1 ×5 | **0** over-calls |
| P1 | 1-round answer ×5 | answers from the result |
| MT24 | `multi_tool` at ~24 k | `CALL get_weather {"city":"Oslo","unit":"Celsius"}` |

**Deploy only on 15/15.**

### 6.3 Gate B — `verify_hp.py`, the poison gate (7 checks)

The same probes, but with a deliberately poisoned history replayed: `hello`, `weather`,
`news`, `files`, `date`, `2+2`, and a poison-only weather case.

### 6.4 Results for v6j

Measured 2026-10-20 23:02–23:03, on the gate server (8016) and re-verified on the live
deployment (8000).

```
v5x gate: 15/15 PASS
  F1 true  F2 true  F3 true  F4 true  F5 true
  R1 true  R2 true  R3 true  R4 true  R5 true
  LOOP true  P7 true  P13 true  P1 true  MT24 true

HP gate: 6/7 PASS
  HP1 true  HP2 true  HP3 true  HP4 true
  HP5 false   <-- poisoned date (known, open)
  HP6 true  HP7 true
```

`HP5` failure output is always a training-pool date (`2026-10-12`, `2026-10-15`,
`2026-10-17`) rather than the header date. It is cosmetic, pre-dates this work, and
does not affect tool calling.

### 6.5 End-to-end checks against the real client

| Check | Result |
|---|---|
| User's exact sequence (`hello` / `who are you` / `date today` / `weather today`) ×10 | 40 turns, 0 slow, 0 degenerate, worst turn **21.4 s** |
| Real `baby-onit agent_loop` end-to-end ×6 | **48/48** turns clean |
| Poisoned replay, `date today` | **120.4 s → 0.6 s** |
| Fresh `hello` | **0.37 s** |
| 4 parallel `hello` | all 200 OK, ~0.4 s each |

### 6.6 Guard activity

`guard_aborts.log` records 116 aborts and 138 history scrubs in total. **All 116 aborts
are dated 2026-10-20** — the pre-v6j period, when the poisoned history was still being
replayed raw. There have been **zero aborts since the v6j deployment** (2026-10-20
22:52 onward); the only activity is history scrubs, which is the guard doing its job on
a pre-poisoned session file.

Abort breakdown:

| Type | Count |
|---|---|
| `degenerate-output` | 113 |
| `runaway-length` | 3 |

Post-deployment scrubs consistently report 6 degenerate history turns, matching the
6 degenerate turns that were sanitized out of `session_history.jsonl`.

### 6.7 Base model benchmark (CORE)

Pre-trained base model at step 10,080 (`base_eval/base_model_010080.csv`):

| Task | Accuracy | Centered |
|---|---|---|
| hellaswag_zeroshot | 0.370544 | 0.160725 |
| hellaswag | 0.364170 | 0.152227 |
| arc_easy | 0.555556 | 0.407407 |
| arc_challenge | 0.283276 | 0.044369 |
| copa | 0.560000 | 0.120000 |
| commonsense_qa | 0.223587 | 0.029484 |
| piqa | 0.674102 | 0.348204 |
| openbook_qa | 0.332000 | 0.109333 |
| lambada_openai | 0.292257 | 0.292257 |
| winograd | 0.593407 | 0.186813 |
| winogrande | 0.512234 | 0.024467 |
| bigbench_qa_wikidata | 0.233453 | 0.233453 |
| bigbench_cs_algorithms | 0.400758 | 0.400758 |
| bigbench_operators | 0.109524 | 0.109524 |
| bigbench_dyck_languages | 0.061000 | 0.061000 |
| agi_eval_lsat_ar | 0.278261 | 0.097826 |
| jeopardy | 0.013226 | 0.013226 |
| squad | 0.163292 | 0.163292 |
| coqa | 0.168734 | 0.168734 |
| boolq | 0.524465 | −0.251408 |
| bigbench_repeat_copy_logic | 0.000000 | 0.000000 |
| bigbench_language_identification | 0.259900 | 0.185809 |
| **CORE (centered)** | | **0.138977** |

This is the *base* model, before any SFT or RL. It is included for reference; the
chat/tool-calling behaviour is measured by the gates in §6.2–6.5, which are the metrics
that actually track deployment quality.

### 6.8 Reproducing

```bash
# 1. Train (GPU 7, ~2 h) — run script lives in the repo
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

---

## 7. Open items

| Item | Status | Note |
|---|---|---|
| `HP5` — date after long replayed history | **Open** | Cosmetic. Needs training rows built from *real* session text, not synthetic stand-ins. |
| Parallel tool calls in one turn | Not supported | By design; the model emits one call per turn. |
| `tool_choice="none"` leaking a `CALL` line as text | Pre-existing | Not a regression. |
| NIAH beyond ~105 k | Window limit | Hits the harness's 131,072-token limit. |
| Poisoned `session_history.jsonl` | Handled at serving time | The guard scrubs on every request; the file on disk was also sanitized. |

The unifying theme of the open items: the harness-facing training data is still
*synthetic*. Rows that mirror the real session text would close `HP5` and probably
tighten the remaining routing edge cases.

---

## 8. File map

| Path | Contents |
|---|---|
| `nanochat/gpt.py` | Model definition (`GPT`, `GPTConfig`) |
| `nanochat/tokenizer.py` | BPE tokenizer, conversation rendering, `no_train` |
| `nanochat/dataset.py` | ClimbMix download and iteration |
| `nanochat/dataloader.py` | BOS-bestfit packing loaders |
| `nanochat/engine.py` | Batch generation engine (RL, evaluation) |
| `nanochat/checkpoint_manager.py` | Save/load with optimizer state and metadata |
| `nanochat/optim.py` | Muon + AdamW optimizers |
| `scripts/base_train.py` | Stage 1 pre-training |
| `scripts/base_train_long.py` | Stage 2 context extension |
| `scripts/chat_sft_long.py` | Stages 3 and 6 — long-context SFT, touch-ups |
| `scripts/chat_rl.py` | Stage 4 — math RL |
| `scripts/chat_rl_agentic.py` | Stage 5 — agentic RL |
| `scripts/tok_train.py`, `scripts/tok_eval.py` | Tokenizer training and evaluation |
| `scripts/base_eval.py`, `scripts/chat_eval.py` | Evaluation harnesses |
| `tasks/*.py` | One dataset module per mixture component |
| `tasks/test_no_literal_newline.py` | Regression test for the v6j bug |
| `chat_template_v7.jinja` | Serving template (shipped in the checkpoint dir) |
| `run_v6*.sh` | One script per touch-up run, each documenting its own fix |
| `pack_v6*.sh` | Checkpoint → vLLM export |
| `serve_v6*.sh` | Deployment scripts |
| `EXECUTION_LOG.md` | Chronological log of every experiment, including failures |
| `MODEL_CARD.md` | Model card |
| `docs/TECHNICAL_REPORT.md` | This document |

---

## 9. Provenance

Built on the `nanochat` harness by Andrej Karpathy (`github.com/karpathy/nanochat`).
The checkpoint, the dataset mixture, the guard proxy, the tool parser, the chat
templates, and the evaluation gates are local work on top of that harness. The full
chronological record — including every rejected experiment — is in `EXECUTION_LOG.md`.
