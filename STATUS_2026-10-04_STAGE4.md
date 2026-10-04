STATUS — Stage 4 (Tool Calling) — 2026-10-04 23:15 PDT
======================================================

RUNNING on GPUs 3,4,5,7.  Stage 4 is "the one stage you must write"
(PLAN.md).  The writing was done in an earlier session; the work here was
scaling it from a corpus that turned out to be too small to learn from.

--------------------------------------------------------------------------
WHERE WE ARE IN THE LADDER
--------------------------------------------------------------------------
  0  Tokenizer ............... DONE
  1  Pretrain d12 (ablation) .. DONE  val_bpb 0.8781
  2  Pretrain d28 (headline) .. DONE  val_bpb 0.715437, 33.7 h, 9.10B tok
  3  SFT (d28) ............... DONE  val_bpb 0.308864, 15,028 steps, 0 NaN
  4  Tool calling ............ RUNNING  <-- you are here
  5  Reasoning RL ............ not started
  6  Long-horizon agentic .... not started

--------------------------------------------------------------------------
THE ACTUAL PROBLEM: 1,036 examples is the binding constraint
--------------------------------------------------------------------------
Stage 3 already trained tool-calling (toolcall_epochs=2), so Stage 4 could
easily have been called "done".  That would have been wrong.  Measured on the
Stage 3 d28 checkpoint, 40 held-out hermes examples:

    exact match (name+args) : 0/40   = 0.0%
    right function name      : 46.5% of 81 calls
    right number of calls    : 1/40  = 2.5%

Reading the failures, the model is picking the RIGHT tools and then failing to
copy the JSON arguments verbatim.  Example: ref asks for a devices list of 3
items, the model emits the right function name and the right argument values
but returns 1 of the 3 items.

hermes-v1 has 1,036 unique train examples.  More epochs of 1,036 cannot teach
verbatim copying; more DISTINCT examples can.  So the fix is data, not compute.

--------------------------------------------------------------------------
xLAM-60k is still inaccessible
--------------------------------------------------------------------------
PLAN.md names xLAM-60k.  From this node, even with HF_TOKEN set:

    Error: Access denied. This repository requires approval.

I nearly reported this as "downloaded" -- my first `hf download` printed a
snapshot path and I read it as success.  It had written 36 KB of metadata and
zero data.  Four repos were tried; only ToolACE is actually ungated and
actually downloaded (37 MB, verified on disk).

SUBSTITUTE: Team-ACE/ToolACE
    11,300 dialogs, 26,507 APIs, Apache-2.0, ungated
    dual-layer (rule + model) verified, arxiv.org/abs/2409.00920

After conversion: 8,957 train / 182 val  =  8.6x hermes.  2,161 records
dropped as unusable (no schema block / no parseable call).  Mixing both
corpora: hermes x2 + ToolACE x3 over an 818,702-row mixture.

--------------------------------------------------------------------------
MASKING: verified, and the verifier took four tries
--------------------------------------------------------------------------
PLAN.md: "apply loss only on the tool-call arguments, not the whole
sequence.  Masking correctly is most of the skill."  So it was verified on ALL
8,957 examples rather than sampled.  Four probe designs, all but the last
produced false alarms:

    v1 keyword probe ('parameters')      -> 51 false hits
    v2 schema-slice probe                ->  6 false hits
    v3 structural-keyword probe          -> 11 false hits
    v4 provenance probe                  ->   0      PASS

v1 matched calls whose own ARGUMENTS happen to be named "parameters".
v2 matched enum values and URL templates the model is SUPPOSED to copy into
its arguments.  v3 matched args legitimately shaped {"parameters":{}}.
v4 strips every CALL out of the supervised span and asserts the residue is
only special tokens -- so surviving schema prose is leakage by definition.

Two real non-bugs, now reported separately rather than as failures:
  - 19 examples truncated at the 2048-token cap (their later CALLs are cut)
  - <|assistant_end|> IS a supervised target by design (tokenizer.py:217)

Final: 0 non-CALL text supervised, 0 tool results supervised, 0 missing CALLs,
0 alternation violations.  Mean supervision 11.3%.

--------------------------------------------------------------------------
BUG FIXED: a latent one of mine, from earlier work
--------------------------------------------------------------------------
chat_sft.py's _rebuild_fullsize_optimizer_state formats `step` into a filename
but was passed args.model_step, which is None whenever --model-step is
omitted.  It only fires when the SFT world size differs from the pretrain world
size -- which is why Stage 3 never hit it (4-GPU pretrain, 4-GPU SFT) and why
Stage 4 hit it on the first try (4-GPU pretrain, 2-GPU smoke test).

        TypeError: unsupported format string passed to NoneType.__format__

Now resolves the step from the meta the loader actually used.

--------------------------------------------------------------------------
MEASURED CONFIG CHOICES
--------------------------------------------------------------------------
4 GPUs, device-batch-size 2, seq 2048 -> 4096 tok/rank/micro-batch.
total_batch_size 32768 inherited from pretrain, so grad_accum = 32768/(4096*4)
= 2, which divides evenly.  The batch is therefore IDENTICAL to Stage 3 v4's.
This is a pure throughput change, not a hyperparameter change.

Smoke-tested on 2 GPUs before committing the four (renders, trains, saves,
exits clean).

--------------------------------------------------------------------------
EVAL COST: I had it wrong by 68%
--------------------------------------------------------------------------
The first attempt used the default --eval-tokens 20971520, which at
eval_every=250 means 1280 eval micro-steps.  Measured: 263 s per eval against
0.489 s/step of training.  That made evals 68% of wall-clock -- TRUE 1.73
s/step vs 0.489 s/step train-only, a 3.5x understatement by dt.

Killed it at step 1,250 and relaunched with --eval-tokens 4194304 (256
micro-steps).  Verified afterwards: the train-time jump across an eval boundary
is now ~1 s, and 4.2M eval tokens is still ~128 passes over the 29.6K-row val
set.  Cost of the restart: 1,250 steps.  Benefit: ~4.5 h.

This is the same dt-vs-wallclock trap I hit twice before, now caught by
measuring the eval boundary directly instead of trusting the field.

--------------------------------------------------------------------------
ONE THING I DID NOT EXPLAIN: a 1.28x train slowdown
--------------------------------------------------------------------------
dt went 492 ms (first attempt) -> 629 ms (second), with --eval-tokens being
the only flag changed.  That flag feeds eval_steps alone (chat_sft.py:561) and
provably cannot change fwd/bwd cost, so the cause is external.  Most likely
CPU/dataloader contention from an unrelated job.

  PID 765926, /data/rowel/robot-decisions, train/train_lora.py, 718% CPU

That is a DIFFERENT project, not launched by me, so I left it alone.  All four
of my ranks sit at 99.3% CPU and 99-100% GPU with dt stable to 0.1 ms, so the
run is GPU-bound and not starved -- but 629 ms may be partly contention.  If
that job finishes and dt drops back to ~492 ms, the ETA below shortens.

--------------------------------------------------------------------------
CURRENT NUMBERS
--------------------------------------------------------------------------
  step            ~500 / ~15,700 implied  (3.2%; total is approx_progress, an
                  ESTIMATE -- Stage 3's meta recorded num_iterations=-1, so
                  there is no authoritative total)
  loss            ~1.51, flat so far
  train dt        629 ms  |  52.1k tok/s  |  30.7% BF16 MFU
  TRUE wall       ~0.63 s/step now that evals are cheap
  val_bpb         0.4330 @ 0  ->  0.4809 @ 250
                  (same early bump as Stage 3 v4, which peaked at 500 and
                  then fell monotonically to 0.3089 -- expected, not alarming)
  GPUs 3,4,5,7    99-100%, 34.7 GB each
  ETA             ~6.5 h, ~06:00 PDT 2026-10-05
  checkpoints     every 5000 steps; first at ~05:00 PDT

--------------------------------------------------------------------------
ACCEPTANCE TEST (what decides whether Stage 4 worked)
--------------------------------------------------------------------------
eval_toolcalling.py, held-out, greedy.  Pre-change baseline on the Stage 3
checkpoint is recorded above; the run is only worth anything if exact match
moves off 0.0%.

SOURCES
  /data/rowel/serve_logs/stage4/stage4.log
  /data/rowel/serve_logs/stage4/cost_model.py, direct_cost.py
  nanochat commit dccaa9a (tasks/toolace.py, verify_toolace_mask.py)
  data/toolace/data.json  (Team-ACE/ToolACE, Apache-2.0)

OPEN ITEMS CARRIED FORWARD
  - nanochat still has NO WRITABLE REMOTE (origin is karpathy/nanochat.git).
    Every Stage 3 and Stage 4 fix is local-only and unbacked-up.  This is now
    urgent: Stage 4's run depends on a fix that exists nowhere else.
  - commit f7a3352 in reasoner-model still unexplained.
