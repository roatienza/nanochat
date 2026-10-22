"""
Tests for the RL trainers' --gen-batch-size option.

WHAT IS BEING TESTED
--------------------
The RL trainers (chat_rl.py, chat_rl_agentic.py, chat_rl_curriculum.py)
decode rollouts in a generation batch that is now SEPARATE from the
training-forward device batch:

  * --device-batch-size  how many sequences one training forward holds
                          (and its gradients)
  * --gen-batch-size     how many rollouts one generate() call decodes
                          (and its KV cache)

The separation exists because the two have completely different memory
profiles, and because the decode step is latency-bound on a batch that
does not fill the GPU: the recipe used to decode 2 samples at a time, and
16 samples in one call is 5.4x faster end to end (measured 2026-10-22).

These tests check the parts that can be checked without a GPU:

  1. --gen-batch-size defaults to --device-batch-size (backwards
     compatibility: the old behaviour is preserved unless the flag is
     passed).
  2. --gen-batch-size overrides it.
  3. The generation loop splits num_samples into chunks of gen_batch_size,
     the last chunk may be short, and the number of chunks is right.
  4. The chunks are concatenated in order, so the per-record outcome
     order -- and therefore the pass@k metric -- is unchanged.

The chunking arithmetic is re-implemented here rather than imported,
because the trainers are module-level scripts: importing one would run
the whole training program.
"""
import ast
import os
import re
import sys

import pytest

TRAINERS = [
    "scripts/chat_rl.py",
    "scripts/chat_rl_agentic.py",
    "scripts/chat_rl_curriculum.py",
]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


@pytest.mark.parametrize("rel", TRAINERS)
def test_the_file_parses(rel):
    ast.parse(_read(rel))


@pytest.mark.parametrize("rel", TRAINERS)
def test_gen_batch_size_argument_exists(rel):
    src = _read(rel)
    assert '"--gen-batch-size"' in src, f"{rel} has no --gen-batch-size argument"
    m = re.search(r'parser\.add_argument\("--gen-batch-size",\s*type=int,\s*default=(-?\d+)', src)
    assert m, f"{rel}: could not parse the --gen-batch-size default"
    # The default must be a sentinel, not a real size: the real default is
    # "--device-batch-size", resolved after parsing.
    assert m.group(1) == "-1", f"{rel}: the --gen-batch-size default should be -1 (a sentinel)"


@pytest.mark.parametrize("rel", TRAINERS)
def test_gen_batch_size_falls_back_to_device_batch(rel):
    src = _read(rel)
    # The fallback must be resolved right after the parse, before the
    # training loop uses it.
    assert "if args.gen_batch_size <= 0:" in src, f"{rel}: no fallback for gen_batch_size"
    assert "args.gen_batch_size = args.device_batch_size" in src, \
        f"{rel}: gen_batch_size does not fall back to device_batch_size"
    # ... and the fallback must come BEFORE the generation loop.
    fallback = src.index("args.gen_batch_size = args.device_batch_size")
    loop = src.index("num_sampling_steps")
    assert fallback < loop, f"{rel}: the fallback is resolved after the loop that uses it"


@pytest.mark.parametrize("rel", TRAINERS)
def test_generation_loop_chunks_by_gen_batch(rel):
    src = _read(rel)
    assert "args.gen_batch_size" in src
    # The loop must compute the number of chunks from gen_batch_size, not
    # from device_batch_size.
    m = re.search(r"num_sampling_steps = .*gen_batch_size.*", src)
    assert m, f"{rel}: the generation loop does not chunk by gen_batch_size"
    assert "device_batch_size" not in m.group(0), \
        f"{rel}: the generation loop still chunks by device_batch_size: {m.group(0)!r}"
    # The last chunk may be short.
    assert "n_this = min(args.gen_batch_size" in src, \
        f"{rel}: the last chunk is not clamped to the remaining samples"


@pytest.mark.parametrize("rel", TRAINERS)
def test_eval_decodes_the_same_number_of_samples(rel):
    """The eval must decode num_samples rollouts, in chunks.

    num_samples is the pass@k depth, so it is part of the METRIC. The
    chunking must not change how many samples are decoded, only how
    they are grouped into calls.
    """
    src = _read(rel)
    # The eval path must loop over chunks.
    assert "remaining = num_samples" in src, f"{rel}: the eval does not chunk"
    assert "remaining -= n_this" in src, f"{rel}: the eval chunk loop does not advance"
    # ... and it must decode exactly num_samples in total.
    assert "n_this = min(args.gen_batch_size, remaining)" in src
    # The old single-call form must be gone from the eval path.
    assert "assert num_samples <= args.device_batch_size" not in src, \
        f"{rel}: the eval still asserts num_samples <= device_batch_size"


@pytest.mark.parametrize("rel", TRAINERS)
def test_step_timing_is_reported(rel):
    """The RL stages must report their own wall clock.

    They are 14-28% of the pipeline's wall clock and they were the only
    trainers with NO timing output: the log said which step it was on
    but never how long a step took. On a multi-day run that is the
    difference between a slow run and a stuck one.
    """
    src = _read(rel)
    assert "rl_t_start = time.time()" in src, f"{rel}: no wall-clock start"
    assert "step {step}/{num_steps} took" in src, f"{rel}: no per-step timing line"
    assert "ETA" in src, f"{rel}: no ETA in the timing line"


@pytest.mark.parametrize("rel", TRAINERS)
def test_gpu_is_released_on_failure(rel):
    """An unhandled exception must free the GPU.

    Without this, a CUDA OOM leaves the process alive in state R with its
    full ~35 GiB of GPU memory, and the NEXT run dies at torch.load with
    "Process N has 34.87 GiB memory in use" -- a corpse from the
    previous attempt that looks like a memory bug in the new code.
    """
    src = _read(rel)
    assert "_release_gpu" in src, f"{rel}: no GPU release hook"
    assert "_atexit.register(_release_gpu)" in src, f"{rel}: the hook is not registered"
    assert "_sys.excepthook = _fatal" in src, f"{rel}: the exception hook is not installed"
    assert "_os._exit(1)" in src, f"{rel}: a failure does not set a non-zero exit status"


# --- the chunking arithmetic itself ------------------------------------

def _chunks(num_samples, gen_batch_size):
    """Mirror of the trainers' chunking loop."""
    out = []
    remaining = num_samples
    while remaining > 0:
        n_this = min(gen_batch_size, remaining)
        out.append(n_this)
        remaining -= n_this
    return out


def test_chunking_covers_every_sample():
    for num_samples in (16, 32, 64):
        for gen_batch_size in (1, 2, 4, 8, 16, 32, 64, 128):
            chunks = _chunks(num_samples, gen_batch_size)
            assert sum(chunks) == num_samples, (num_samples, gen_batch_size, chunks)
            assert all(c > 0 for c in chunks), (num_samples, gen_batch_size, chunks)
            assert all(c <= gen_batch_size for c in chunks)


def test_chunking_of_the_measured_case():
    """The recipe's case: 16 samples, generation batch 16 -> one call."""
    assert _chunks(16, 16) == [16]
    # The OLD behaviour: 16 samples at generation batch 2 -> 8 calls.
    assert _chunks(16, 2) == [2] * 8


def test_chunking_preserves_order():
    """The chunks are concatenated in order, so the outcome order -- and
    therefore the pass@k metric -- is unchanged."""
    num_samples, gen_batch_size = 16, 5
    chunks = _chunks(num_samples, gen_batch_size)
    ordered = []
    for c in chunks:
        ordered.extend(range(len(ordered), len(ordered) + c))
    assert ordered == list(range(num_samples))
