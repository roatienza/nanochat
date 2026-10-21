"""
Test the prefetching dataloader.

The pretraining loop asks the loader for the next batch right after
backward(), so with grad_accum_steps == 1 the GPU sits idle while the CPU
tokenizes. PrefetchLoader runs the loader in a background thread and keeps
one batch ready, which measured out at cutting the loader stall from
6.11 ms to 0.03 ms against a 445 ms step.

The risk with any prefetch is CORRECTNESS: the underlying loader yields
VIEWS into one persistent device buffer that it overwrites on the next
iteration, so a producer that runs ahead would clobber a batch that is
still in flight. PrefetchLoader clones each batch to prevent that.

These tests check:
  1. the prefetched stream is identical to the un-prefetched stream,
  2. successive batches do not alias,
  3. the prefetch actually hides the loader's latency,
  4. close() shuts the worker down cleanly.

Run: python -m pytest tests/test_prefetch.py -v -s
"""
import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.dataloader import (  # noqa: E402
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
    PrefetchLoader,
)

B, T = 4, 2048


def _tokenizer():
    from nanochat.tokenizer import get_tokenizer
    return get_tokenizer()


def _loader(tok, device):
    return tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, B, T, split="train", device=device)


@pytest.fixture(scope="module")
def tok():
    return _tokenizer()


def test_prefetch_stream_is_identical(tok):
    """The prefetched stream must equal the un-prefetched stream, exactly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = _loader(tok, device)
    plain = []
    for _ in range(8):
        x, y, s = next(raw)
        plain.append((x.clone(), y.clone(), s))

    pre = PrefetchLoader(_loader(tok, device))
    for i in range(8):
        x, y, s = next(pre)
        xa, ya, sa = plain[i]
        assert torch.equal(xa, x), "inputs differ at batch %d" % i
        assert torch.equal(ya, y), "targets differ at batch %d" % i
        assert sa == s, "state differs at batch %d" % i
    pre.close()


def test_prefetch_does_not_alias(tok):
    """Successive batches must not share storage.

    The underlying loader reuses ONE device buffer. If the prefetch handed
    out views into it, the arrival of batch N+1 would overwrite batch N
    while the GPU was still computing on it.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre = PrefetchLoader(_loader(tok, device))
    x1, y1, _ = next(pre)
    snapshot = x1.clone()
    x2, y2, _ = next(pre)

    assert torch.equal(snapshot, x1), (
        "x1 was mutated after x2 arrived -- the prefetch is aliasing")
    assert x1.data_ptr() != x2.data_ptr(), (
        "x1 and x2 share storage -- the clone is missing")
    pre.close()


def test_prefetch_hides_the_loader_latency(tok):
    """With a slow consumer, the wait for the next batch should vanish."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(use_prefetch, n=10):
        loader = PrefetchLoader(_loader(tok, device)) if use_prefetch else _loader(tok, device)
        waits = []
        x, y, s = next(loader)
        for _ in range(n):
            if device == "cuda":
                # stand-in for a training step: keep the GPU busy
                a = torch.randn(1024, 1024, device=device)
                for _ in range(8):
                    a = a @ a.T / 1024.0
                torch.cuda.synchronize()
            else:
                time.sleep(0.05)
            t0 = time.time()
            x, y, s = next(loader)
            waits.append((time.time() - t0) * 1000)
        if use_prefetch:
            loader.close()
        return waits

    raw = run(False)
    pre = run(True)
    assert sum(pre) < sum(raw), (
        "prefetch did not reduce the wait: raw %.2f ms vs prefetch %.2f ms"
        % (sum(raw) / len(raw), sum(pre) / len(pre)))


def test_prefetch_close_is_clean(tok):
    """close() must stop the worker without aborting the process."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre = PrefetchLoader(_loader(tok, device))
    next(pre)
    pre.close()
    # A second close must not raise.
    pre.close()
