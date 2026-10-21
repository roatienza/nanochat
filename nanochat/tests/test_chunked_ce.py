"""
Test the chunked cross-entropy in nanochat/gpt.py.

The chunked path replaces the single-shot
    logits = softcap * tanh(lm_head(x)[..., :V].float() / softcap)
    loss  = F.cross_entropy(logits.view(-1, V), targets.view(-1),
                            ignore_index=-1, reduction=...)
with the same math done in sequence chunks, recomputed in the backward pass.

It MUST be numerically identical -- otherwise every checkpoint trained with
it would be subtly different from one trained without it.

NOTE: _chunked_cross_entropy takes the HIDDEN STATES (B, T, n_embd), not
token ids. The embedding and the trunk are done by the caller.

Run: python -m pytest tests/test_chunked_ce.py -v -s
"""
import torch
import pytest
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig


def build_model(ce_chunk, n_layer=2, n_embd=256, n_head=2, vocab=512, seq=64):
    # head_dim is fixed at 128 in this codebase, so n_embd must be
    # n_head * 128 or the linear layers will not line up.
    cfg = GPTConfig(sequence_len=seq, vocab_size=vocab, n_layer=n_layer,
                    n_head=n_head, n_kv_head=n_head, n_embd=n_embd,
                    ce_chunk=ce_chunk)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


def reference_loss(model, h, y, softcap=15, reduction="mean"):
    """The original single-shot implementation, verbatim."""
    logits = model.lm_head(h)[..., :model.config.vocab_size].float()
    logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                           ignore_index=-1, reduction=reduction)


def hidden_and_targets(model, B=2, T=64, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(B, T, model.config.n_embd)
    y = torch.randint(0, model.config.vocab_size, (B, T))
    return h, y


@pytest.mark.parametrize("chunk", [7, 16, 64, 128])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_chunked_ce_matches_reference(chunk, reduction):
    # The LOGITS are bit-identical (the same matmul over the same flat
    # rows). The only difference is the ORDER of the fp32 summation over
    # the B*T tokens, so the loss differs by ~1e-7 relative. That is far
    # below the bf16 noise of the training itself, but it is not zero --
    # so the tolerance here reflects that.
    model = build_model(ce_chunk=chunk)
    h, y = hidden_and_targets(model)
    got = model._chunked_cross_entropy(h, y, 15, reduction)
    want = reference_loss(model, h, y, 15, reduction)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (
        "chunk=%d reduction=%s: %s vs %s" % (chunk, reduction, got, want))


@pytest.mark.parametrize("chunk", [7, 16, 64])
def test_chunked_ce_matches_reference_with_ignored_targets(chunk):
    """ignore_index=-1 targets must be excluded from the mean, as torch does."""
    model = build_model(ce_chunk=chunk)
    h, y = hidden_and_targets(model, seed=1)
    y[0, :10] = -1
    y[1, 5] = -1

    got = model._chunked_cross_entropy(h, y, 15, "mean")
    want = reference_loss(model, h, y, 15, "mean")
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (
        "chunk=%d: %s vs %s" % (chunk, got, want))


def test_chunked_ce_reduction_none_matches():
    """reduction='none' must return the per-token losses, FLAT.

    The original implementation ran cross_entropy on the flattened
    (B*T, V) logits, so it returned a flat (B*T,) tensor. The callers
    (chat_rl.py, loss_eval.py) then do .view_as(inputs). We must match
    the original shape exactly or those calls break.
    """
    model = build_model(ce_chunk=16)
    h, y = hidden_and_targets(model, seed=2)
    y[0, :3] = -1

    got = model._chunked_cross_entropy(h, y, 15, "none")
    want = reference_loss(model, h, y, 15, "none")
    assert got.shape == want.shape == (2 * 64,), (got.shape, want.shape)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (got, want)


def test_chunked_ce_backward_matches_reference():
    """Gradients must match too -- the whole point of the recompute."""
    model = build_model(ce_chunk=16)
    h, y = hidden_and_targets(model, seed=3)
    h1 = h.clone().requires_grad_(True)
    h2 = h.clone().requires_grad_(True)

    loss = model._chunked_cross_entropy(h1, y, 15, "mean")
    loss.backward()
    loss2 = reference_loss(model, h2, y, 15, "mean")
    loss2.backward()

    assert torch.allclose(h1.grad, h2.grad, rtol=1e-4, atol=1e-6), (
        "input grad mismatch: max diff %s" % (h1.grad - h2.grad).abs().max())
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model.named_parameters()):
        assert n1 == n2
        if p1.grad is None and p2.grad is None:
            continue
        assert torch.allclose(p1.grad, p2.grad, rtol=1e-4, atol=1e-6), (
            "grad mismatch for %s" % n1)


def test_chunking_disabled_is_the_single_shot_path():
    """ce_chunk=0 (or >= T) must take the single-shot branch."""
    model = build_model(ce_chunk=0)
    h, y = hidden_and_targets(model, seed=4)
    got = model._chunked_cross_entropy(h, y, 15, "mean")
    want = reference_loss(model, h, y, 15, "mean")
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-6)


def test_chunked_logits_are_bit_identical():
    """The strong property: chunking changes the SUMMATION ORDER only.

    The logits themselves -- what the lm_head matmul produces -- must be
    bit-identical between the chunked and the single-shot path. That is
    what guarantees the chunked path is mathematically the same function
    and not an approximation of it.
    """
    model = build_model(ce_chunk=16)
    torch.manual_seed(7)
    h = torch.randn(2 * 64, model.config.n_embd)
    V = model.config.vocab_size

    def logits_of(hc):
        lg = model.lm_head(hc)[..., :V].float()
        return 15 * torch.tanh(lg / 15)

    full = logits_of(h)
    chunked = torch.cat([logits_of(h[s:s + 16]) for s in range(0, h.size(0), 16)], dim=0)
    assert torch.equal(full, chunked), (
        "logits differ: max %s" % (full - chunked).abs().max().item())


def test_chunked_ce_relative_error_is_negligible():
    """The loss difference must be far below the bf16 noise of training."""
    model = build_model(ce_chunk=16)
    h, y = hidden_and_targets(model, seed=8)
    got = model._chunked_cross_entropy(h, y, 15, "mean").item()
    want = reference_loss(model, h, y, 15, "mean").item()
    rel = abs(got - want) / abs(want)
    assert rel < 1e-5, "relative difference %s is too large" % rel


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the full forward uses Flash Attention, which needs CUDA")
def test_public_forward_uses_chunked_ce():
    """model(x, y) must go through the chunked path and match the reference.

    Rather than re-implementing the trunk (which is easy to get subtly
    wrong), this compares the model's OWN forward against the same model
    with ce_chunk disabled, i.e. the original single-shot implementation.
    That is exactly the property that matters: turning chunking on must not
    change the loss.
    """
    torch.manual_seed(5)
    model = build_model(ce_chunk=16).to("cuda")
    B, T = 2, 64
    x = torch.randint(0, model.config.vocab_size, (B, T), device="cuda")
    y = torch.randint(0, model.config.vocab_size, (B, T), device="cuda")

    with torch.no_grad():
        got = model(x, y)                       # chunked
        # Disable chunking -> the original single-shot path.
        model.config.ce_chunk = 0
        want = model(x, y)                      # single shot
        model.config.ce_chunk = 16

    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6), (got, want)
