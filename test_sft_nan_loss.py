"""
Regression test for the Stage 3 NaN-loss bug (chat_sft.py micro-batch loss logging).

What went wrong: the training step loop did

    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach()      # overwritten every micro-batch
        loss = loss / grad_accum_steps
        loss.backward()

and logged `train_loss.item()` after the loop. Two consequences:

1. Only the LAST micro-batch was ever logged, so the reported loss was one
   micro-batch's mean-per-valid-token, not a batch average.
2. When a micro-batch is entirely padding, every target is ignore_index=-1 and
   F.cross_entropy(..., reduction='mean') divides by a zero valid-token count:
   loss = nan. That nan went straight into the sticky EMA
   (smooth = 0.9*smooth + 0.1*loss), which never recovers. The run then reported
   "loss: nan" for thousands of steps while every weight, every optimizer moment,
   and the validation bpb were demonstrably healthy -- a phantom dead run.

This test asserts the two properties the fix must preserve:
  * an all-padding micro-batch produces nan loss but an all-zero (finite) gradient,
    so it is harmless to the weights and need only be excluded from *logging*;
  * the new averaging excludes non-finite micro-batch losses and raises when every
    micro-batch is degenerate, rather than silently poisoning the EMA.
"""
import torch
import torch.nn.functional as F

VOCAB = 32
SEQ = 8


def make_batch(rows, pad_rows=0):
    """Build (x, y) where `pad_rows` trailing rows are entirely masked padding."""
    total = rows + pad_rows
    x = torch.randint(0, VOCAB, (total, SEQ))
    y = torch.randint(0, VOCAB, (total, SEQ))
    for r in range(rows, total):
        y[r] = -1  # ignore_index: a fully-padded conversation
    return x, y


def loss_for(model, x, y):
    logits = model(x)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction="mean"
    )


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(VOCAB, 16)
        self.head = torch.nn.Linear(16, VOCAB)

    def forward(self, x):
        return self.head(self.emb(x))


def average_finite(losses):
    """The fixed logging rule, mirroring chat_sft.py."""
    finite = [v.item() for v in losses if torch.isfinite(v)]
    if not finite:
        raise RuntimeError("every micro-batch had no supervised tokens")
    return sum(finite) / len(finite)


def main():
    torch.manual_seed(0)
    model = Tiny()

    # --- 1. the degenerate micro-batch: nan loss, but a harmless gradient -------
    x_pad, y_pad = make_batch(rows=0, pad_rows=2)
    loss_pad = loss_for(model, x_pad, y_pad)
    assert loss_pad.isnan(), f"expected nan from an all-padding micro-batch, got {loss_pad}"
    model.zero_grad(set_to_none=True)
    loss_pad.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected gradients to exist"
    for g in grads:
        assert torch.isfinite(g).all(), "gradient from a padded micro-batch must be finite"
        assert g.abs().sum() == 0, "gradient from a padded micro-batch must be exactly zero"
    print("PASS all-padding micro-batch -> loss=nan, grad=all-zero & finite (weights safe)")

    # --- 2. a normal micro-batch still produces a usable loss ------------------
    x_real, y_real = make_batch(rows=2, pad_rows=0)
    loss_real = loss_for(model, x_real, y_real)
    assert torch.isfinite(loss_real), "a supervised micro-batch must give a finite loss"
    print(f"PASS supervised micro-batch -> finite loss {loss_real.item():.4f}")

    # --- 3. the fix: one degenerate micro-batch must not poison the average ----
    losses = [loss_real.detach(), loss_pad.detach()]
    avg = average_finite(losses)
    assert torch.isfinite(torch.tensor(avg)), "average must stay finite despite a nan micro-batch"
    assert abs(avg - loss_real.item()) < 1e-6, "the nan micro-batch must be excluded entirely"
    print(f"PASS mixed micro-batches -> average {avg:.4f} (nan excluded, not zeroed)")

    # --- 4. old behaviour, for contrast: averaging the last value only ----------
    old_logged = losses[-1].item()  # train_loss is overwritten each iteration
    assert torch.isnan(torch.tensor(old_logged)), "the old code logged exactly the nan"
    print("CONFIRM old code logged the nan (this is the bug that was fixed)")

    # --- 5. all micro-batches degenerate -> loud failure, not silent corruption --
    try:
        average_finite([loss_pad.detach(), loss_pad.detach()])
    except RuntimeError as e:
        print(f"PASS all-degenerate batch raises RuntimeError: {e}")
    else:
        raise AssertionError("expected RuntimeError when every micro-batch is degenerate")

    print("\nAll SFT NaN-loss regression checks passed.")


if __name__ == "__main__":
    main()
