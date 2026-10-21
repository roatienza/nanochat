"""
Stage 7a - long-context continued pretraining at sequence_len=131072.

Extends d28b-rl100 (trained at 2048) to 128k context WITHOUT YaRN: the model
is actually trained at the long sequence length. Design (all measured on
A100-40GB, see dev/mem_probe_128k_v2.py, dev/throughput_probe_128k.py,
dev/find_leak_128k.py):

- Per-block activation checkpointing with the resid/x0 lambda multiply folded
  INSIDE the checkpoint and value embeddings recomputed from idx inside.
- MLP computed in 16384-token sequence chunks (relu^2 intermediate at 131k
  x 7168 would be 1.9 GB per tensor otherwise). Exact per-token math.
- Chunked cross-entropy (4096-token chunks), each chunk's lm_head+softcap+CE
  wrapped in a checkpoint so logits are recomputed during backward.
- S sliding-window layers keep the TRAINED window (512); L layers cover the
  full 131072 (GPTConfig.short_window knob). Rope base unchanged (100000).
- torch.optim.AdamW for all params (the custom MuonAdamW leaks ~2-3 GB/step
  under torchrun at this scale); optimizer states CPU-offloaded between steps
  with plain .to() swaps (measured leak-free).
- Measured: backward peak 32.6 GB/GPU, ~6800 tok/s/GPU, stable across steps.

Run (2 GPUs):
  cd /data/rowel/nanochat && NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2 \
    CUDA_VISIBLE_DEVICES=3,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --standalone --nproc_per_node=2 -m scripts.base_train_long \
    -- --run=stage7-longctx-128k --num-iterations=2000
"""
import argparse, os, sys, math, time
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit, tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_tokenizer, get_token_bytes

parser = argparse.ArgumentParser(description="Stage 7a: long-context continued pretraining (131072)")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--model-tag", type=str, default="d28b-rl100")
parser.add_argument("--source-phase", type=str, default="rl", choices=["base", "sft", "rl"])
parser.add_argument("--model-step", type=int, default=177)
parser.add_argument("--output-tag", type=str, default="d28b-128k")
parser.add_argument("--max-seq-len", type=int, default=131072)
parser.add_argument("--short-window", type=int, default=512)
parser.add_argument("--device-batch-size", type=int, default=1)
parser.add_argument("--total-batch-size", type=int, default=262144)
parser.add_argument("--num-iterations", type=int, default=2000)
parser.add_argument("--embedding-lr", type=float, default=0.09)
parser.add_argument("--unembedding-lr", type=float, default=0.0024)
parser.add_argument("--matrix-lr", type=float, default=0.006)
parser.add_argument("--scalar-lr", type=float, default=0.15)
parser.add_argument("--weight-decay", type=float, default=0.028)
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--eval-tokens", type=int, default=2*131072)
parser.add_argument("--save-every", type=int, default=250)
parser.add_argument("--ce-chunk", type=int, default=4096)
args = parser.parse_args()

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
use_dummy_wandb = args.run == "dummy" or not master_process
if use_dummy_wandb:
    wandb_run = DummyWandb()
else:
    import wandb
    wandb_run = wandb.init(project="nanochat-longctx", name=args.run, config=vars(args))

# ---------------------------------------------------------------------------
# Load source checkpoint and rebuild the model at the LONG sequence length.
base_dir = get_base_dir()
src_dir = os.path.join(base_dir, {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[args.source_phase], args.model_tag)
model_data, optimizer_data, meta_data = load_checkpoint(src_dir, args.model_step, device, load_optimizer=False)
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
mc = dict(meta_data["model_config"])
mc["sequence_len"] = args.max_seq_len
mc["short_window"] = args.short_window
mc.setdefault("rope_base", 100000)
print0(f"Building long-context model: {mc}")
model_config = GPTConfig(**mc)
with torch.device("meta"):
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
model.load_state_dict(model_data, strict=True, assign=True)
model.train()
del model_data
tokenizer = get_tokenizer()
assert tokenizer.get_vocab_size() == mc["vocab_size"]
token_bytes = get_token_bytes(device=device)

# ---------------------------------------------------------------------------
# Long-context forward: per-block checkpointing (lambdas + ve inside), MLP in
# sequence chunks, chunked CE with per-chunk checkpointing.
x0_holder = [None]
CE_CHUNK = args.ce_chunk

def _group_forward(model, idx, cos_sin, x_in, lo, hi):
    x = x_in
    for i in range(lo, hi):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0_holder[0]
        ws = model.window_sizes[i]
        ve = model.value_embeds[str(i)](idx).to(x.dtype) if str(i) in model.value_embeds else None
        blk = model.transformer.h[i]
        x = x + blk.attn(norm(x), ve, cos_sin, ws, None)
        xn = norm(x)
        out = torch.empty_like(xn)
        M = 16384
        T = xn.size(1)
        for s in range(0, T, M):
            e = min(s + M, T)
            h = blk.mlp.c_fc(xn[:, s:e])
            h = F.relu(h).square()
            out[:, s:e] = blk.mlp.c_proj(h)
        x = x + out
    return x

def longctx_forward_loss(model, idx, targets):
    B, T = idx.size()
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = model.transformer.wte(idx).to(COMPUTE_DTYPE)
    x = norm(x)
    gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    x0_holder[0] = x
    x_backout = None
    n_layer = model.config.n_layer
    for lo in range(0, n_layer):
        x = cp.checkpoint(_group_forward, model, idx, cos_sin, x, lo, lo + 1, use_reentrant=False)
        if lo == n_layer // 2:
            x_backout = x
    x = x - model.backout_lambda.to(x.dtype) * x_backout
    h = norm(x)

    def chunk_loss(hc, yc):
        logits = model.lm_head(hc)[..., :model.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), yc.reshape(-1),
                               ignore_index=-1, reduction="sum")

    total = 0.0
    ntok = (targets >= 0).sum()
    for s in range(0, T, CE_CHUNK):
        e = min(s + CE_CHUNK, T)
        total = total + cp.checkpoint(chunk_loss, h[:, s:e], targets[:, s:e], use_reentrant=False)
    return total / ntok

# ---------------------------------------------------------------------------
# Optimizer: torch.optim.AdamW for all params (MuonAdamW leaks ~2-3 GB/step
# under torchrun at this scale). Same grouping/LR semantics as
# model.setup_optimizer. States CPU-offloaded between steps.
adamw_groups = [
    dict(params=list(model.lm_head.parameters()), lr=args.unembedding_lr, weight_decay=0.01),
    dict(params=list(model.transformer.wte.parameters()), lr=args.embedding_lr, weight_decay=0.001),
    dict(params=list(model.value_embeds.parameters()), lr=args.embedding_lr * 0.5, weight_decay=0.01),
    dict(params=[model.resid_lambdas], lr=args.scalar_lr * 0.01, weight_decay=0.05),
    dict(params=[model.x0_lambdas], lr=args.scalar_lr, weight_decay=0.0),
    dict(params=[model.smear_gate.weight, model.smear_lambda, model.backout_lambda], lr=0.2, weight_decay=0.0),
]
for shape in sorted({p.shape for p in model.transformer.h.parameters()}):
    adamw_groups.append(dict(
        params=[p for p in model.transformer.h.parameters() if p.shape == shape],
        lr=args.matrix_lr, weight_decay=args.weight_decay))
optimizer = torch.optim.AdamW(adamw_groups, betas=(0.9, 0.95), eps=1e-10, fused=True)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

def _iter_state_tensors():
    for k, v in optimizer.state.items():
        if torch.is_tensor(v):
            yield optimizer.state, k, v
        elif isinstance(v, dict):
            for kk, vv in list(v.items()):
                if torch.is_tensor(vv):
                    yield v, kk, vv

def offload_opt_to_cpu():
    for sd, kk, vv in _iter_state_tensors():
        if vv.device.type == "cuda":
            sd[kk] = vv.to("cpu")

def reload_opt_to_gpu():
    for sd, kk, vv in _iter_state_tensors():
        if vv.device.type == "cpu":
            sd[kk] = vv.to(device)

# ---------------------------------------------------------------------------
# Data: climbmix, packed at 131072 with BOS bestfit (same loader as pretrain).
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens/micro-batch/rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch {args.total_batch_size:,} => grad accum {grad_accum_steps}")

def get_lr_multiplier(step):
    warmup_iters = args.warmup_steps
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    warmdown_iters = round(args.num_iterations * args.warmdown_ratio)
    warmdown_start = args.num_iterations - warmdown_iters
    if step >= warmdown_start:
        progress = (step - warmdown_start + 1) / (args.num_iterations - warmdown_start + 1)
        return args.final_lr_frac + (1 - args.final_lr_frac) * (1 - progress)
    return 1.0

checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.output_tag)
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0
x, y, dataloader_state_dict = next(train_loader)

for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations

    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
        with torch.no_grad():
            for _ in range(eval_steps):
                vx, vy = next(val_loader)
                valid = vy >= 0
                vy_safe = torch.where(valid, vy, torch.zeros_like(vy))
                nb = token_bytes[vy_safe]
                loss_sum = longctx_forward_loss(model, vx, vy)
                total_nats += loss_sum * valid.sum()
                total_bytes += nb.sum()
        if ddp_world_size > 1:
            torch.distributed.all_reduce(total_nats, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_bytes, op=torch.distributed.ReduceOp.SUM)
        val_bpb = total_nats.item() / (math.log(2) * max(1, total_bytes.item()))
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        min_val_bpb = min(min_val_bpb, val_bpb)
        wandb_run.log({"step": step, "val/bpb": val_bpb})
        model.train()

    # save BEFORE the last_step break (the original order skipped the final
    # checkpoint entirely — step 2000 was never written)
    if (step > 0 and step % args.save_every == 0) or last_step:
        save_checkpoint(
            checkpoint_dir, step,
            model.state_dict(), optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": min_val_bpb,
                "model_config": mc,
                "user_config": vars(args),
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": args.total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {"min_val_bpb": min_val_bpb, "smooth_train_loss": smooth_train_loss, "total_training_time": total_training_time},
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    torch.cuda.synchronize()
    t0 = time.time()
    for micro in range(grad_accum_steps):
        loss = longctx_forward_loss(model, x, y)
        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        x, y, dataloader_state_dict = next(train_loader)
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    reload_opt_to_gpu()
    optimizer.step()
    offload_opt_to_cpu()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss.item()
    total_training_time += dt if step > 2 else 0
    tok_per_sec = int(args.total_batch_size / dt)
    if step % 10 == 0:
        eta_min = (args.num_iterations - step) * dt / 60
        print0(f"Step {step:05d}/{args.num_iterations} | loss {smooth_train_loss:.4f} | {tok_per_sec:,} tok/s | {dt:.1f}s/step | ETA {eta_min:.0f} min | mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    wandb_run.log({"step": step, "train/loss": smooth_train_loss, "tok/s": tok_per_sec})


compute_cleanup()
print0("Stage 7a done.")
