"""
Stage 7b - long-context SFT at sequence_len=131072.

Fine-tunes the Stage 7a long-context base (base_checkpoints/d28b-128k) on a
mixture of:
  - LongContextSFT synthetic tasks (needle QA, doc QA, tool calls over long
    docs, long summaries) - teaches 128k skills.
  - Short replay: ToolCalling (hermes) + ToolACE + SmolTalk - keeps the
    Stage-4/6 tool-calling and chat behavior sharp at short lengths.

Same memory design as scripts/base_train_long.py (per-block checkpointing,
chunked MLP, chunked CE, CPU-offloaded AdamW states). Conversations are
rendered with max_tokens=args.max_seq_len (chat_sft.py's default 2048
truncation would cut the long tasks) and packed BOS-bestfit into 131072 rows.

Run (2 GPUs):
  cd /data/rowel/nanochat && NANOCHAT_BASE_DIR=/data/rowel/nanochat_store/v2 \\
    CUDA_VISIBLE_DEVICES=3,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft_long \\
    -- --run=stage7b-sft-128k --num-iterations=600
"""
import argparse, os, sys, math, time, copy
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from tasks.longctx_sft import LongContextSFT
from tasks.toolcalling import ToolCalling
from tasks.toolace import ToolACE
from tasks.smoltalk import SmolTalk
from tasks.common import TaskMixture

parser = argparse.ArgumentParser(description="Stage 7b: long-context SFT (131072)")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--model-tag", type=str, default="d28b-128k")
parser.add_argument("--source-phase", type=str, default="base", choices=["base", "sft", "rl"],
                    help="which checkpoint dir the source lives in")
parser.add_argument("--model-step", type=int, default=None, help="default: last checkpoint in the tag dir")
parser.add_argument("--resume-step", type=int, default=0,
                    help="resume this --output-tag run from its own checkpoint at this step (0 = fresh start)")
parser.add_argument("--output-tag", type=str, default="d28b-128k-sft")
parser.add_argument("--max-seq-len", type=int, default=131072)
parser.add_argument("--short-window", type=int, default=512)
parser.add_argument("--device-batch-size", type=int, default=1)
parser.add_argument("--total-batch-size", type=int, default=262144)
parser.add_argument("--num-iterations", type=int, default=600, help="-1 = one epoch over the mixture")
parser.add_argument("--longctx-epochs", type=int, default=2, help="copies of LongContextSFT in the mixture")
parser.add_argument("--longctx-families", type=str, default="niah,doc_qa,multi_tool,long_summary")
parser.add_argument("--levels", type=str, default="1,2,3",
                    help="comma-separated LongContextSFT levels (0 = short ~4k rows)")
parser.add_argument("--instances-per-level", type=str, default="40",
                    help="int, or comma-separated per-level counts aligned with --levels")
parser.add_argument("--toolcall-epochs", type=int, default=1)
parser.add_argument("--toolace-epochs", type=int, default=1)
parser.add_argument("--followup-epochs", type=int, default=0,
                    help="copies of ToolFollowUp (tool-result -> prose answer) in the mixture")
parser.add_argument("--smoltalk-cap", type=int, default=0,
                    help="cap SmolTalk train rows via Task slicing (0 = no cap)")
parser.add_argument("--glaive-mr-epochs", type=int, default=0,
                    help="copies of GlaiveMultiRound (result->answer->re-ask->call) in the mixture")
parser.add_argument("--toolstop-epochs", type=int, default=0,
                    help="copies of ToolStop (bare-chain-then-stop, no-tool-needed) in the mixture")
parser.add_argument("--smoltalk-epochs", type=int, default=1)
parser.add_argument("--embedding-lr", type=float, default=0.072, help="0.24x of pretrain (SFT-ish)")
parser.add_argument("--unembedding-lr", type=float, default=0.0019)
parser.add_argument("--matrix-lr", type=float, default=0.0048)
parser.add_argument("--scalar-lr", type=float, default=0.12)
parser.add_argument("--weight-decay", type=float, default=0.028)
parser.add_argument("--warmup-ratio", type=float, default=0.02)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
parser.add_argument("--eval-every", type=int, default=50)
parser.add_argument("--eval-tokens", type=int, default=2*131072)
parser.add_argument("--save-every", type=int, default=100)
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
# Load the Stage 7a long-context base.
base_dir = get_base_dir()
src_dir = os.path.join(base_dir, {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[args.source_phase], args.model_tag)
if args.model_step is None:
    steps = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(src_dir) if f.startswith("model_")]
    args.model_step = max(steps)
print0(f"Loading source {src_dir} step {args.model_step}")
model_data, optimizer_data, meta_data = load_checkpoint(src_dir, args.model_step, device, load_optimizer=False)
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
mc = dict(meta_data["model_config"])
mc["sequence_len"] = args.max_seq_len
mc["short_window"] = args.short_window
mc.setdefault("rope_base", 100000)
print0(f"Building model: {mc}")
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
# Long-context forward (same as base_train_long.py).
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

def longctx_forward_loss(model, idx, targets, loss_reduction="mean"):
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
    # guard: a packed row can have zero supervised tokens (rare SmolTalk rows
    # whose assistant turn falls past the 131072 truncation) — 0/0 would be NaN
    # and would poison the optimizer moments permanently.
    ntok = (targets >= 0).sum()
    if ntok.item() == 0:
        return x.sum() * 0.0  # zero loss, still graph-connected
    for s in range(0, T, CE_CHUNK):
        e = min(s + CE_CHUNK, T)
        total = total + cp.checkpoint(chunk_loss, h[:, s:e], targets[:, s:e], use_reentrant=False)
    loss = total / ntok
    return loss

# ---------------------------------------------------------------------------
# Optimizer (AdamW everywhere; CPU-offloaded states).
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
# Data mixture: long synthetic + short replay. Rendered with the FULL
# max_seq_len (chat_sft.py's default 2048 truncation would kill the long tasks).
train_tasks = []
_levels = tuple(int(x) for x in args.levels.split(","))
_ipl = [int(x) for x in args.instances_per_level.split(",")]
if len(_ipl) == 1:
    _ipl = _ipl[0]
assert (not isinstance(_ipl, list)) or len(_ipl) == len(_levels), \
    f"--instances-per-level {_ipl} must align with --levels {_levels}"
_fams = tuple(f.strip() for f in args.longctx_families.split(",") if f.strip())
for _ in range(args.longctx_epochs):
    train_tasks.append(LongContextSFT(families=_fams, levels=_levels, instances_per_level=_ipl))
for _ in range(args.toolcall_epochs):
    train_tasks.append(ToolCalling(split="train"))
for _ in range(args.toolace_epochs):
    train_tasks.append(ToolACE(split="train"))
if args.followup_epochs > 0:
    from tasks.toolfollowup import ToolFollowUp
    for _ in range(args.followup_epochs):
        train_tasks.append(ToolFollowUp(split="train"))
if args.glaive_mr_epochs > 0:
    from tasks.glaive_multiround import GlaiveMultiRound
    for _ in range(args.glaive_mr_epochs):
        train_tasks.append(GlaiveMultiRound(split="train", max_rows=20000))
if args.toolstop_epochs > 0:
    from tasks.toolstop import ToolStop
    for _ in range(args.toolstop_epochs):
        train_tasks.append(ToolStop(n_rows=6000, seed=777))
for _ in range(args.smoltalk_epochs):
    st = SmolTalk(split="train", stop=args.smoltalk_cap or None)  # Task(start,stop) view
    train_tasks.append(st)
train_dataset = TaskMixture(train_tasks)
print0(f"Training mixture: {len(train_dataset):,} rows (LongCtx x{args.longctx_epochs}, "
       f"ToolCalling x{args.toolcall_epochs}, ToolACE x{args.toolace_epochs}, "
       f"FollowUp x{args.followup_epochs}, GlaiveMR x{args.glaive_mr_epochs}, "
       f"ToolStop x{args.toolstop_epochs}, SmolTalk x{args.smoltalk_epochs})")
val_dataset = TaskMixture([
    LongContextSFT(levels=(1,), instances_per_level=4, seed=999),
    ToolCalling(split="val"),
    ToolACE(split="val"),
    SmolTalk(split="test"),
])

# BOS-bestfit packing at full max_seq_len (adapted from chat_sft.py).
last_step_flag = False
def sft_data_generator(split, buffer_size=32):
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    row_capacity = args.max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()
    conv_buffer = []
    cursor = ddp_rank
    epoch = 1

    def refill():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor % dataset_size]
            ids, mask = tokenizer.render_conversation(conversation, max_tokens=args.max_seq_len)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
            # skip degenerate rows: no supervised tokens (assistant turn cut by
            # truncation) or empty render
            if sum(mask) == 0 or len(ids) < 8:
                continue
            conv_buffer.append((ids, mask))

    while True:
        rows = []
        mask_rows = []
        row_lengths = []
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill()
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for j, (ids, _) in enumerate(conv_buffer):
                    L = len(ids)
                    if L <= remaining and L > best_len:
                        best_idx = j
                        best_len = L
                if best_idx >= 0:
                    ids, mask = conv_buffer.pop(best_idx)
                    row.extend(ids)
                    mask_row.extend(mask)
                else:
                    row.append(bos_token)
                    mask_row.append(0)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])
            row_lengths.append(min(len(row), row_capacity))
        yield rows, mask_rows, row_lengths

def batches(split):
    gen = sft_data_generator(split)
    while True:
        rows, mask_rows, row_lengths = next(gen)
        inputs = torch.tensor([r[:-1] for r in rows], dtype=torch.int32, device=device)
        targets = torch.tensor([r[1:] for r in rows], dtype=torch.int64, device=device)
        mt = torch.tensor([m[1:] for m in mask_rows], dtype=torch.int8, device=device)
        targets[mt == 0] = -1
        for i, L in enumerate(row_lengths):
            if L < args.max_seq_len + 1:
                targets[i, L - 1:] = -1
        yield inputs, targets

train_batches = batches("train")
val_batches = lambda: batches("val")

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens/micro-batch/rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch {args.total_batch_size:,} => grad accum {grad_accum_steps}")

# dataset-driven stopping (like chat_sft.py): one epoch unless num_iterations set
if args.num_iterations == -1:
    num_iterations = (len(train_dataset) * args.max_seq_len) // args.total_batch_size
else:
    num_iterations = args.num_iterations
print0(f"Training for {num_iterations} steps")

def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        frac = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return args.final_lr_frac + (1 - args.final_lr_frac) * (1 - frac)

checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", args.output_tag)
start_step = 0
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0
if args.resume_step > 0:
    print0(f"Resuming {args.output_tag} from its own checkpoint at step {args.resume_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_step, device, load_optimizer=True)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model.load_state_dict(model_data, strict=True)
    optimizer.load_state_dict(optimizer_data)
    del model_data, optimizer_data
    loop_state = meta_data.get("loop_state", {})
    min_val_bpb = loop_state.get("min_val_bpb", float("inf"))
    smooth_train_loss = loop_state.get("smooth_train_loss", 0.0)
    total_training_time = loop_state.get("total_training_time", 0.0)
    start_step = args.resume_step
    print0(f"Restored loop state: min_val_bpb={min_val_bpb:.6f} smooth_train_loss={smooth_train_loss:.4f} total_training_time={total_training_time:.0f}s")
    # Keep the restored AdamW states on CPU, matching the fresh-run layout:
    # this script offloads optimizer states between steps, and materializing
    # them on GPU before the first forward OOMs at 131k context.
    for sd, kk, vv in _iter_state_tensors():
        if vv.device.type == "cuda":
            sd[kk] = vv.to("cpu")
    torch.cuda.empty_cache()
    # Fast-forward the (deterministic) packing stream: steps 0..start_step-1
    # consumed start_step*grad_accum_steps packed rows. Replay them so the
    # resumed run sees exactly the data it would have seen uninterrupted.
    n_replay = start_step * grad_accum_steps
    print0(f"Replaying data generator for {n_replay} packed rows to restore the cursor...")
    _t0 = time.time()
    for _ in range(n_replay):
        next(train_batches)
    print0(f"Replay done in {time.time()-_t0:.0f}s")

x, y = next(train_batches)

for step in range(start_step, num_iterations + 1):
    last_step = step == num_iterations
    progress = step / max(1, num_iterations)

    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))
        # NOTE: evaluate_bpb calls model(x, y, loss_reduction='none') which
        # materializes full logits (16 GB at 131k) — OOM. Use the chunked
        # forward instead, accumulating sum-loss and bytes manually.
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
        with torch.no_grad():
            for _ in range(eval_steps):
                vx, vy = next(val_batches())
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

    if (step > 0 and step % args.save_every == 0 and step != args.resume_step) or last_step:
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
        x, y = next(train_batches)
    lrm = get_lr_multiplier(progress)
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
    if step % 5 == 0:
        eta_min = (num_iterations - step) * dt / 60
        print0(f"Step {step:05d}/{num_iterations} | loss {smooth_train_loss:.4f} | {tok_per_sec:,} tok/s | {dt:.1f}s/step | ETA {eta_min:.0f} min | mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    wandb_run.log({"step": step, "train/loss": smooth_train_loss, "tok/s": tok_per_sec})

compute_cleanup()
print0("Stage 7b done.")