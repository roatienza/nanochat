"""
Stage 6b — Agentic RL with weighted block curriculum + replay (scripts/chat_rl_curriculum.py).

Derived from scripts/chat_rl_agentic.py. The optimization core (REINFORCE with
DAPO-style token-level normalization, advantage = r - mu) is UNTOUCHED, as is the
eval (same val split, same pass@k reduction, same per-block breakdown) — results
are directly comparable to the stage-6 pilot.

What changes and why (evidence from the pilot log, stage6-rl-agentic-pilot):
1) WEIGHTED BLOCK SAMPLING replaces the sequential walk over the TaskSequence.
   The pilot walked blocks in order, so 42/178 steps (24%) landed entirely inside
   uncracked blocks (format_rows@L3 steps 100-105, list_ops@L2+L3 steps 119-141,
   math_fns@L3 steps 167-177) where all 16 samples of every example scored 0 ->
   advantage = 0 -> zero gradient. Meanwhile data_fix@L1 (trained once at steps
   36-47, never revisited) regressed from 1.00 to 0.00 on eval. The fix: sample
   example indices from a weighted pool every step — replay blocks keep their
   capability (and their temp-1.0 failures still carry gradient), frontier blocks
   get maximal exposure.
2) TRAIN-ONLY EXEMPLAR SCAFFOLD (optional, --exemplar-scaffold 1) for the 5 frontier
   blocks. The pilot never scored a single success on these blocks (0/256 samples
   per step), so plain reweighting cannot crack them: with a 0/1 reward and
   advantage = r - mu, an all-zero sample group contributes nothing no matter how
   often it is drawn. A worked example (instance 0 of the block with its verified solution)
   is prepended to the user prompt at train time only, giving the policy a path
   to its first successes; the exemplar is annealed off at --exemplar-anneal-step
   so the eval (which never sees exemplars) measures whether the capability stuck. Eval comparability is preserved because the hint touches
   only the user message — the conversation's own "tests" travel with it, so
   reward delegation is unaffected.

1 GPU:
python -m scripts.chat_rl_curriculum --run=stage6b-rl-curriculum ...

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl_curriculum -- --run=default
"""

import argparse
import os
import random
import itertools
import wandb
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.agentic import AgenticSandbox, FAMILIES, stable_seed

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Stage 6b agentic RL — weighted block curriculum + replay")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of passes over the weighted example pool")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--agentic-instances-per-level", type=int, default=200, help="instances per (family, level) in the AgenticSandbox envs")
parser.add_argument("--agentic-levels", type=str, default="1,2,3", help="difficulty levels to include, comma-separated")
parser.add_argument("--agentic-eval-split-frac", type=float, default=0.05, help="tail fraction of instances reserved for eval (never trained on)")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
# --- Stage 6b curriculum knobs ---
parser.add_argument("--frontier-frac", type=float, default=0.5,
                    help="fraction of each step's examples drawn from frontier blocks (rest from replay blocks)")
parser.add_argument("--exemplar-scaffold", type=int, default=1, help="1 = prepend a worked example to frontier-block prompts at train time")
parser.add_argument("--exemplar-anneal-step", type=int, default=89, help="step at which exemplars are turned off (0 = never on)")
parser.add_argument("--curriculum-seed", type=int, default=7, help="seed for the weighted example pool shuffles")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

# Agentic envs: fresh instance per index. Reserve the tail of each
# (family, level) block for eval so eval instances are never trained on.
levels = tuple(int(x) for x in args.agentic_levels.split(","))
ipl = args.agentic_instances_per_level
holdout = max(1, int(ipl * args.agentic_eval_split_frac))
train_ipl = ipl - holdout

# Block layout (must match _make_task below): family_names is sorted(FAMILIES),
# blocks are (family, level) in the same nested loop order as the slices.
def _block_list():
    probe = AgenticSandbox(levels=levels, instances_per_level=ipl)
    blocks = []
    for f in range(len(probe.family_names)):
        for ls in range(len(levels)):
            blocks.append((probe.family_names[f], levels[ls]))
    return blocks

BLOCKS = _block_list() # [(family, level), ...] in TaskSequence slice order
N_BLOCKS = len(BLOCKS)
FRONTIER = {("count_chars", 3), ("format_rows", 3), ("list_ops", 2), ("list_ops", 3), ("math_fns", 3)}
# Frontier = blocks the pilot never scored on (0/256 samples/step) + the exact
# set that sat at 0.00 in every eval from step 120 to 177.

# Train-time exemplar scaffold for frontier blocks. A worked example (instance 0
# of the block, solution verified against its own hidden tests) is prepended to
# the user message at train time only. Probe evidence (stage6b_exemplar_probe,
# model_000177, 16 samples/instance, training params): exemplars lift frontier
# pass@16 from 0.000 to 1.000 on count_chars@L3 and list_ops@L2, to 0.500 on
# math_fns@L3; format_rows@L3 and list_ops@L3 stay 0. Prose hints lift nothing.
EXEMPLAR_SOLUTIONS = {
    ("list_ops", 2):
        "def solve(xs, k):\n    k = k % len(xs)\n    return xs[-k:] + xs[:-k]\n",
    ("count_chars", 3):
        "def solve(s, k):\n"
        "    counts = {}\n"
        "    for c in s:\n"
        "        counts[c] = counts.get(c, 0) + 1\n"
        "    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]\n"
        "    return [c for c, _ in order]\n",
    ("format_rows", 3):
        "def solve(rows):\n"
        "    lines = ['name,score,tag']\n"
        "    for n, s, t in rows:\n"
        "        lines.append(','.join((n, str(s), t)))\n"
        "    return '\\n'.join(lines)\n",
    ("math_fns", 3):
        "def solve(n):\n    a, b = 1, 2\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return a\n",
    ("list_ops", 3):
        "def solve(xs):\n"
        "    best, cur = [], []\n"
        "    for x in xs:\n"
        "        if cur and x > cur[-1]:\n"
        "            cur.append(x)\n"
        "        else:\n"
        "            cur = [x]\n"
        "        if len(cur) > len(best):\n"
        "            best = cur\n"
        "    return best\n",
}

def make_exemplar(block):
    """Worked example from instance 0 of this block (train-side)."""
    fam, lvl = block
    rng0 = random.Random(stable_seed(1234, fam, lvl, 0))
    prompt0, tests0, _ = FAMILIES[fam](rng0, lvl)
    sol = EXEMPLAR_SOLUTIONS[block]
    return (f"Example:\n{prompt0}\nCorrect solution:\n```python\n{sol}```\n\n"
            f"Now solve this new problem:\n")

# Precompute the exemplar prefix per frontier block once (instances are
# deterministic, so this is stable across ranks/steps).
EXEMPLARS = {block: make_exemplar(block) for block in FRONTIER if block in EXEMPLAR_SOLUTIONS}

def _make_task(train=True):
    # Task slicing is start/stop/step over the flattened index
    # index = family_idx * (levels*ipl) + level_slot*ipl + i
    # We keep i < train_ipl for train and i >= train_ipl for eval, per block.
    task = AgenticSandbox(levels=levels, instances_per_level=ipl)
    if train:
        # build a TaskSequence of per-(family,level) slices [0, train_ipl)
        slices = []
        n_fam = len(task.family_names)
        for f in range(n_fam):
            for ls in range(len(levels)):
                base = f * len(levels) * ipl + ls * ipl
                slices.append(AgenticSandbox(levels=levels, instances_per_level=ipl,
                                             start=base, stop=base + train_ipl))
        from tasks.common import TaskSequence
        return TaskSequence(slices)
    else:
        slices = []
        n_fam = len(task.family_names)
        for f in range(n_fam):
            for ls in range(len(levels)):
                base = f * len(levels) * ipl + ls * ipl
                slices.append(AgenticSandbox(levels=levels, instances_per_level=ipl,
                                             start=base + train_ipl, stop=base + ipl))
        from tasks.common import TaskSequence
        return TaskSequence(slices)

train_task = _make_task(train=True)
val_task = _make_task(train=False)
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

# -----------------------------------------------------------------------------
# Stage 6b: weighted example pool.
# Each block contributes train_ipl sequence indices [b*train_ipl, (b+1)*train_ipl).
# Weights: frontier blocks share --frontier-frac of the pool, replay blocks share
# the rest. The pool is rebuilt (re-shuffled) at the start of every epoch, so the
# weighted distribution is realized exactly within each epoch.
pool_rng = random.Random(args.curriculum_seed + ddp_rank)

def build_pool():
    frontier_total = max(1, int(round(len(train_task) * args.frontier_frac)))
    per_frontier = frontier_total // sum(1 for b in BLOCKS if b in FRONTIER)
    n_replay_blocks = sum(1 for b in BLOCKS if b not in FRONTIER)
    per_replay = (len(train_task) - per_frontier * sum(1 for b in BLOCKS if b in FRONTIER)) // n_replay_blocks
    pool = []
    for b_idx, block in enumerate(BLOCKS):
        lo = b_idx * train_ipl
        n = per_frontier if block in FRONTIER else per_replay
        # draw n indices from this block's range, with replacement (n can exceed
        # train_ipl for oversampled frontier blocks)
        picks = [pool_rng.randrange(lo, lo + train_ipl) for _ in range(n)]
        pool.extend(picks)
    pool_rng.shuffle(pool)
    return pool

_pool = build_pool()
_pool_pos = 0
print0(f"Weighted pool: {len(_pool)} examples | frontier-frac {args.frontier_frac}"
       + " | frontier: " + " ".join(f"{f}@L{l}" for (f, l) in sorted(FRONTIER)))

def next_train_index():
    global _pool, _pool_pos
    if _pool_pos >= len(_pool):
        _pool = build_pool()
        _pool_pos = 0
    idx = _pool[_pool_pos]
    _pool_pos += 1
    return idx

def exemplars_active(step):
    return bool(args.exemplar_scaffold) and (args.exemplar_anneal_step <= 0 or step < args.exemplar_anneal_step)

@torch.no_grad()
def get_batch(step):
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # ok to use this token, it's only for padding and isn't used in the loss.
    for example_step in range(args.examples_per_rank):

        example_idx = next_train_index()
        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Stage 6b: train-only exemplar scaffold for frontier blocks. A worked
        # example is prepended to the user message; the conversation's own
        # "tests" are untouched, so reward delegation is unaffected.
        block = (conversation.get("family"), conversation.get("level"))
        if exemplars_active(step) and block in EXEMPLARS:
            scaffolded = dict(conversation)
            scaffolded["messages"] = [dict(m) for m in conversation["messages"]]
            scaffolded["messages"][0]["content"] = EXEMPLARS[block] + scaffolded["messages"][0]["content"]
            conversation = scaffolded

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval() # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed, # must make sure to change the seed for each sampling step
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        pad_id = assistant_end
        padded_generated_token_sequences = [seq + [pad_id] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        mu = rewards.mean()
        advantages = rewards - mu
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards, advantages, block

# -----------------------------------------------------------------------------
# Simple evaluation loop for pass@k over any Task (agentic envs here)
def run_agentic_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates a Task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= args.device_batch_size # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
            "family": conversation.get("family", "?"),
            "level": conversation.get("level", "?"),
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}") # total batch size in sequences/step
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size # per GPU
args.examples_per_rank = examples_per_rank # used by get_batch
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch(0)
for step in range(num_steps):

    # Rebind the current step into the generator (hints anneal by step)
    batch_iterator = get_batch(step)

    # Evaluate the model once in a while and log to wandb
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device) # pass@k for k=1..device_batch_size
        records_iter = run_agentic_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0)
        records = list(records_iter) # collect all records
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # normalize by the total number of records
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        # per-family/level Pass@1 breakdown (records carry family/level)
        fam_stats = {}
        for r in records:
            key = (r.get("family", "?"), r.get("level", "?"))
            s = fam_stats.setdefault(key, [0, 0])
            s[1] += 1
            s[0] += any(o["is_correct"] for o in r["outcomes"][:1])
        fam_str = " | ".join(f"{f}@L{l}:{c/n:.2f}" for (f, l), (c, n) in sorted(fam_stats.items()))
        print0(f"Step {step} | per-family/level Pass@1: {fam_str}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        log_fam = {f"pass1_{f}_L{l}": c / n for (f, l), (c, n) in sorted(fam_stats.items())}
        wandb_run.log({
            "step": step,
            **log_passk,
            **log_fam,
        })

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    sequence_lengths = []
    block_reward_sums = {} # stage 6b: per-block train reward for curriculum visibility
    block_reward_counts = {}
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all, block = next(batch_iterator)
        # Evaluate the loss and gradients
        model.train() # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        # For logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)
        block_reward_sums[block] = block_reward_sums.get(block, 0.0) + rewards_all.mean().item()
        block_reward_counts[block] = block_reward_counts.get(block, 0) + 1

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp: # aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })
    # stage 6b: per-block train reward (which blocks are giving signal this step)
    block_str = " | ".join(
        f"{f}@L{l}:{block_reward_sums[(f, l)] / block_reward_counts[(f, l)]:.2f}"
        for (f, l) in sorted(block_reward_sums.keys())
    )
    print0(f"Step {step}/{num_steps} | per-block train reward: {block_str}")
    wandb_run.log({
        "step": step,
        **{f"trainreward_{f}_L{l}": block_reward_sums[(f, l)] / block_reward_counts[(f, l)]
           for (f, l) in block_reward_sums.keys()},
    })

    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # base the model tag on the depth of the base model
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish() # wandb run finish
compute_cleanup()