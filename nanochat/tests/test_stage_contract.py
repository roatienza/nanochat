"""
The stage chain's step contract.

THE CONTRACT
------------
Each stage reads the PREVIOUS stage's output at a hardcoded step:

    stage 2  reads the pretrain output   at step 277760
    stage 3  reads stage 2's output       at step 13630
    stage 4  reads stage 3's output       at step 509
    stage 5  reads stage 4's output       at step 100
    stage 6  reads stage 5's output       at step 177
    stage 7  reads stage 6's output       at step 177
    stage 8  reads stage 7b's output      at step 250
    stage 9  reads stage 8's output       at step 800

If any of those is wrong, the pipeline fails DAYS into a 4-day run, with
the expensive part already done. The steps are a property of the RECIPE,
not of the weights, so they can be checked ahead of time.

WHY THE STEPS ARE THE SAME FOR EVERY RUN
-----------------------------------------
The SFT stages stop on DATASET EXHAUSTION, not on a fixed step count: the
generator sets last_step when it has consumed the whole training set, and
the dataloader shards conversations across ranks (`cursor +=
ddp_world_size`), so the TOTAL consumed -- and therefore the step count
-- is independent of the world size. The RL stages derive their step
count from the dataset too (`num_steps = (len(train_task) //
examples_per_step) * num_epochs`).

So the step numbers are stable across the 1-GPU, 2-GPU and 4-GPU
profiles, and the contract below holds for this run.

WHAT IS CHECKED
---------------
The contract is checked against the checkpoints the ORIGINAL run
produced (the d28b-* tags), which are on disk. This is a valid proxy:
the recipe is the same, so the same steps will be written.
"""
import os
import re

import pytest

RECIPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recipe")
BASE = os.environ.get("NANOCHAT_BASE_DIR", "/data/rowel/nanochat_store/v2")

# The original run's tags.
ORIG = {
    "base": "base_checkpoints/d28b-v2",
    "sft_tools": "chatsft_checkpoints/d28b-v2",
    "warmup": "chatsft_checkpoints/d28b-warmup",
    "rl_warmup": "chatrl_checkpoints/d28b-warmup",
    "rl": "chatrl_checkpoints/d28b-rl100",
    "longc": "base_checkpoints/d28b-128k",
    "longc_final": "base_checkpoints/d28b-128k-final",
    "sft_long": "chatsft_checkpoints/d28b-128k-sft-v5",
    "touch_b": "chatsft_checkpoints/d28b-128k-sft-v6j",
}

# (stage script, the checkpoint it reads, the step it must find)
CONTRACT = [
    ("02_sft_tools.sh",     "base",        277760),
    ("03_warmup.sh",        "sft_tools",   13630),
    ("04_rl_math.sh",       "warmup",      509),
    ("05_rl_agentic.sh",    "rl_warmup",   100),
    ("06_rl_curriculum.sh", "rl",          177),
    ("07_longctx.sh",       "rl",          177),
    ("08_sft_long.sh",      "longc_final", 250),
    ("09_touchup.sh",       "sft_long",    800),
]


def _steps(rel):
    d = os.path.join(BASE, rel)
    if not os.path.isdir(d):
        return None
    out = set()
    for f in os.listdir(d):
        m = re.match(r"model_(\d+)\.pt$", f)
        if m:
            out.add(int(m.group(1)))
    return out


@pytest.mark.parametrize("script,key,step", CONTRACT)
def test_the_stage_reads_a_step_that_exists(script, key, step):
    have = _steps(ORIG[key])
    if have is None:
        pytest.skip(f"{ORIG[key]} is not on this machine")
    assert step in have, (
        f"{script} reads {ORIG[key]} at step {step}, but that directory has "
        f"steps {sorted(have)}. The stage would fail on a missing file."
    )


@pytest.mark.parametrize("script,key,step", CONTRACT)
def test_the_hardcoded_step_matches_the_contract(script, key, step):
    """The step in the script must be the one the contract names."""
    path = os.path.join(RECIPE, script)
    if not os.path.exists(path):
        pytest.skip(f"{script} is not in this tree")
    src = open(path).read()
    steps = re.findall(r"--model-step (\d+)", src)
    assert steps, f"{script} has no --model-step"
    # A stage may read more than one checkpoint (stage 7 reads the RL
    # output and its own 7a output; stage 9 reads six). The contract
    # names the FIRST one, which is the one that comes from the previous
    # stage.
    assert steps[0] == str(step), (
        f"{script} reads step {steps[0]} first, but the contract says {step}. "
        f"all of its steps: {steps}"
    )


def test_the_final_served_checkpoint_step():
    """Stage 9's last output is what stage 10 packs and stage 11 serves."""
    have = _steps(ORIG["touch_b"])
    if have is None:
        pytest.skip(f"{ORIG['touch_b']} is not on this machine")
    assert 240 in have, f"the served checkpoint must exist at step 240, have {sorted(have)}"

    # ... and the recipe must agree.
    cfg = open(os.path.join(RECIPE, "config.sh")).read()
    m = re.search(r'FINAL_STEP="\$\{FINAL_STEP:-(\d+)\}"', cfg)
    assert m, "config.sh has no FINAL_STEP"
    assert m.group(1) == "240", f"FINAL_STEP is {m.group(1)}, expected 240"


def test_the_single_gpu_stages_are_pinned_to_one_card():
    """Stages 7, 8 and 9 must run on ONE card.

    base_train_long.py and chat_sft_long.py use a plain torch.optim.AdamW
    and never reduce gradients across ranks, so on two cards each rank
    would optimise its own shard alone and rank 0's checkpoint -- the one
    that is kept -- would be the product of a single shard. The ranks
    would silently diverge.
    """
    for script in ("07_longctx.sh", "08_sft_long.sh", "09_touchup.sh"):
        path = os.path.join(RECIPE, script)
        if not os.path.exists(path):
            pytest.skip(f"{script} is not in this tree")
        src = open(path).read()
        assert "require_single_gpu" in src, (
            f"{script} does not call require_single_gpu: on two cards its "
            f"ranks would silently diverge."
        )
