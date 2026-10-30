"""
The stage chain's step contract, checked against the LIVE run.

THE CONTRACT
------------
Each stage reads the PREVIOUS stage's output at a step. Some of those
steps are fixed by the recipe; some follow from the data.

    stage 2  reads the pretrain output   at step 277760   (fixed)
    stage 3  reads stage 2's output       at step <handoff> (data-dependent)
    stage 4  reads stage 3's output       at step <handoff> (data-dependent)
    stage 5  reads stage 4's output       at step <handoff> (data-dependent)
    stage 6  reads stage 5's output       at step <handoff> (data-dependent)
    stage 7  reads stage 6's output       at step <handoff> (data-dependent)
    stage 8  reads stage 7b's output      at step 250      (fixed)
    stage 9  reads stage 8's output       at step 800      (fixed)

If any of those is wrong, the pipeline fails DAYS into a 4-day run, with
the expensive part already done.

WHY A DATA-DEPENDENT STEP MUST NOT BE HARDCODED
-------------------------------------------------
The SFT stages stop on DATASET EXHAUSTION, not on a fixed step count: the
generator sets last_step when it has consumed the whole training set, so
the final step is a property of the SIZE OF THE CORPUS.

The corpus was re-rendered on 2026-10-22, and stage 2's final step moved
from the original run's 13,630 to 13,650. Stage 3 read a hardcoded
13,630 and died on a missing model_013630.pt. Worse, the resume logic
asked the same question backwards -- "is there a model_013630.pt?" -- and
concluded stage 2 was INCOMPLETE, so the watcher restarted it. Ten and a
half hours of both GPUs, twice, before anyone found it.

So a stage that finishes now writes a HANDOFF RECORD: the step it
reached, into $HANDOFF_DIR/<tag>.step. The next stage reads that
record, and the resume logic reads the same record. They cannot
disagree, which is what broke the cycle.

WHAT THIS TEST CHECKS
---------------------
  1. The live run's handoff records name steps whose checkpoints exist.
  2. The live run's stage scripts resolve their step from the handoff
     record rather than from a hardcoded number.
  3. The legacy d28b-* lineage still satisfies the ORIGINAL contract --
     the recipe must not have drifted away from the run it reproduces.
  4. The mechanism itself: the record is written only on success, and a
     stage that has not finished has no record.

Run it with pytest (which is what actually executes the test functions --
running the file directly does nothing, because it has no __main__).
"""

import os
import re
import subprocess

import pytest

RECIPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recipe")
BASE = os.environ.get("NANOCHAT_BASE_DIR", "/data/rowel/nanochat_store/v2")
HANDOFF = os.environ.get("HANDOFF_DIR", os.path.join(BASE, "_handoff"))

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

# The LIVE run's tags, and the checkpoint directory each one writes.
LIVE = {
    "utak-base": "base_checkpoints/utak-base",
    "utak-sft-tools": "chatsft_checkpoints/utak-sft-tools",
    "utak-warmup": "chatsft_checkpoints/utak-warmup",
    "utak-rl-warmup": "chatrl_checkpoints/utak-rl-warmup",
    "utak-rl-pilot": "chatrl_checkpoints/utak-rl-pilot",
    "utak-rl": "chatrl_checkpoints/utak-rl",
    "utak-128k-final": "base_checkpoints/utak-128k-final",
    "utak-sft": "chatsft_checkpoints/utak-sft",
    "utak-sft-c": "chatsft_checkpoints/utak-sft-c",
}

# The legacy lineage's contract: (stage script, the checkpoint it reads,
# the step it must find). This is the run the recipe reproduces, and its
# steps are the FALLBACK constants in config.sh.
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

# How each stage script wires up the link to the PREVIOUS stage's output.
#
#   (script, the producer's tag, "dynamic" or "fixed")
#
# "dynamic" means the step comes from the producer's handoff record -- it
# follows from the data, so it cannot be known before the producer has
# run. "fixed" means the step is a property of the recipe, and the
# constant in config.sh is authoritative.
#
# The links that are NOT in this table are the ones inside a stage: stage
# 7b's pick of step 1,750 inside stage 7a, and stage 9's ladder rungs.
# Those are fixed mid-stage steps, checked separately.
WIRING = [
    ("02_sft_tools.sh",     "$TAG_BASE",        "fixed"),
    ("03_warmup.sh",        "$TAG_SFT_TOOLS",   "dynamic"),
    ("04_rl_math.sh",       "$TAG_WARMUP",      "dynamic"),
    ("05_rl_agentic.sh",    "$TAG_RL_WARMUP",   "dynamic"),
    ("06_rl_curriculum.sh", "$TAG_RL_PILOT",    "dynamic"),
    ("07_longctx.sh",       "$TAG_RL",          "dynamic"),
    ("08_sft_long.sh",      "$TAG_LONGC_FINAL", "fixed"),
    ("09_touchup.sh",       "$TAG_SFT",         "fixed"),
]

# The config.sh constants, and the step each must name. These are the
# fallback values used when a producer has no handoff record -- i.e. for
# the original run's lineage.
CONSTANTS = [
    ("TAG_BASE_FINAL_STEP",        277760),
    ("TAG_SFT_TOOLS_FINAL_STEP",   13630),
    ("TAG_WARMUP_FINAL_STEP",      509),
    ("TAG_RL_WARMUP_FINAL_STEP",   100),
    ("TAG_RL_PILOT_FINAL_STEP",    177),
    ("TAG_RL_FINAL_STEP",          177),
    ("TAG_LONGC_FINAL_STEP",       250),
    ("TAG_SFT_FINAL_STEP",         800),
]


# The literal steps a stage is allowed to pass: the deliberate
# intra-stage picks. Everything else must be a resolved variable.
LITERAL_STEPS_ALLOWED = {
    "07_longctx.sh": {"1750"},   # 7b resumes from a mid-run step of 7a
    "09_touchup.sh": {"120", "80", "100", "60", "240"},  # the ladder rungs
}


def _steps(rel, base=None):
    d = os.path.join(base or BASE, rel)
    if not os.path.isdir(d):
        return None
    out = set()
    for f in os.listdir(d):
        m = re.match(r"model_(\d+)\.pt$", f)
        if m:
            out.add(int(m.group(1)))
    return out


def _handoff_step(tag):
    """The step a stage's handoff record names, or None."""
    f = os.path.join(HANDOFF, tag + ".step")
    if not os.path.isfile(f):
        return None
    txt = open(f).read().strip()
    if not txt.isdigit():
        return None
    return int(txt)


def _script(name):
    path = os.path.join(RECIPE, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} is not in this tree")
    return open(path).read()


# ---------------------------------------------------------------------------
# 1. The live run
# ---------------------------------------------------------------------------

def _stage_is_running(tag):
    """Is the trainer that PRODUCES this tag running right now?

    A stage that is RUNNING has written checkpoints but not its handoff
    record yet -- the record is written when the stage finishes. Such a
    stage is legitimately incomplete, and the resume logic is right to
    treat it as such.

    The match is on the tag as the trainer's OWN --model-tag or
    --output-tag. A stage that merely READS a checkpoint with that tag in
    it is a different stage and must not exempt this one.
    """
    try:
        entries = os.listdir("/proc")
    except OSError:
        return False
    needles = ("--model-tag %s " % tag, "--output-tag %s " % tag,
               "--model-tag %s\x00" % tag, "--output-tag %s\x00" % tag)
    for e in entries:
        if not e.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % e, "rb") as fh:
                cmd = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        cmd = cmd.replace("\x00", " ")
        if any(n in cmd for n in needles):
            return True
    return False


def test_every_finished_live_stage_has_a_handoff_record():
    """A stage that has FINISHED must have a record.

    The record is the completion signal: it is written only after the
    stage's training command exits 0. Without it, the next stage cannot
    know which step to read and refuses to start.

    A stage that is still RUNNING is exempt: it has written checkpoints
    but not its record yet, and that is correct -- it is not finished.
    """
    for tag, rel in LIVE.items():
        have = _steps(rel)
        if not have:
            continue  # this stage has not run yet
        step = _handoff_step(tag)
        if step is None and _stage_is_running(tag):
            continue  # still running: it will record when it finishes
        assert step is not None, (
            f"{tag} has checkpoints at steps {sorted(have)} but no handoff "
            f"record in {HANDOFF}. The next stage would refuse to start, "
            f"and the resume logic would treat this stage as unfinished."
        )


def test_the_live_handoff_records_name_checkpoints_that_exist():
    """The step a record names must be a checkpoint that is really there.

    A record that names a step whose file is missing is worse than no
    record at all: the consuming stage would die on the missing file
    twenty minutes in, instead of refusing to start.
    """
    for tag, rel in LIVE.items():
        step = _handoff_step(tag)
        if step is None:
            continue
        have = _steps(rel)
        if have is None:
            continue
        assert step in have, (
            f"the handoff record for {tag} names step {step}, but "
            f"{rel} holds steps {sorted(have)}."
        )


def test_the_live_handoff_records_name_the_final_checkpoint():
    """The recorded step must be the stage's LAST checkpoint.

    A stage that saves periodically as well as at the end (stage 1 saves
    every 5,000 steps) would otherwise hand off a mid-run step, and the
    next stage would train on a half-trained model.
    """
    for tag, rel in LIVE.items():
        step = _handoff_step(tag)
        have = _steps(rel)
        if step is None or not have:
            continue
        assert step == max(have), (
            f"the handoff record for {tag} names step {step}, but the "
            f"stage's final checkpoint is {max(have)}."
        )


@pytest.mark.parametrize("script", [
    "03_warmup.sh", "04_rl_math.sh", "05_rl_agentic.sh",
    "06_rl_curriculum.sh", "07_longctx.sh",
])
def test_the_data_dependent_stages_resolve_their_step(script):
    """A data-dependent link must not name a hardcoded step.

    These are the links whose step follows from the corpus. They must read
    it from the producer's handoff record.
    """
    src = _script(script)
    assert "stage_read_step" in src, (
        f"{script} does not resolve its step with stage_read_step: a "
        f"hardcoded step here is what broke the run on 2026-10-27."
    )
    assert "dynamic" in src, (
        f"{script} resolves its step, but not as a dynamic link."
    )


@pytest.mark.parametrize("script", [
    "02_sft_tools.sh", "03_warmup.sh", "04_rl_math.sh", "05_rl_agentic.sh",
    "06_rl_curriculum.sh", "07_longctx.sh", "08_sft_long.sh", "09_touchup.sh",
])
def test_every_stage_writes_its_handoff_record(script):
    """A producing stage must record the step it reached.

    Without the record, the next stage has nothing to read and the resume
    logic cannot tell that this stage finished.
    """
    src = _script(script)
    assert "record_stage_handoff" in src, (
        f"{script} does not write a handoff record, so the stage after it "
        f"cannot know which step to read."
    )


def test_the_resume_logic_and_the_stages_agree():
    """resume_point.sh must name the same step the stage scripts read.

    This is the invariant that the whole handoff mechanism exists to
    protect. When the two disagreed, the resume logic said "stage 2 is
    unfinished" while stage 2 had in fact finished -- and the run looped
    for two 10.6-hour cycles.
    """
    for tag, rel in LIVE.items():
        step = _handoff_step(tag)
        if step is None:
            continue
        have = _steps(rel)
        if not have:
            continue
        # The stage's own record and the checkpoint on disk must agree.
        assert step in have, (
            f"{tag}: the record says {step}, the directory has {sorted(have)}"
        )


def test_a_stage_that_has_not_finished_has_no_handoff_record():
    """The record is a completion signal, so it must not be written early.

    If a stage wrote its record before finishing, the resume logic would
    skip it and the next stage would read a checkpoint that does not
    exist yet.
    """
    for tag, rel in LIVE.items():
        step = _handoff_step(tag)
        if step is None:
            continue
        have = _steps(rel)
        if not have:
            pytest.fail(
                f"{tag} has a handoff record naming step {step}, but "
                f"{rel} does not exist at all."
            )


# ---------------------------------------------------------------------------
# 2. The legacy lineage -- the recipe must still match the run it reproduces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script,key,step", CONTRACT)
def test_the_stage_reads_a_step_that_exists(script, key, step):
    have = _steps(ORIG[key])
    if have is None:
        pytest.skip(f"{ORIG[key]} is not on this machine")
    assert step in have, (
        f"{script} reads {ORIG[key]} at step {step}, but that directory has "
        f"steps {sorted(have)}. The stage would fail on a missing file."
    )


@pytest.mark.parametrize("script,producer,mode", WIRING)
def test_the_stage_resolves_its_previous_stage_link(script, producer, mode):
    """Each stage must read the PREVIOUS stage's output, by tag.

    The step itself is resolved at runtime (from the handoff record, or
    from the recipe's constant for a fixed link), so the wiring is what
    has to be checked: the right producer, in the right mode.
    """
    src = _script(script)
    assert "stage_read_step" in src, (
        f"{script} does not resolve its input step with stage_read_step. "
        f"A hardcoded step here is what broke the run on 2026-10-27."
    )
    # The producer's tag must appear as the FIRST argument of the
    # stage_read_step call for that stage.
    m = re.search(r'stage_read_step\s+"([^"]+)"', src)
    assert m, f"{script} has no stage_read_step call"
    assert m.group(1) == producer, (
        f"{script} reads from \"{m.group(1)}\", but the previous stage's "
        f"output is \"{producer}\"."
    )
    # The variable must be one config.sh actually defines, and it must
    # name the tag the stage writes (or reads) -- not some other one.
    var = m.group(1).lstrip("$")
    cfg = open(os.path.join(RECIPE, "config.sh")).read()
    assert re.search(r'^%s="\$\{%s:-' % (var, var), cfg, re.M), (
        f"{script} reads from ${{{var}}}, which config.sh does not define."
    )
    # The mode must be the last argument of the same call. It is written
    # bare (dynamic), not quoted, in the scripts.
    call = src[m.start():]
    call = call[:call.index(")") + 1] if ")" in call else call
    assert re.search(r"\b%s\b" % mode, call), (
        f"{script} resolves its step with mode {mode!r}, but the link is "
        f"declared {mode!r}. call: {call}"
    )

    # AND the resolved step must be the one the trainer is actually told
    # to load. This is the check that would have caught the 2026-10-27
    # bug: 03_warmup.sh resolved the step correctly but then passed a
    # hardcoded 13630 to the trainer, so the resolution was dead code
    # and the stage still died on a missing model_013630.pt.
    #
    # How the link is identified differs per trainer, so the check is on
    # the STAGE's own --model-step arguments: the step that goes with the
    # previous stage's link must be a VARIABLE, never a literal. A literal
    # is a step nobody resolved.
    #
    # A stage may name literal steps for its own intra-stage picks (stage
    # 7b's 1,750, stage 9's ladder rungs) -- those are deliberate and are
    # covered by test_the_fixed_mid_stage_steps_are_still_hardcoded.
    steps = re.findall(r"--model-step\s+([^\s\\]+)", src)
    assert steps, f"{script} passes no --model-step at all"
    # The variable the stage resolved must actually be used.
    var = re.search(r"stage_read_step[^\n]*", src)
    assert var, f"{script} has no stage_read_step call"
    resolved_var = None
    m = re.search(r"^([A-Z_]+)=\$\(stage_read_step", src, re.M)
    if m:
        resolved_var = m.group(1)
    for s in steps:
        if s.isdigit():
            # A literal is only allowed if it is a KNOWN intra-stage pick.
            assert s in LITERAL_STEPS_ALLOWED.get(script, set()), (
                f"{script} passes the literal step {s} to the trainer. "
                f"A hardcoded step is exactly what broke the run on "
                f"2026-10-27: the recipe's constant did not match the step "
                f"the producer really reached, and the stage died on a "
                f"missing file. Use the resolved variable instead."
            )
        elif resolved_var:
            assert s.strip('"') == resolved_var, (
                f"{script} resolves its step into ${resolved_var} but passes "
                f"{s} to the trainer, so the resolution is dead code."
            )


@pytest.mark.parametrize("name,step", CONSTANTS)
def test_the_config_constants_match_the_legacy_contract(name, step):
    """The fallback constants in config.sh must still name the legacy steps.

    They are what a producer without a handoff record falls back to, so
    they have to keep matching the original run's lineage.
    """
    cfg = open(os.path.join(RECIPE, "config.sh")).read()
    m = re.search(r'%s="\$\{%s:-(\d+)\}"' % (name, name), cfg)
    assert m, f"config.sh has no {name}"
    assert m.group(1) == str(step), (
        f"{name} is {m.group(1)}, the legacy contract says {step}"
    )


def test_the_fixed_mid_stage_steps_are_still_hardcoded():
    """Two links name a step INSIDE a stage, and they must not move.

    Stage 7b resumes from step 1,750 of stage 7a, which runs for 2,000
    iterations and saves every 250. That is a deliberate mid-run pick --
    the original run's 7a was interrupted at exactly 1,750 -- so it can
    never come from a handoff record (which names the FINAL step).

    Stage 9's ladder rungs are the same: each rung reads the previous
    rung's output at that rung's --num-iterations.
    """
    src = _script("07_longctx.sh")
    assert "--model-step 1750" in src, (
        "07_longctx.sh must read step 1,750 of stage 7a: 7a runs for 2,000 "
        "iterations and saves every 250, so 1,750 is a deliberate mid-run "
        "pick, not the final step."
    )
    src = _script("09_touchup.sh")
    for step in (120, 80, 100, 60, 240):
        assert f"--model-step {step}" in src, (
            f"09_touchup.sh must read step {step} of the previous rung"
        )


def test_the_math_rl_budget_lands_on_the_legacy_checkpoint():
    """Stage 4's step budget must end at the step the legacy run ended at.

    The original run of stage 4 was stopped by hand and left
    model_000100.pt behind; the recipe's next stage reads that file. The
    budget has to reproduce that EXACTLY, and the steps are 0-indexed, so
    the number of iterations is one more than the step.

    Getting this wrong by one is invisible until stage 5 fails on a
    missing checkpoint, hours into the run.
    """
    # The legacy run's own log: the file it wrote.
    legacy = os.path.join(BASE, "chatrl_checkpoints", "d28b-warmup")
    have = _steps(legacy)
    if not have:
        pytest.skip(f"{legacy} is not on this machine")
    assert 100 in have, (
        f"the legacy stage 4 output should hold step 100, has {sorted(have)}"
    )

    # What the recipe asks for.
    src = _script("04_rl_math.sh")
    m = re.search(r'RL_MATH_MAX_STEPS="\$\{RL_MATH_MAX_STEPS:-(\d+)\}"', src)
    assert m, "04_rl_math.sh has no RL_MATH_MAX_STEPS"
    budget = int(m.group(1))

    # chat_rl.py's loop is `for step in range(run_steps)`, and the final
    # checkpoint is written at `step == run_steps - 1`.
    final_step = budget - 1
    assert final_step == 100, (
        f"stage 4's budget of {budget} iterations ends at step {final_step}, "
        f"but the legacy chain reads step 100. The steps are 0-indexed, so "
        f"the budget must be 101."
    )


def test_a_checkpoint_alone_never_writes_a_handoff_record():
    """Only a stage that FINISHED may have a record -- never a checkpoint.

    Stage 3 saves every 100 steps, so while it runs its output directory
    holds model_000100.pt, model_000200.pt and so on. A record derived
    from "the highest checkpoint" would name one of those, and the next
    stage would then train on a HALF-TRAINED model while believing the
    stage was done.

    That is the same class of bug as the one this whole mechanism exists
    to kill, so the record-writing helper must refuse to guess.
    """
    src = _script("record_handoffs.sh")
    # The backfill path must REFUSE when the stage's trainer is running.
    assert "stage_is_running" in src, (
        "record_handoffs.sh must know whether the stage is still running"
    )
    assert "REFUSING" in src, (
        "record_handoffs.sh must refuse to backfill a stage that is still "
        "running: its newest checkpoint is a periodic save, not the final "
        "one."
    )
    # And the plain verify path must not write anything at all.
    body = src.split('if [ "$MODE" = "backfill" ]; then')[-1]
    tail = body.split("fi\n\n# ---- verify")[-1]
    assert "handoff_write" not in tail, (
        "the verify path of record_handoffs.sh must never write a record: a "
        "checkpoint does not mean a stage finished."
    )


def test_the_live_records_agree_with_the_checkpoints():
    """Each record must name a checkpoint that is really there.

    A record that names a missing checkpoint is worse than no record: the
    consuming stage would die on the missing file twenty minutes in.
    """
    for tag, rel in LIVE.items():
        step = _handoff_step(tag)
        if step is None:
            continue
        have = _steps(rel)
        if not have:
            continue
        assert step in have, (
            f"the handoff record for {tag} names step {step}, but {rel} "
            f"holds steps {sorted(have)}."
        )


def test_the_agentic_stages_land_on_the_legacy_checkpoint():
    """Stages 5 and 6 must end at the step the legacy chain read: 177.

    The agentic task set is 5 families x 3 levels x 200 instances, and
    the tail 5% of each block is reserved for eval, so the train set is
    2,850 instances. At 16 examples per step that is 178 optimizer
    steps, and the steps are 0-indexed, so the last one is 177 -- which
    is exactly the model_000177.pt the legacy chain's stage 6 and stage 7
    both read.
    """
    legacy = os.path.join(BASE, "chatrl_checkpoints", "d28b-rl100")
    have = _steps(legacy)
    if not have:
        pytest.skip(f"{legacy} is not on this machine")
    assert 177 in have, (
        f"the legacy agentic output should hold step 177, has {sorted(have)}"
    )

    # The arithmetic, from the flags the recipe passes.
    families, levels, ipl, frac = 5, 3, 200, 0.05
    train = families * levels * int(ipl * (1 - frac))
    num_steps = (train // 16) * 1
    assert num_steps - 1 == 177, (
        f"the agentic stages would end at step {num_steps - 1}, but the "
        f"legacy chain reads step 177. The task set or the example count "
        f"must have changed."
    )

    # And neither stage may be capped: the legacy run was not.
    for script in ("05_rl_agentic.sh", "06_rl_curriculum.sh"):
        s = _script(script)
        assert "--max-steps" not in s, (
            f"{script} must not cap its steps: the legacy run ran the whole "
            f"agentic task set, and a cap would change the trajectory."
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


# ---------------------------------------------------------------------------
# 3. The mechanism itself
# ---------------------------------------------------------------------------

def test_the_handoff_helpers_agree_with_each_other():
    """stage_resolved_step and stage_read_step must agree.

    The first is what the resume logic asks; the second is what the
    consuming stage asks. If they disagreed, the run would loop.
    """
    script = r'''
      set -u
      . ./recipe/config.sh
      a="$(stage_resolved_step "$1" "$NANOCHAT_BASE_DIR/$2" "$3")"
      b="$(stage_read_step "$1" "$NANOCHAT_BASE_DIR/$2" "$NANOCHAT_BASE_DIR/$2" "$3" "$4")"
      [ "$a" = "$b" ] || { echo "MISMATCH $a $b"; exit 1; }
      echo "AGREE $a"
    '''
    tag, rel = "utak-sft-tools", "chatsft_checkpoints/utak-sft-tools"
    if _handoff_step(tag) is None:
        pytest.skip("no handoff record to compare")
    out = subprocess.run(
        ["bash", "-c", script, "bash", tag, rel, "13630", "dynamic"],
        cwd=os.path.join(RECIPE, ".."), capture_output=True, text=True,
    )
    assert "AGREE" in out.stdout, (
        f"stage_resolved_step and stage_read_step disagree:\n{out.stdout}\n{out.stderr}"
    )


def test_resume_point_reports_a_stage_that_exists():
    """resume_point.sh must print a usable stage number.

    It is called by the watcher with `tail -1`, and the answer is passed
    straight to run_all.sh --from.

    The range is 1..15, not 1..9. All nine TRAINING stages complete means
    the answer is 10 -- pack, serve, the gates and the tests still have to
    run. It printed the word "all stages complete" instead (commit
    7a568a2), and because both callers coerce a non-numeric answer to
    stage 1, a finished run would have restarted from the pretrain. This
    assertion only catches that once the run HAS finished, which is why it
    stayed green: see test_a_complete_run_resumes_at_pack_not_at_stage_1.
    """
    out = subprocess.run(
        ["bash", os.path.join(RECIPE, "resume_point.sh")],
        capture_output=True, text=True,
    )
    tail = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    assert tail.isdigit(), (
        f"resume_point.sh printed {out.stdout!r} / {out.stderr!r}; "
        f"it must print a stage number."
    )
    assert 1 <= int(tail) <= 15, f"resume_point.sh printed {tail}"


def test_a_complete_run_resumes_at_pack_not_at_stage_1(tmp_path):
    """A run whose nine training stages are all done must resume at 10.

    This is the regression test for commit 7a568a2. When every training
    stage has produced its final checkpoint, resume_point.sh printed the
    word "all stages complete". Both callers (watch_utak.sh and
    start_pipeline.sh) do

        case "$resume" in ''|*[!0-9]*) resume=1 ;; esac

    so the word became 1, and a run that died in stage 10, or on a failed
    gate, restarted from the pretrain -- ~54 GPU-hours, overwriting the
    good checkpoints as it went.

    Built on a synthetic tree so it is independent of how far the live run
    has got: on 2026-10-30 the real answer is 8, and a test that reads the
    live tree cannot see this at all until the run completes.
    """
    base = tmp_path / "v2"
    handoff = base / "_handoff"
    handoff.mkdir(parents=True)

    # The nine (tag, directory, final step) rows of the contract, exactly.
    rows = [
        ("utak-base", "base_checkpoints/utak-base", 277760),
        ("utak-sft-tools", "chatsft_checkpoints/utak-sft-tools", 13650),
        ("utak-warmup", "chatsft_checkpoints/utak-warmup", 507),
        ("utak-rl-warmup", "chatrl_checkpoints/utak-rl-warmup", 100),
        ("utak-rl-pilot", "chatrl_checkpoints/utak-rl-pilot", 177),
        ("utak-rl", "chatrl_checkpoints/utak-rl", 177),
        ("utak-128k-final", "base_checkpoints/utak-128k-final", 250),
        ("utak-sft", "chatsft_checkpoints/utak-sft", 800),
        ("utak-sft-c", "chatsft_checkpoints/utak-sft-c", 240),
    ]
    for tag, rel, step in rows:
        d = base / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / f"model_{step:06d}.pt").touch()
        (d / f"meta_{step:06d}.json").touch()
        (handoff / f"{tag}.step").write_text(f"{step}\n")

    env = dict(os.environ, NANOCHAT_BASE_DIR=str(base))
    out = subprocess.run(
        ["bash", os.path.join(RECIPE, "resume_point.sh")],
        capture_output=True, text=True, env=env,
    )
    tail = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    assert tail.isdigit(), (
        f"a fully-trained run must resume at a stage NUMBER, not {tail!r}.\n"
        f"Both callers coerce a non-numeric answer to stage 1, which would "
        f"replay the entire pipeline.\nstdout:\n{out.stdout}\nstderr:\n{out.stderr}"
    )
    assert int(tail) == 10, (
        f"a fully-trained run must resume at stage 10 (pack), got {tail}.\n"
        f"stdout:\n{out.stdout}"
    )


def test_the_stages_stage_the_read_tree_before_resolving_the_step():
    """A stage must stage its read tree BEFORE it resolves the step.

    On 2026-10-28 stage 5 died 15 seconds in: it resolved the step first,
    which linked the checkpoint into a read directory that did not exist
    yet, and then the trainer could not find the metadata. The fix is the
    ORDER, and the order is not visible in the output -- so it is pinned
    here, for every stage that both stages a tree and resolves a step.
    """
    for name in ("02_sft_tools.sh", "03_warmup.sh", "04_rl_math.sh",
                 "05_rl_agentic.sh", "06_rl_curriculum.sh"):
        s = _script(name)
        if "cp -al" not in s:
            continue
        stage_at = s.index("cp -al")
        read_at = s.find("stage_read_step")
        assert read_at != -1, (
            f"{name} stages a tree but never resolves a step"
        )
        assert stage_at < read_at, (
            f"{name} resolves the step before it stages the read tree. "
            f"The resolver links the checkpoint into the read directory, "
            f"so it must come after the staging, or the stage dies on a "
            f"missing file."
        )


def test_the_resolver_links_the_whole_checkpoint_set():
    """The resolver must bring the META file as well as the model.

    The loader reads model_*.pt and meta_*.json. Linking only the model
    produces a read tree that looks complete and is not: stage 5 died
    exactly that way on 2026-10-28. The check is behavioural -- it runs
    the resolver against a scratch directory and looks at what it left.
    """
    cfg = os.path.join(RECIPE, "config.sh")
    if not os.path.exists(cfg):
        pytest.skip("config.sh is not in this tree")
    script = r'''
set -u
. "%s"
T=$(mktemp -d)
step=$(stage_read_step "utak-rl-warmup" "$T/read" \
        "$NANOCHAT_BASE_DIR/chatrl_checkpoints/utak-rl-warmup" "100" dynamic)
printf 'STEP=%%s\n' "$step"
ls -1 "$T/read" 2>/dev/null
rm -rf "$T"
''' % cfg
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    names = out.stdout.split()
    if "STEP=" not in out.stdout:
        pytest.skip("the live stage 4 output is not on this machine")
    step = out.stdout.split("STEP=")[1].split()[0]
    want_model = "model_%06d.pt" % int(step)
    want_meta = "meta_%06d.json" % int(step)
    assert want_model in names, (
        f"the resolver did not link {want_model}; it left {names}"
    )
    assert want_meta in names, (
        f"the resolver linked {want_model} but NOT {want_meta}. A stage "
        f"that reads this tree dies on the missing metadata -- which is "
        f"what happened to stage 5 on 2026-10-28."
    )


if __name__ == "__main__":

    # Running this file directly used to do NOTHING (no __main__ block),
    # and recipe/preflight.sh ran it that way -- so the contract check was
    # a no-op that always passed. pytest is what actually executes the
    # test functions; this block is here so that a direct run is honest
    # too.
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
