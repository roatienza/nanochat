"""
The step BUDGET, and why it is not the same thing as --num-iterations.

A stage's recipe has two numbers that look interchangeable and are not:

  --num-iterations   how many steps the schedule is, AND the schedule
                     itself. The learning rate is a function of it
                     (progress = step / num_iterations), so changing it
                     changes the trajectory of every step.

  --max-steps        how many steps actually RUN. The learning rate
                     still follows the full --num-iterations schedule.

They differ exactly when a run was stopped part way through by hand and
the next stage resumes from where it stopped. The original run was
stopped twice like that -- stage 8 at step 100, stage 7a at step 1,750 --
and the recipe has to reproduce those stopping points without changing
the learning rate the original had when it got there.

THE ORACLE IS THE ORIGINAL RUN'S OWN CHECKPOINT METADATA
--------------------------------------------------------
Each checkpoint's meta_*.json records the run's own arguments, so the
original's behaviour can be read off the disk rather than guessed at:

  chatsft_checkpoints/d28b-128k-sft-v5/meta_000100.json  has no
      resume_step: that run was INTERRUPTED at step 100.
  .../meta_000200.json .. meta_000800.json  all have
      resume_step=100: the continuation ran 101..800 once.

  base_checkpoints/d28b-128k/  has metas for 250..1,750 and NO
      meta_002000: the original's 7a never wrote a step-2,000
      checkpoint, so 1,750 is the step the original's 7b read.

These tests assert the recipe agrees with that.
"""

import json
import os
import re

import pytest


RECIPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recipe")
BASE = os.environ.get("NANOCHAT_BASE_DIR", "/data/rowel/nanochat_store/v2")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _script(name):
    path = os.path.join(RECIPE, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} is not in this tree")
    return open(path).read()


def _meta(rel, step):
    """The original run's own metadata for one checkpoint, or None."""
    path = os.path.join(BASE, rel, "meta_%06d.json" % step)
    if not os.path.isfile(path):
        return None
    try:
        return json.load(open(path))
    except (ValueError, OSError):
        return None


def _trainer(name):
    path = os.path.join(ROOT, "scripts", name)
    if not os.path.exists(path):
        pytest.skip(f"{name} is not in this tree")
    return open(path).read()


def _function_body(src, name):
    """The source of one function, parsed properly rather than by slicing.

    Slicing on the next "def" is not safe here: these scripts have
    module-level code after the function, so a slice would swallow the
    whole rest of the file and any mention of the budget anywhere in it.
    """
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = src.splitlines()
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"no function named {name}")


# ---------------------------------------------------------------------------
# The trainers must offer a budget that does not change the schedule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trainer", ["chat_sft_long.py", "base_train_long.py"])
def test_the_long_trainers_take_a_step_budget(trainer):
    """--max-steps must exist, and default to "run the whole schedule"."""
    src = _trainer(trainer)
    assert '"--max-steps"' in src, (
        f"{trainer} has no --max-steps: a stage that must stop part way "
        f"through its schedule has no way to say so."
    )
    m = re.search(r'"--max-steps",\s*type=int,\s*default=(-?\d+)', src)
    assert m, f"{trainer}'s --max-steps has no int default"
    assert m.group(1) == "-1", (
        f"{trainer}'s --max-steps defaults to {m.group(1)}; -1 means "
        f"'no budget', and a positive default would silently truncate "
        f"every run that does not ask for a budget."
    )


@pytest.mark.parametrize("trainer", ["chat_sft_long.py", "base_train_long.py"])
def test_a_budget_does_not_change_the_learning_rate_schedule(trainer):
    """The budget caps the COMPUTE. The SCHEDULE must be untouched.

    This is the whole point of the flag. If the budget changed the
    learning rate, then a run stopped at step 100 of an 800-step
    schedule would have a different learning rate at step 100 than the
    original did -- and the checkpoint it produces is not the original's
    checkpoint, whatever the step number says.

    The two trainers compute the multiplier differently (one takes a
    progress fraction, the other a step), so the invariant is asserted
    on the thing that matters: the multiplier is a function of the
    SCHEDULE, and run_steps appears nowhere in the learning-rate path.
    """
    src = _trainer(trainer)

    # The schedule length is the one the multiplier is built from. The
    # two trainers express the warmdown differently -- one by ratio
    # (chat_sft_long) and one by iteration count (base_train_long) --
    # so each is checked against its own mechanism. What both must do
    # is derive it from --num-iterations.
    if "warmdown_iters" in src:
        assert re.search(r"warmdown_iters\s*=\s*round\(args\.num_iterations", src), (
            f"{trainer}: the warmdown must be computed from --num-iterations."
        )
        assert re.search(r"warmdown_start\s*=\s*args\.num_iterations", src), (
            f"{trainer}: the warmdown must start at a point computed from "
            f"--num-iterations."
        )
    else:
        # The ratio form: the multiplier is a function of progress, and
        # progress is the fraction of the SCHEDULE.
        assert re.search(r"progress\s*=\s*step\s*/\s*max\(1,\s*num_iterations\)", src), (
            f"{trainer}: the learning-rate progress must be the fraction "
            f"of the SCHEDULE (num_iterations)."
        )
        assert "progress = step / max(1, run_steps)" not in src, (
            f"{trainer}: the learning-rate progress must not be the "
            f"fraction of the budget (run_steps)."
        )

    # The budget must not appear anywhere in the learning-rate path.
    assert "run_steps" not in _function_body(src, "get_lr_multiplier"), (
        f"{trainer}: run_steps appears in get_lr_multiplier: the budget "
        f"must not reach the learning rate."
    )

    # And the budget must actually stop the loop. The loop's start
    # varies (chat_sft_long can resume from a step), but its END must
    # be the budget.
    assert re.search(r"for step in range\((?:start_step, )?run_steps \+ 1\)", src), (
        f"{trainer}: the training loop must run to run_steps, not to "
        f"--num-iterations, or the budget does nothing."
    )


# ---------------------------------------------------------------------------
# Stage 8: the original trained 0..100 once, then 101..800 once
# ---------------------------------------------------------------------------

def test_stage_8_does_not_train_the_same_steps_twice():
    """8a must stop at 100, because 8b runs 101..800.

    Without the budget, 8a runs the whole 800-step schedule and 8b then
    runs 101..800 again: 1,500 steps where the original did 800, and
    ~4.6 hours of a GPU at the original's own step time.
    """
    src = _script("08_sft_long.sh")

    # 8a: the first half, capped at 100, but on the 800-step schedule.
    m = re.search(
        r'run_mod\s+"\$LOG_DIR/stage8a_sft_long\.log".*?'
        r'--num-iterations\s+(\d+)\s*(?:--max-steps\s+(\d+))?',
        src, re.S)
    assert m, "08_sft_long.sh has no stage 8a invocation"
    num_iterations = int(m.group(1))
    budget = int(m.group(2)) if m.group(2) else None

    assert num_iterations == 800, (
        f"8a's schedule is {num_iterations} steps, expected 800: the "
        f"schedule is what sets the learning rate, and it must match the "
        f"original's."
    )
    assert budget == 100, (
        f"8a has no --max-steps 100, so it runs all 800 steps and 8b "
        f"trains 101..800 a second time. The budget must be 100."
    )

    # 8b: the continuation, still on the 800-step schedule.
    m = re.search(
        r'run_mod\s+"\$LOG_DIR/stage8b_sft_long_resume\.log".*?'
        r'--num-iterations\s+(\d+)\s*(?:--max-steps\s+(\d+))?',
        src, re.S)
    assert m, "08_sft_long.sh has no stage 8b invocation"
    assert int(m.group(1)) == 800, (
        f"8b's schedule is {m.group(1)} steps, expected 800."
    )
    assert not m.group(2), (
        f"8b must not be capped: it is the continuation that reaches the "
        f"stage's final step."
    )
    assert "--resume-step 100" in src, (
        "8b must resume from step 100, the step 8a stops at."
    )


def test_stage_8s_budget_matches_what_the_original_actually_did():
    """The oracle: the original run's own checkpoint metadata.

    If the original's step-100 checkpoint has no resume_step, the run
    that wrote it was interrupted there -- which is what a budget of 100
    reproduces. If the later checkpoints have resume_step=100, the
    continuation covered 101..800 -- which is what 8b does.
    """
    first = _meta("chatsft_checkpoints/d28b-128k-sft-v5", 100)
    if first is None:
        pytest.skip("the original run's stage 8 output is not on this machine")
    assert "resume_step" not in first.get("user_config", {}) or \
        first["user_config"]["resume_step"] in (0, None), (
        "the original's step-100 checkpoint has a resume_step, so the "
        "original's first half was itself a continuation, and 8a's "
        "budget is not 100."
    )
    assert first["user_config"]["num_iterations"] == 800, (
        "the original's stage 8 ran an 800-step schedule"
    )

    later = _meta("chatsft_checkpoints/d28b-128k-sft-v5", 200)
    if later is not None:
        assert later["user_config"].get("resume_step") == 100, (
            "the original's step-200 checkpoint does not resume from 100, "
            "so the original did not have a 100/700 split"
        )
        assert later["user_config"]["num_iterations"] == 800, (
            "the original's continuation still followed the 800-step "
            "schedule, which is why the budget must not change it"
        )


# ---------------------------------------------------------------------------
# Stage 7a: the original's 7b read step 1,750
# ---------------------------------------------------------------------------

def test_stage_7a_stops_where_7b_reads():
    """7a must stop at 1,750, which is the step 7b resumes from.

    7a's schedule is 2,000 steps and 7b reads step 1,750, so without a
    budget 7a computes 250 steps (at the original's ~24.4s a step, ~1.7
    hours of a GPU) that nothing reads.
    """
    src = _script("07_longctx.sh")

    m = re.search(
        r'run_mod\s+"\$LOG_DIR/stage7a_longctx\.log".*?'
        r'--num-iterations\s+(\d+)\s*(?:--max-steps\s+(\d+))?',
        src, re.S)
    assert m, "07_longctx.sh has no stage 7a invocation"
    assert int(m.group(1)) == 2000, (
        f"7a's schedule is {m.group(1)} steps, expected 2000."
    )
    assert m.group(2) and int(m.group(2)) == 1750, (
        "7a has no --max-steps 1750, so it computes 250 steps that 7b "
        "never reads."
    )

    # 7b must still read step 1,750, and must not be capped: it is the
    # stage's final output.
    assert "--model-step 1750" in src, "7b must read step 1,750 of 7a"
    m = re.search(
        r'run_mod\s+"\$LOG_DIR/stage7b_longctx_anneal\.log".*?'
        r'--num-iterations\s+(\d+)\s*(?:--max-steps\s+(\d+))?',
        src, re.S)
    assert m, "07_longctx.sh has no stage 7b invocation"
    assert not m.group(2), (
        "7b must not be capped: it produces the checkpoint stage 8 reads."
    )


def test_stage_7as_budget_matches_what_the_original_actually_did():
    """The oracle: the original's 7a never wrote a step-2,000 checkpoint.

    base_checkpoints/d28b-128k holds metas for 250..1,750 and no
    meta_002000, so the original's 7a stopped at 1,750 (its save
    ordering never wrote the final checkpoint). The recipe's 7a runs a
    2,000-step schedule, so the budget is what reproduces the original.
    """
    d = os.path.join(BASE, "base_checkpoints", "d28b-128k")
    if not os.path.isdir(d):
        pytest.skip("the original run's stage 7a output is not on this machine")
    metas = sorted(f for f in os.listdir(d) if re.match(r"meta_\d+\.json$", f))
    if not metas:
        pytest.skip("the original run's stage 7a output has no metadata")

    steps = sorted(int(re.search(r"meta_(\d+)\.json", f).group(1)) for f in metas)
    assert 1750 in steps, (
        f"the original's 7a should hold step 1,750, holds {steps}"
    )
    assert 2000 not in steps, (
        "the original's 7a holds a step-2,000 checkpoint, so it ran to the "
        "end of its 2,000-step schedule and the budget is not 1,750"
    )
    # The original's 7a was on a 2,000-step schedule: the budget changes
    # the compute, not the schedule.
    m = _meta("base_checkpoints/d28b-128k", 1750)
    if m is not None:
        assert m["user_config"]["num_iterations"] == 2000, (
            "the original's 7a ran a 2,000-step schedule"
        )


# ---------------------------------------------------------------------------
# The agentic stages must NOT be capped
# ---------------------------------------------------------------------------

def test_the_agentic_stages_are_not_capped():
    """Stages 5 and 6 ran the WHOLE agentic task set; a cap would change
    the trajectory.

    The legacy chain reads step 177, which is where the task set
    naturally ends (5 families x 3 levels x 190 instances, 16 per step).
    That is not a hand-stopped run like stage 8's, so there is nothing
    to reproduce -- and capping it would land somewhere else.
    """
    for script in ("05_rl_agentic.sh", "06_rl_curriculum.sh"):
        src = _script(script)
        assert "--max-steps" not in src, (
            f"{script} must not cap its steps: the legacy run ran the "
            f"whole agentic task set, and a cap would change the "
            f"trajectory."
        )


# ---------------------------------------------------------------------------
# The budget and the schedule must not be confused in the other trainers
# ---------------------------------------------------------------------------

def test_chat_rl_keeps_its_own_budget_semantics():
    """chat_rl.py is 0-indexed; its budget must not drift.

    chat_rl.py's loop is `for step in range(run_steps)` and its final
    checkpoint is written at `step == run_steps - 1`, so a budget of N
    ends at step N-1. The long trainers are 1-indexed and end at step N.
    Both are correct for their own trainer, but the difference has to
    be deliberate: stage 4's budget of 101 ends at step 100, which is
    the step stage 5 reads.
    """
    src = _trainer("chat_rl.py")
    if '"--max-steps"' not in src:
        pytest.skip("chat_rl.py has no --max-steps")
    assert re.search(r"for step in range\(run_steps\)", src), (
        "chat_rl.py's loop must run to run_steps"
    )
    assert re.search(r"step == run_steps - 1", src), (
        "chat_rl.py must write its final checkpoint at run_steps - 1: its "
        "steps are 0-indexed, so a budget of N ends at step N-1."
    )
    # And the learning rate still follows the full schedule.
    assert re.search(r"def get_lr_multiplier\(it\):\s*\n\s*lrm = 1\.0 - it / num_steps", src), (
        "chat_rl.py's learning rate must still follow the full num_steps "
        "schedule, not the budget"
    )
