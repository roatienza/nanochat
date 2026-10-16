"""
Stage 6 — Agentic task environments for long-horizon RL on the execution.py sandbox.

Why this file exists
--------------------
Stage 5 (GSM8K RL) died of starvation: rollouts sampled the same 7,473 train
problems the warmup SFT had already memorized, so train reward saturated at 1.0
and the advantage (r - mu) went to zero — gradient death at step ~40. Stage 6's
fix is to change the reward source, not the optimizer: synthetic, verifiable
environments whose instances are generated fresh, so the model can never
memorize the rollout set.

Design (the decisions that matter)
----------------------------------
1) *Reward = execute_code success, nothing else.* nanochat's
   `nanochat/execution.py` runs the completion in a subprocess (rlimits, scrubbed
   env, temp dir). If the hidden tests pass, reward 1; else 0. No partial credit
   at RL time — partial credit invites reward hacking, and a 0/1 signal is all
   REINFORCE needs as long as the *group* has variance (some samples 0, some 1).

2) *Instances are generated, not fixed.* Every env is parameterized by a seed;
   `get_example(i)` derives the seed from i, so each rollout sees a fresh
   instance. This is the direct answer to Stage 5's failure mode.

3) *Rename-tolerant verification.* Measured on the Stage-5 RL checkpoint
   (2026-10-08): asked to write `solve(s)`, the model emitted a correct function
   but NAMED it `resolve`, and emitted code without an opening fence followed by
   prose. A verifier that regexes for `def solve` or requires full ``` fences
   would score correct code at 0. So: extract code fence-tolerantly, discover
   the function name the model actually defined, and bind the hidden tests to
   that name. The *contract* is "one function taking the stated args", not the
   literal name.

4) *Difficulty curriculum.* Each env family has level 1..3 (EnvScaler, ACL 2026
   Findings: steepest gains 0 -> 20 environments, diminishing after; curriculum
   easy -> hard is the highest-leverage thing in the stage). Level is part of
   the instance; the RL driver can slice by level via Task(start, stop, step).

5) *The conversation shape matches what the model already knows.* Prompt is a
   plain user turn; the trailing assistant message is empty (render_for_completion
   asserts the last message is assistant). No tool schemas in the prompt — the
   action space is plain code, not the CALL format (that's Stage 4's skill, not
   Stage 6's).

What "long-horizon" means here, honestly
----------------------------------------
True long-horizon agentic tasks (multi-step tool loops, stateful environments)
need an engine that re-prompts between steps. nanochat's Engine forces tool
outputs only for its built-in calculator loop. This file's envs are therefore
*single-shot but multi-constraint*: one completion must satisfy several
independent hidden tests, which is the honest first rung of the ladder on this
codebase. The multi-turn loop is the next rung and needs engine work (see
STATUS notes).

Families (each cheap to verify, each with 3 levels):
  count_chars   — string statistics
  list_ops      — list transformations
  math_fns      — pure arithmetic functions
  data_fix      — fix a broken function given its wrong output
  format_rows   — text/table formatting

Sandbox caveat (from execution.py's own docstring, repeated here): it is NOT a
security boundary — no seccomp, no network block. Fine for training rewards;
must be stated to anyone deploying.
"""

import hashlib
import json
import random
import re

from nanochat.execution import execute_code
from tasks.common import Task


def stable_seed(*parts):
    """Process-independent seed (tuple.__hash__ is PYTHONHASHSEED-salted, which
    made instances differ between torchrun ranks / reruns)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "little")


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def extract_code(completion):
    """
    Fence-tolerant code extraction. Measured failure modes on the Stage-5 ckpt:
      (a) full ```python ... ``` block  -> standard case
      (b) raw code, then an ORPHAN closing fence, then prose
          (no opening fence!)          -> regex above misses; cut at the fence
      (c) raw code + trailing prose    -> cut at the fence too
    """
    m = re.findall(r"```(?:python)?\s*\n(.*?)\n```", completion, re.DOTALL)
    if m:
        return m[0].strip()
    return completion.split("```")[0].strip()


def discover_fn_name(code, fallback="solve"):
    """The name the model actually gave the function (measured: it renames)."""
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
    return m.group(1) if m else fallback


def run_hidden_tests(code, tests, timeout=5.0):
    """
    Execute code + hidden tests in the sandbox. Returns (reward, info).
    The model's function is called by whatever name it defined; tests are
    written with {fn} placeholders and substituted before execution.
    Substitution is a plain string replace, NOT str.format — the tests embed
    dict/repr literals whose braces str.format would misread as fields.
    """
    fn = discover_fn_name(code)
    test_src = "\n".join(t.replace("{fn}", fn) for t in tests)
    program = code + "\n" + test_src
    result = execute_code(program, timeout=timeout)
    info = {
        "fn": fn,
        "error": result.error,
        "timeout": result.timeout,
        "memory_exceeded": result.memory_exceeded,
    }
    return int(result.success), info


def make_conversation(prompt):
    """nanochat conversation shape for completion. Trailing assistant turn is
    REQUIRED (render_for_completion asserts it) and empty."""
    return {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""},
    ]}


# -----------------------------------------------------------------------------
# Family 1: count_chars — string statistics
# -----------------------------------------------------------------------------

def _gen_count_chars(rng, level):
    letters = "abcdefghij"
    if level == 1:
        # redraw until the target char occurs at least twice in s — a count of
        # 0 or 1 is satisfiable by degenerate wrong solutions (count('z')==0
        # leaks when ch absent; len(s) can equal 1) (measured leaks)
        while True:
            s = "".join(rng.choice(letters) for _ in range(rng.randint(4, 8)))
            ch = rng.choice(letters)
            if s.count(ch) >= 2:
                break
        prompt = (f"Write a Python function `solve(s)` that returns the number of "
                  f"times the character '{ch}' appears in the string s.")
        tests = [f"assert {{fn}}({s!r}) == {s.count(ch)}"]
        ref = s.count(ch)
    elif level == 2:
        s = "".join(rng.choice(letters) for _ in range(rng.randint(10, 20)))
        prompt = ("Write a Python function `solve(s)` that returns a dict mapping each "
                  "character in s to how many times it appears.")
        expected = {c: s.count(c) for c in sorted(set(s))}
        tests = [f"assert {{fn}}({s!r}) == {json.dumps(expected)}"]
        ref = expected
    else:  # level 3
        # redraw until the target ordering differs from ALL degenerate wrong
        # answers: first-k chars, first-k distinct, sorted-set[:k], last-k chars
        # (measured leaks at seeds 6/7)
        while True:
            s = "".join(rng.choice(letters) for _ in range(rng.randint(20, 30)))
            k = rng.randint(2, 3)
            counts = {}
            for c in s:
                counts[c] = counts.get(c, 0) + 1
            expected = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
            expected = [c for c, _ in expected]
            degenerate = [list(s)[:k], list(dict.fromkeys(s))[:k],
                          sorted(set(s))[:k], list(s)[-k:]]
            if all(w != expected for w in degenerate):
                break
        prompt = (f"Write a Python function `solve(s, k)` that returns the k most "
                  f"common characters in s, ordered by count descending then "
                  f"alphabetically ascending, as a list.")
        tests = [f"assert {{fn}}({s!r}, {k}) == {json.dumps(expected)}"]
        ref = expected
    return prompt, tests, {"ref": ref, "level": level}


# -----------------------------------------------------------------------------
# Family 2: list_ops — list transformations
# -----------------------------------------------------------------------------

def _gen_list_ops(rng, level):
    if level == 1:
        # redraw until the list has a negative — identity passes otherwise
        # (measured leak: 2/10 seeds drew all-non-negative lists)
        while True:
            xs = [rng.randint(-10, 10) for _ in range(rng.randint(3, 6))]
            if any(x < 0 for x in xs):
                break
        prompt = ("Write a Python function `solve(xs)` that returns the list with "
                  "every negative number replaced by 0.")
        expected = [max(0, x) for x in xs]
        tests = [f"assert {{fn}}({xs!r}) == {expected!r}"]
    elif level == 2:
        # redraw until the rotation is direction-discriminating:
        #   k % len(xs) != 0            (full cycle is directionless)
        #   2k % len(xs) != 0           (k == len/2 makes left == right)
        # (measured leaks: 3/10 seeds at k%len==0, 2/30 at 2k==len)
        while True:
            xs = [rng.randint(-20, 20) for _ in range(rng.randint(5, 9))]
            k = rng.randint(1, 3)
            if k % len(xs) != 0 and (2 * k) % len(xs) != 0:
                break
        prompt = (f"Write a Python function `solve(xs, k)` that returns the list "
                  f"rotated right by k positions (cyclically).")
        expected = xs[-k % len(xs):] + xs[:-k % len(xs)] if xs else []
        tests = [f"assert {{fn}}({xs!r}, {k}) == {expected!r}"]
    else:  # level 3
        xs = [rng.randint(-30, 30) for _ in range(rng.randint(6, 10))]
        prompt = ("Write a Python function `solve(xs)` that returns the longest "
                  "strictly increasing contiguous sublist of xs (return the first "
                  "one if there are ties).")
        best, cur = [], []
        for i, x in enumerate(xs):
            if cur and x > cur[-1]:
                cur.append(x)
            else:
                cur = [x]
            if len(cur) > len(best):
                best = cur
        tests = [f"assert {{fn}}({xs!r}) == {best!r}"]
    return prompt, tests, {"level": level}


# -----------------------------------------------------------------------------
# Family 3: math_fns — pure arithmetic
# -----------------------------------------------------------------------------

def _gen_math_fns(rng, level):
    if level == 1:
        # redraw until gcd(a,b) is a PROPER divisor of both — min(a,b) passes
        # whenever a divides b or b divides a (measured leak: 2/20 seeds)
        import math
        while True:
            a, b = rng.randint(2, 20), rng.randint(2, 20)
            g = math.gcd(a, b)
            if g != min(a, b) and g != max(a, b):
                break
        prompt = ("Write a Python function `solve(a, b)` that returns the greatest "
                  "common divisor of a and b.")
        tests = [f"assert {{fn}}({a}, {b}) == {g}",
                 f"assert {{fn}}({b}, {a}) == {g}"]
    elif level == 2:
        n = rng.randint(5, 40)
        prompt = ("Write a Python function `solve(n)` that returns True if n is "
                  "prime, else False.")
        def is_prime(n):
            if n < 2:
                return False
            return all(n % d for d in range(2, int(n ** 0.5) + 1))
        tests = [f"assert {{fn}}({n}) == {is_prime(n)}",
                 f"assert {{fn}}({n + 1}) == {is_prime(n + 1)}"]
    else:  # level 3
        n = rng.randint(3, 8)
        prompt = ("Write a Python function `solve(n)` that returns the number of "
                  "ways to tile a 2 x n board with 2x1 dominoes (Fibonacci).")
        def domino(n):
            a, b = 1, 1
            for _ in range(n):
                a, b = b, a + b
            return a
        tests = [f"assert {{fn}}({n}) == {domino(n)}",
                 f"assert {{fn}}({n + 1}) == {domino(n + 1)}"]
    return prompt, tests, {"level": level}


# -----------------------------------------------------------------------------
# Family 4: data_fix — fix a broken function (agentic debugging shape)
# -----------------------------------------------------------------------------

def _gen_data_fix(rng, level):
    if level == 1:
        # redraw until product != sum — [2,1,3]-style lists collide
        # (measured leak at seed 7)
        while True:
            xs = [rng.randint(1, 9) for _ in range(rng.randint(3, 6))]
            p = 1
            for x in xs:
                p *= x
            if p != sum(xs):
                break
        prompt = (f"The function below is supposed to return the SUM of the list "
                  f"but it returns the product. Fix it. Keep the same function "
                  f"name and signature.\n\n```python\ndef solve(xs):\n"
                  f"    p = 1\n    for x in xs:\n        p *= x\n    return p\n```\n\n"
                  f"Return the corrected function.")
        tests = [f"assert {{fn}}({xs!r}) == {sum(xs)!r}"]
    elif level == 2:
        # draw xs until the test discriminates the broken fn from common wrong
        # ones: need max(xs) != max(abs(x)) (broken fn leaks) AND
        # sum(xs) != max(abs(x)) (sum-guess leaks) (measured leaks at seeds 7)
        while True:
            xs = [rng.randint(-9, 9) for _ in range(rng.randint(4, 7))]
            ma = max(abs(x) for x in xs)
            if max(xs) != ma and sum(xs) != ma:
                break
        prompt = (f"The function below is supposed to return the maximum "
                  f"absolute value in the list but it returns the maximum value "
                  f"(ignoring sign). Fix it. Keep the same function name.\n\n"
                  f"```python\ndef solve(xs):\n    return max(xs)\n```\n\n"
                  f"Return the corrected function.")
        tests = [f"assert {{fn}}({xs!r}) == {max(abs(x) for x in xs)!r}"]
    else:  # level 3: off-by-one
        # n=1..3 make the "off by one" broken fn correct (sum 1..n-1 == sum 1..n
        # only when n<=1, but small n also collide with other wrong fns);
        # require n >= 4 (unchanged range, floor raised)
        n = rng.randint(4, 10)
        prompt = (f"The function below is supposed to return the sum of integers "
                  f"from 1 to n INCLUSIVE but it is off by one (excludes n). Fix "
                  f"it. Keep the same function name.\n\n```python\ndef solve(n):\n"
                  f"    total = 0\n    for i in range(1, n):\n        total += i\n"
                  f"    return total\n```\n\nReturn the corrected function.")
        tests = [f"assert {{fn}}({n}) == {n * (n + 1) // 2}"]
    return prompt, tests, {"level": level}


# -----------------------------------------------------------------------------
# Family 5: format_rows — text/table formatting (office-task transfer shape)
# -----------------------------------------------------------------------------

def _gen_format_rows(rng, level):
    names = ["ann", "bob", "cyd", "dan", "eva", "fay"]
    if level == 1:
        rows = [(rng.choice(names), rng.randint(1, 99)) for _ in range(rng.randint(2, 4))]
        prompt = ("Write a Python function `solve(rows)` that takes a list of "
                  "(name, score) tuples and returns them as a single string, one "
                  "per line, formatted as 'name: score' (single spaces).")
        expected = "\n".join(f"{n}: {s}" for n, s in rows)
        tests = [f"assert {{fn}}({rows!r}) == {expected!r}"]
    elif level == 2:
        # redraw until the naive sort `sorted(rows)` (name-then-score) differs
        # from the target sort AND rows are not already in target order —
        # otherwise wrong solutions pass (measured leaks)
        while True:
            rows = [(rng.choice(names), rng.randint(1, 99)) for _ in range(rng.randint(3, 5))]
            target = sorted(rows, key=lambda r: (-r[1], r[0]))
            if sorted(rows) != target and rows != target:
                break
        prompt = ("Write a Python function `solve(rows)` that takes a list of "
                  "(name, score) tuples and returns the rows sorted by score "
                  "descending, then name ascending, as a list of tuples.")
        expected = sorted(rows, key=lambda r: (-r[1], r[0]))
        tests = [f"assert {{fn}}({rows!r}) == {expected!r}"]
    else:  # level 3: CSV-ish
        rows = [(rng.choice(names), rng.randint(1, 99), rng.choice(["x", "y"])) for _ in range(rng.randint(3, 5))]
        prompt = ("Write a Python function `solve(rows)` that takes a list of "
                  "(name, score, tag) tuples and returns a CSV string with a "
                  "header 'name,score,tag' and one row per tuple, comma-separated, "
                  "no spaces after commas.")
        expected = "name,score,tag\n" + "\n".join(f"{n},{s},{t}" for n, s, t in rows)
        tests = [f"assert {{fn}}({rows!r}) == {expected!r}"]
    return prompt, tests, {"level": level}


# -----------------------------------------------------------------------------
# The Task class
# -----------------------------------------------------------------------------

FAMILIES = {
    "count_chars": _gen_count_chars,
    "list_ops": _gen_list_ops,
    "math_fns": _gen_math_fns,
    "data_fix": _gen_data_fix,
    "format_rows": _gen_format_rows,
}


class AgenticSandbox(Task):
    """
    Dataset of freshly generated sandbox-verified coding tasks.

    Index layout (deterministic, so any rank derives the same instance):
        index = family_idx * (levels * instances_per_level) + level_slot * instances + i

    Slicing for a curriculum: Task(start, stop) over the flattened index space,
    or filter by level via the `levels` argument.
    """

    def __init__(self, levels=(1, 2, 3), instances_per_level=200, seed=1234, **kwargs):
        super().__init__(**kwargs)
        self.family_names = sorted(FAMILIES)
        self.levels = tuple(sorted(levels))
        self.instances_per_level = instances_per_level
        self.seed = seed

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.family_names) * len(self.levels) * self.instances_per_level

    def _decompose(self, index):
        # NOTE: get_example() receives the PHYSICAL index (Task.__getitem__
        # already applied start + i*step), so do NOT add start again here.
        n_per_family = len(self.levels) * self.instances_per_level
        family_idx = index // n_per_family
        rem = index % n_per_family
        level_slot = rem // self.instances_per_level
        i = rem % self.instances_per_level
        family = self.family_names[family_idx % len(self.family_names)]
        level = self.levels[level_slot % len(self.levels)]
        return family, level, i

    def get_example(self, index):
        family, level, i = self._decompose(index)
        rng = random.Random(stable_seed(self.seed, family, level, i))
        prompt, tests, extra = FAMILIES[family](rng, level)
        conversation = make_conversation(prompt)
        conversation["family"] = family
        conversation["level"] = level
        conversation["tests"] = tests
        return conversation

    def evaluate(self, conversation, completion):
        """0/1: do the hidden tests pass in the sandbox?"""
        assert isinstance(completion, str)
        code = extract_code(completion)
        if not code or "def " not in code:
            return 0
        reward, _ = run_hidden_tests(code, conversation["tests"])
        return reward

    def reward(self, conversation, completion):
        return float(self.evaluate(conversation, completion))


def _reference_solution(family, level, rng, tests=None):
    """Handwritten correct solutions used only by the __main__ self-test."""
    if family == "count_chars":
        if level == 1:
            # The prompt names the character: "...the character 'c' appears...".
            # Parse it from the prompt-shaped test instead of guessing from counts.
            import re as _re
            m = _re.search(r"assert \{fn\}\('(.*?)'\) == (\d+)", tests[0])
            s, n = m.group(1), int(m.group(2))
            # recover the target char: the generator picked ch then built the
            # test as s.count(ch) == n; the char is whichever one makes it true
            # AND is named in the prompt. Simplest robust source: the prompt is
            # not passed here, so use the count-matching char, disambiguated by
            # preferring the one the generator's rng drew — but any char with
            # count == n satisfies the test, so just pick the first match.
            from collections import Counter
            cnt = Counter(s)
            matches = [c for c, k in cnt.items() if k == n]
            if not matches:
                # n == 0 case: char absent from s; any absent char works
                ch = next(c for c in "abcdefghij" if c not in cnt)
            else:
                ch = matches[0]
            return f"def solve(s):\n    return s.count({ch!r})"
        if level == 2:
            return "def solve(s):\n    return {c: s.count(c) for c in sorted(set(s))}"
        return ("def solve(s, k):\n"
                "    counts = {}\n    for c in s:\n        counts[c] = counts.get(c, 0) + 1\n"
                "    return [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]")
    if family == "list_ops":
        if level == 1:
            return "def solve(xs):\n    return [max(0, x) for x in xs]"
        if level == 2:
            return ("def solve(xs, k):\n    k = k % len(xs) if xs else 0\n"
                    "    return xs[-k:] + xs[:-k] if k else xs")
        return ("def solve(xs):\n    best, cur = [], []\n    for x in xs:\n"
                "        if cur and x > cur[-1]:\n            cur.append(x)\n"
                "        else:\n            cur = [x]\n"
                "        if len(cur) > len(best):\n            best = cur\n    return best")
    if family == "math_fns":
        if level == 1:
            return ("def solve(a, b):\n    while b:\n        a, b = b, a % b\n    return a")
        if level == 2:
            return ("def solve(n):\n    if n < 2:\n        return False\n"
                    "    d = 2\n    while d * d <= n:\n        if n % d == 0:\n"
                    "            return False\n        d += 1\n    return True")
        return ("def solve(n):\n    a, b = 1, 1\n    for _ in range(n):\n"
                "        a, b = b, a + b\n    return a")
    if family == "data_fix":
        if level == 1:
            return "def solve(xs):\n    return sum(xs)"
        if level == 2:
            return "def solve(xs):\n    return max(abs(x) for x in xs)"
        return ("def solve(n):\n    total = 0\n    for i in range(1, n + 1):\n"
                "        total += i\n    return total")
    if family == "format_rows":
        if level == 1:
            return ("def solve(rows):\n    return '\\n'.join(f'{n}: {s}' for n, s in rows)")
        if level == 2:
            return ("def solve(rows):\n    return sorted(rows, key=lambda r: (-r[1], r[0]))")
        return ("def solve(rows):\n    out = ['name,score,tag']\n"
                "    for n, s, t in rows:\n        out.append(f'{n},{s},{t}')\n"
                "    return '\\n'.join(out)")
    raise ValueError(family)


if __name__ == "__main__":
    # Self-test: every family x level generates, and a known-good solution passes.
    t = AgenticSandbox()
    print(f"total instances: {len(t)}")
    from collections import Counter
    fams = Counter()
    for i in range(0, len(t), 97):
        conv = t[i]
        fams[(conv["family"], conv["level"])] += 1
    print("coverage sample:", dict(sorted(fams.items())))

    # End-to-end reward check with a handwritten correct solution per family/level
    for family, gen in FAMILIES.items():
        for level in (1, 2, 3):
            rng = random.Random(stable_seed(1234, family, level, 0))
            prompt, tests, extra = gen(rng, level)
            conv = make_conversation(prompt)
            conv["tests"] = tests
            # a generic correct solver per family/level; for count_chars L1 the
            # solution must match the instance's character, so build it from
            # the tests themselves instead of regenerating the rng stream.
            sol = _reference_solution(family, level, rng, tests)
            r = t.evaluate(conv, sol)
            status = "OK " if r == 1 else "FAIL"
            print(f"{status} {family} L{level}: reward={r}")
            assert r == 1, f"reference solution failed for {family} L{level}"
