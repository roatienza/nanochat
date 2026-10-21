"""Stage 6 env validation: reference solutions pass, wrong solutions fail.

Run with a pinned hash so the leak test is reproducible across processes
(the env now uses stable_seed, so this is belt-and-braces).

Usage: .venv/bin/python -m tasks.agentic_validate  (from /data/rowel/nanochat)
"""
import random
import sys

sys.path.insert(0, "/data/rowel/nanochat")

from tasks.agentic import FAMILIES, make_conversation, AgenticSandbox


def main():
    t = AgenticSandbox()
    print(f"total instances: {len(t)}")

    # --- positive: handwritten correct solutions must score 1 ---
    from tasks.agentic import _reference_solution, stable_seed
    n_pos = 0
    for family, gen in FAMILIES.items():
        for level in (1, 2, 3):
            rng = random.Random(stable_seed(1234, family, level, 0))
            prompt, tests, extra = gen(rng, level)
            conv = make_conversation(prompt)
            conv["tests"] = tests
            sol = _reference_solution(family, level, rng, tests)
            r = t.evaluate(conv, sol)
            status = "OK " if r == 1 else "FAIL"
            print(f"{status} {family} L{level}: reward={r}")
            assert r == 1, f"reference solution failed for {family} L{level}"
            n_pos += 1
    print(f"positive: {n_pos}/15 pass")

    # --- negative: a battery of wrong solutions must all score 0 ---
    wrong = {
        "count_chars": {
            1: ["def solve(s):\n    return len(s)",
                "def solve(s):\n    return 0",
                "def solve(s):\n    return s.count('z')"],
            2: ["def solve(s):\n    return {c: 1 for c in s}",
                "def solve(s):\n    return len(s)",
                "def solve(s):\n    return sorted(set(s))"],
            3: ["def solve(s, k):\n    return list(s)[:k]",
                "def solve(s, k):\n    return sorted(set(s))[:k]",
                "def solve(s, k):\n    return list(s)[-k:]"],
        },
        "list_ops": {
            1: ["def solve(xs):\n    return xs",
                "def solve(xs):\n    return [x if x < 0 else 0 for x in xs]",
                "def solve(xs):\n    return sorted(xs)"],
            2: ["def solve(xs, k):\n    return xs[k:] + xs[:k]",  # left rotation
                "def solve(xs, k):\n    return xs",
                "def solve(xs, k):\n    return xs[::-1]"],
            3: ["def solve(xs):\n    return sorted(xs)",
                "def solve(xs):\n    return xs",
                "def solve(xs):\n    return [x for x in xs if x > 0]"],
        },
        "math_fns": {
            1: ["def solve(a, b):\n    return a * b",
                "def solve(a, b):\n    return a + b",
                "def solve(a, b):\n    return min(a, b)"],
            2: ["def solve(n):\n    return n % 2 == 0",
                "def solve(n):\n    return True",
                "def solve(n):\n    return n > 1"],
            3: ["def solve(n):\n    return n",
                "def solve(n):\n    return n * 2",
                "def solve(n):\n    return 2 ** n"],
        },
        "data_fix": {
            1: ["def solve(xs):\n    p = 1\n    for x in xs:\n        p *= x\n    return p",
                "def solve(xs):\n    return len(xs)",
                "def solve(xs):\n    return xs"],
            2: ["def solve(xs):\n    return max(xs)",
                "def solve(xs):\n    return min(xs)",
                "def solve(xs):\n    return sum(xs)"],
            3: ["def solve(n):\n    total = 0\n    for i in range(1, n):\n        total += i\n    return total",
                "def solve(n):\n    return n",
                "def solve(n):\n    return n * (n + 1) // 2 + n"],
        },
        "format_rows": {
            1: ["def solve(rows):\n    return str(rows)",
                "def solve(rows):\n    return '\\n'.join(str(r) for r in rows)",
                "def solve(rows):\n    return ''"],
            2: ["def solve(rows):\n    return sorted(rows)",
                "def solve(rows):\n    return rows",
                "def solve(rows):\n    return sorted(rows, key=lambda r: (r[1], r[0]))"],
            3: ["def solve(rows):\n    return '\\n'.join(str(r) for r in rows)",
                "def solve(rows):\n    return str(rows)",
                "def solve(rows):\n    return 'name,score,tag'"],
        },
    }

    n_neg = 0
    leaks = 0
    for family, gens in FAMILIES.items():
        for level in (1, 2, 3):
            for wi, wrong_sol in enumerate(wrong[family][level]):
                for seed in range(20):
                    rng = random.Random(stable_seed(999, family, level, seed))
                    prompt, tests, extra = gens(rng, level)
                    conv = make_conversation(prompt)
                    conv["tests"] = tests
                    r = t.evaluate(conv, wrong_sol)
                    n_neg += 1
                    if r != 0:
                        print(f"LEAK: {family} L{level} wrong#{wi} seed {seed} scored {r}")
                        leaks += 1
    print(f"negative: {n_neg} wrong-solution trials, {leaks} leaks (want 0)")
    assert leaks == 0, "reward leaks detected"
    print("ALL VALIDATION PASSED")


if __name__ == "__main__":
    main()