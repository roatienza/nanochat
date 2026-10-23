#!/usr/bin/env python3
"""Regression test: no PROSE training target may contain a literal backslash-n.

Why: a supervised target containing the two characters '\\' 'n' (instead of a
real newline) teaches the model to EMIT a literal backslash-n. At inference
that becomes a period-2 byte-BPE cycle ("\\n\\n\\n\\n...") which runs to the
token cap (120s+ per request on the live harness) and then gets saved into
the session history, where it becomes an in-context attractor that poisons
every later turn.

Caught in the wild on 2026-10-20: tasks/harness_basics.py _LONG_ANSWER_PADS[1]
used "\\n\\n" in a Python literal, so the target itself was poisoned.

Run: .venv/bin/python tasks/test_no_literal_newline.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BS = chr(92) + "n"  # literal backslash-n (NOT a newline)

# Prose answers may legitimately contain escaped newlines when they are
# quoting CODE (e.g. "return '\\n'.join(...)"). Those live in 'python' parts,
# which this test does not inspect. Everything else is prose.


def iter_prose_targets(conv):
    for i, m in enumerate(conv.get("messages", [])):
        c = m.get("content")
        if isinstance(c, str):
            yield f"[{i}]{m.get('role')}", c
        elif isinstance(c, list):
            for j, part in enumerate(c):
                if part.get("type") == "text":
                    yield f"[{i}]{m.get('role')}.text{j}", part.get("text") or ""


def check_task(cls, name, n=250):
    hits = []
    try:
        t = cls(n_rows=n, seed=99)
    except TypeError:
        t = cls(n_rows=n)
    n_avail = t.num_examples() if hasattr(t, "num_examples") else len(t)
    for i in range(min(n, n_avail)):
        conv = t.get_example(i)
        for where, text in iter_prose_targets(conv):
            c = text.count(BS)
            if c:
                hits.append((f"{name}#{i}{where}", c, text[:100]))
    return hits


def main():
    from tasks import harness_basics, harness_replay, harness_poison

    all_hits = []
    for cls, nm in ((harness_basics.HarnessBasics, "harness_basics"),
                    (harness_replay.HarnessReplay, "harness_replay"),
                    (harness_poison.HarnessPoison, "harness_poison")):
        hits = check_task(cls, nm)
        print(f"  {nm}: {len(hits)} offending row(s)")
        all_hits += hits

    # also check the static pad templates directly
    for i, pad in enumerate(harness_basics._LONG_ANSWER_PADS):
        if pad.count(BS):
            all_hits.append((f"_LONG_ANSWER_PADS[{i}]", pad.count(BS), pad[:100]))

    if all_hits:
        print("\nFAIL: literal backslash-n found in prose targets:")
        for w, n, s in all_hits[:30]:
            print(f"   {w}: {n}x  {s!r}")
        print("\nFix: use a real newline in the Python string literal "
              "(\\n), not an escaped backslash (\\\\n).")
        return 1
    print("\nPASS: no literal backslash-n in any prose target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
