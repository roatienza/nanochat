"""
Multi-step agentic SFT task on the DEPLOYED tool surface (Stage 9g).

Why this file exists
--------------------
The ladder trains agentic behaviour in stages 5-6 and then overwrites it in
stages 8-9. Worse, what stages 5-6 train is NOT the deployed capability:

  * `tasks/agentic.py` emits ONE user turn and an empty assistant turn, and the
    reward is `run_hidden_tests` on a generated Python function. That is
    single-turn code generation, not tool use.
  * stages 8-9 contain no multi-turn tool rows except `toolfollowup`, which is
    hermes replay of call -> result -> prose. Nothing in the ladder requires a
    SECOND call whose arguments depend on the FIRST result.

So the model never learns the loop the harness actually runs. Measured on the
deployed checkpoint: a request needing two dependent calls produces one call and
then either stops or answers from imagination.

This task generates the missing shape: verifiable multi-step episodes on the
EXACT 8 tools the deployed agent exposes (baby_onit.py:749-766):

    web_search, fetch_content, bash, write_file, read_file, edit_file,
    local_search, grep

Not the long-context task's get_weather/search_web/send_email/create_ticket,
which the deployed agent does not have.

Design decisions that matter
----------------------------
1) Every episode runs against a VIRTUAL filesystem (`Sandbox`), so an instance
   is reproducible from its seed and needs no disk, network or subprocess. The
   same Sandbox grades a model's trajectory, which is why this file serves the
   SFT task, the eval gate and (later) the RL reward with one implementation.

2) The answer is a function of SANDBOX STATE the question does not reveal. A
   single call cannot produce it: either the second call's arguments are the
   first call's output (dependent chain), or the answer requires combining
   values from several files. This is what makes the row unlearnable by
   pattern and the gate meaningful.

3) Conversation shape is byte-identical to `tasks/toolfollowup.py`, the shape
   the serving template renders: tools block in the first USER turn (so the
   loss never sees the schemas), one `CALL name {json}` per assistant turn,
   `Tool result: ...` as the following user turn, final prose answer last.

4) `bash` is a closed subset (ls, cat, wc, grep -c). Anything else returns an
   error string, which is itself a useful training signal: the model must
   recover rather than assume.
"""
import json
import random
import re

from tasks.common import Task
from tasks.harness_replay import _HARNESS_TOOLS
from tasks.toolcalling import render_tools_prompt

TOOL_NAMES = [t["function"]["name"] for t in _HARNESS_TOOLS]

# ---------------------------------------------------------------------------
# Virtual filesystem
# ---------------------------------------------------------------------------

_NAMES = ["report", "notes", "ledger", "config", "summary", "draft", "memo",
          "inventory", "schedule", "budget", "manifest", "changelog"]
_EXTS = ["md", "txt", "yaml", "csv"]
_DIRS = ["", "docs/", "data/", "src/", "logs/"]


class Sandbox:
    """A tiny in-memory filesystem that implements the deployed tool surface.

    Only the file-touching tools are modelled. `web_search` and
    `fetch_content` are deliberately absent: episodes must be offline and
    deterministic, so a task never requires them and `call()` returns an
    error if the model tries (a recoverable dead end, not a crash).
    """

    def __init__(self, files):
        self.files = dict(files)          # path -> text
        self.calls = []                   # [(name, args)] in order
        self.dead_ends = 0

    # -- tools ------------------------------------------------------------
    def read_file(self, path):
        if path not in self.files:
            return f"Error: no such file: {path}"
        return self.files[path]

    def write_file(self, path, content, mode="write"):
        if mode == "append":
            self.files[path] = self.files.get(path, "") + content
        else:
            self.files[path] = content
        return f"Wrote {len(content)} chars to {path}"

    def edit_file(self, path, old_string, new_string):
        if path not in self.files:
            return f"Error: no such file: {path}"
        body = self.files[path]
        if old_string not in body:
            return f"Error: old_string not found in {path}"
        self.files[path] = body.replace(old_string, new_string, 1)
        return f"Replaced 1 occurrence in {path}"

    def grep(self, pattern):
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Error: bad regex: {e}"
        hits = []
        for path, body in sorted(self.files.items()):
            for n, line in enumerate(body.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{path}:{n}: {line.strip()}")
        return "\n".join(hits) if hits else "No matches."

    def local_search(self, query):
        terms = [t for t in re.split(r"\W+", query.lower()) if t]
        scored = []
        for path, body in sorted(self.files.items()):
            low = body.lower()
            score = sum(low.count(t) for t in terms)
            if score:
                scored.append((score, path, body))
        if not scored:
            return "No matches."
        scored.sort(key=lambda s: -s[0])
        top = scored[0]
        head = top[2].splitlines()[:3]
        return f"{top[1]}\n" + "\n".join(head)

    def bash(self, command):
        cmd = command.strip()
        if cmd in ("ls", "ls -la", "ls ."):
            return "\n".join(sorted(p for p in self.files if "/" not in p)) or "(empty)"
        m = re.fullmatch(r"ls\s+(\S+)", cmd)
        if m:
            d = m.group(1).rstrip("/") + "/"
            out = sorted(p for p in self.files if p.startswith(d))
            return "\n".join(out) if out else "(empty)"
        m = re.fullmatch(r"cat\s+(\S+)", cmd)
        if m:
            return self.read_file(m.group(1))
        m = re.fullmatch(r"wc\s+-l\s+(\S+)", cmd)
        if m:
            p = m.group(1)
            if p not in self.files:
                return f"wc: {p}: No such file or directory"
            return f"{len(self.files[p].splitlines())} {p}"
        m = re.fullmatch(r"grep\s+-c\s+(\S+)\s+(\S+)", cmd)
        if m:
            pat, p = m.group(1), m.group(2)
            if p not in self.files:
                return f"grep: {p}: No such file or directory"
            return str(sum(1 for line in self.files[p].splitlines() if re.search(pat, line)))
        return (f"Error: command not permitted in this environment: {command!r}. "
                "Permitted: ls, cat FILE, wc -l FILE, grep -c PATTERN FILE.")

    # -- dispatch ---------------------------------------------------------
    def call(self, name, args):
        """Execute one tool call the way the harness would. Never raises."""
        self.calls.append((name, dict(args)))
        try:
            if name == "read_file":
                return self.read_file(args.get("path", ""))
            if name == "write_file":
                return self.write_file(args.get("path", ""), args.get("content", ""),
                                       args.get("mode", "write"))
            if name == "edit_file":
                return self.edit_file(args.get("path", ""), args.get("old_string", ""),
                                      args.get("new_string", ""))
            if name == "grep":
                return self.grep(args.get("pattern", ""))
            if name == "local_search":
                return self.local_search(args.get("query", ""))
            if name == "bash":
                return self.bash(args.get("command", ""))
            if name in ("web_search", "fetch_content"):
                self.dead_ends += 1
                return ("Error: no network in this environment. Use the files in the "
                        "working directory.")
            self.dead_ends += 1
            return f"Error: unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}"
        except Exception as e:  # a malformed call must not kill the episode
            self.dead_ends += 1
            return f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Workspace generation
# ---------------------------------------------------------------------------

def _path(rng, used):
    while True:
        p = rng.choice(_DIRS) + rng.choice(_NAMES)
        if rng.random() < 0.6:
            p += "_" + str(rng.randint(1, 99))
        p += "." + rng.choice(_EXTS)
        if p not in used:
            used.add(p)
            return p


def _code(rng):
    return f"{rng.choice('ABCDEFGH')}{rng.choice('JKLMNP')}-{rng.randint(1000, 9999)}"


def _body(rng, n_lines, codes):
    out = []
    for _ in range(n_lines):
        out.append(f"line {len(out) + 1}: status recorded, reviewer noted, nothing flagged.")
    for code in codes:
        out.insert(rng.randrange(len(out) + 1), f"tracking {code} owner desk")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Episode families. Each returns (question, files, gold_calls, answer, check)
# where check(sandbox) -> bool verifies the final state.
# ---------------------------------------------------------------------------

def _fam_grep_then_read(rng):
    """grep locates the file holding a code; read_file gets its owner."""
    used = set()
    paths = [_path(rng, used) for _ in range(rng.randint(3, 6))]
    target = rng.choice(paths)
    code = _code(rng)
    owner = rng.choice(["alvarez", "okonkwo", "lindqvist", "haddad", "moreau",
                        "tanaka", "nwosu", "petrov", "silva", "abara"])
    files = {}
    for p in paths:
        codes = [code] if p == target else [_code(rng) for _ in range(rng.randint(1, 3))]
        files[p] = _body(rng, rng.randint(4, 12), codes)
        if p == target:
            files[p] += f"\ncase {code} owner {owner} closes friday\n"
    q = (f"Which owner is recorded for tracking code {code}? Find the file that "
         f"mentions the code, then read it. Answer with only the owner name.")
    return q, files, [("grep", {"pattern": code}), ("read_file", {"path": target})], owner


def _fam_list_then_read(rng):
    """bash ls a directory; read the one file whose name matches a constraint."""
    used = set()
    d = rng.choice(["docs/", "data/", "logs/", "src/"])
    n = rng.randint(3, 5)
    paths = []
    for _ in range(n):
        p = d + rng.choice(_NAMES) + "_" + str(rng.randint(1, 99)) + "." + rng.choice(_EXTS)
        if p not in used:
            used.add(p)
            paths.append(p)
    target = max(paths, key=lambda p: int(re.search(r"_(\d+)\.", p).group(1)))
    token = _code(rng)
    files = {}
    for p in paths:
        files[p] = _body(rng, rng.randint(4, 10), [_code(rng) for _ in range(2)])
    files[target] += f"\nshipment token {token} confirmed\n"
    q = (f"List the files in {d.rstrip('/')}/, then read the one with the highest "
         f"numeric suffix. What is the shipment token in that file? "
         f"Answer with only the token.")
    return q, files, [("bash", {"command": f"ls {d.rstrip('/')}"}),
                      ("read_file", {"path": target})], token


def _fam_edit_then_verify(rng):
    """read -> edit -> read again. The answer is the post-edit state.

    The old code appears EXACTLY ONCE: `edit_file` replaces one occurrence, so
    a file with several would leave the old code behind and the gold
    trajectory would not satisfy its own check (caught by build_episode).
    """
    used = set()
    path = _path(rng, used)
    old = _code(rng)
    new = _code(rng)
    while new == old:
        new = _code(rng)
    files = {path: _body(rng, rng.randint(5, 10), [_code(rng)]) +
                     f"\nactive code {old} pending review\n"}
    assert files[path].count(old) == 1, "old code must appear exactly once"
    q = (f"In {path}, replace the tracking code {old} with {new}, then read the "
         f"file back to confirm. Reply with the code that is now active.")
    def check(sb, path=path, old=old, new=new):
        body = sb.files.get(path, "")
        return old not in body and new in body
    return q, files, [("read_file", {"path": path}),
                      ("edit_file", {"path": path, "old_string": old, "new_string": new}),
                      ("read_file", {"path": path})], new, check


def _fam_sum_three(rng):
    """Three reads, then arithmetic. No single call yields the answer."""
    used = set()
    paths = [_path(rng, used) for _ in range(3)]
    vals = [rng.randint(11, 499) for _ in range(3)]
    files = {}
    for p, v in zip(paths, vals):
        files[p] = _body(rng, rng.randint(4, 9), [_code(rng)]) + f"\nquantity {v} units\n"
    q = ("Read " + ", ".join(paths) + ". Each file records a quantity. "
         "What is the total across all three? Answer with only the number.")
    return q, files, [("read_file", {"path": p}) for p in paths], str(sum(vals))


def _fam_search_then_read(rng):
    """local_search ranks a file; read it for the value."""
    used = set()
    paths = [_path(rng, used) for _ in range(rng.randint(3, 5))]
    topic = rng.choice(["hydraulic", "calibration", "inventory", "inspection",
                        "throughput", "clearance", "retention"])
    target = rng.choice(paths)
    val = rng.randint(100, 9999)
    files = {}
    for p in paths:
        files[p] = _body(rng, rng.randint(4, 9), [_code(rng)])
    files[target] = (f"{topic} {topic} {topic} audit worksheet\n" + files[target] +
                     f"\n{topic} reading {val} units\n")
    q = (f"Search the working directory for {topic}, then open the file that "
         f"discusses it most. What reading is recorded? Answer with only the number.")
    return q, files, [("local_search", {"query": topic}),
                      ("read_file", {"path": target})], str(val)


_FAMILIES = {
    "grep_then_read": _fam_grep_then_read,
    "list_then_read": _fam_list_then_read,
    "edit_then_verify": _fam_edit_then_verify,
    "sum_three": _fam_sum_three,
    "search_then_read": _fam_search_then_read,
}


# ---------------------------------------------------------------------------
# Conversation construction (toolfollowup shape)
# ---------------------------------------------------------------------------

def _render_call(name, args):
    return f"CALL {name} {json.dumps(args, separators=(',', ':'))}"


def build_episode(family, seed):
    """Return (messages, sandbox, answer, min_calls) for one instance."""
    rng = random.Random(f"{seed}:{family}")
    out = _FAMILIES[family](rng)
    if len(out) == 4:
        question, files, gold_calls, answer = out
        check = None
    else:
        question, files, gold_calls, answer, check = out

    sandbox = Sandbox(files)
    messages = [{"role": "user",
                 "content": question + "\n\n" + render_tools_prompt(_HARNESS_TOOLS)}]
    for name, args in gold_calls:
        result = sandbox.call(name, args)
        messages.append({"role": "assistant",
                         "content": f"<|python_start|>{_render_call(name, args)}<|python_end|>"})
        messages.append({"role": "user", "content": f"Tool result: {result}"})
    messages.append({"role": "assistant", "content": answer})

    # the gold trajectory must itself satisfy the family's check
    if check is not None and not check(sandbox):
        raise AssertionError(f"{family} gold trajectory failed its own check")
    return messages, sandbox, answer, len(gold_calls)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class MultiStep(Task):
    """Verifiable multi-step tool episodes on the deployed tool surface.

    Every instance requires at least two calls whose arguments are chained.
    """

    def __init__(self, families=None, n_rows=3000, seed=2027, **kwargs):
        super().__init__(**kwargs)
        self.families = tuple(families) if families else tuple(_FAMILIES)
        for f in self.families:
            assert f in _FAMILIES, f"unknown family {f!r}; have {sorted(_FAMILIES)}"
        self.n_rows = n_rows
        self.seed = seed

    def num_examples(self):
        return self.n_rows

    def _pick(self, index):
        return self.families[index % len(self.families)], f"{self.seed}:{index}"

    def get_example(self, index):
        family, seed = self._pick(index)
        messages, _, _, _ = build_episode(family, seed)
        return {"messages": messages}

    def gold(self, index):
        """(family, question, files, answer, min_calls) — for the eval gate."""
        family, seed = self._pick(index)
        messages, sandbox, answer, min_calls = build_episode(family, seed)
        return family, messages[0]["content"], dict(sandbox.files), answer, min_calls

    def evaluate(self, conversation, assistant_response):
        raise NotImplementedError("SFT data task; grading runs through eval/multistep_gate.py")
