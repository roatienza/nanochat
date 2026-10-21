"""
Regression test for the compile-before-def ordering bug.

scripts/chat_sft_long.py compiled its per-block inner function with

    _group_forward = torch.compile(_group_forward, dynamic=False)

placed ABOVE the `def _group_forward(...)` it refers to. That raises

    NameError: name '_group_forward' is not defined

at import time, so the script died before it trained anything. The
stage-7b/8/9 runs that produced the shipped checkpoints predate the
compile change, which is why nobody hit it until the recipe was
actually run.

The sibling script scripts/base_train_long.py has the same block, in the
correct order (def first, compile second). Both are checked here, so a
future refactor cannot silently reintroduce the bug in either one.
"""
import ast
import os

import pytest

NANOCHAT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(NANOCHAT_ROOT, "scripts")

# The scripts that compile a module-level function, and the name.
COMPILE_TARGETS = [
    ("base_train_long.py", "_group_forward"),
    ("chat_sft_long.py", "_group_forward"),
]


def _def_node(tree, name):
    """The module-level `def name(...)` node, or None."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _compile_block(tree, name):
    """The module-level `if` block that assigns `name = torch.compile(name)`.

    The assignment sits inside an `if _COMPILE:` block, so we look at the
    module body AND at the bodies of module-level `if` statements.
    """
    found = []

    def scan(body):
        for node in body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == name:
                        found.append(node.lineno)
            elif isinstance(node, ast.If):
                scan(node.body)
                scan(node.orelse)

    scan(tree.body)
    return min(found) if found else None


def _compile_if_node(tree, name):
    """The module-level `if` statement whose body assigns `name`."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        for sub in node.body:
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == name:
                        return node
    return None


@pytest.mark.parametrize("filename,func", COMPILE_TARGETS)
def test_compile_comes_after_def(filename, func):
    """The torch.compile assignment must come AFTER the def it wraps."""
    path = os.path.join(SCRIPTS, filename)
    assert os.path.exists(path), f"{path} not found"

    tree = ast.parse(open(path, encoding="utf-8").read())

    def_node = _def_node(tree, func)
    assert def_node is not None, f"{filename}: no module-level def {func}"

    assign_line = _compile_block(tree, func)
    assert assign_line is not None, f"{filename}: no assignment to {func}"

    assert def_node.lineno < assign_line, (
        f"{filename}: the assignment to {func} is at line {assign_line} but its "
        f"def is at line {def_node.lineno}. The torch.compile() call would "
        f"raise NameError at import time."
    )
