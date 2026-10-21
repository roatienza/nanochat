"""Smoke-test the Stage 6 env-design pattern against nanochat's sandbox.

The model writes a function body; the harness appends hidden asserts and
executes the whole program with nanochat.execution.execute_code.
Reward = result.success. This is the HumanEval pattern applied to
synthetic, curriculum-able tasks.
"""
import sys
sys.path.insert(0, "/data/rowel/nanochat")

from nanochat.execution import execute_code

# --- pattern 1: model writes a function, harness appends asserts ---
code = '''
def solve(s):
    return s.count("a")
'''
test = '''
assert solve("banana") == 3
assert solve("") == 0
print("OK")
'''
r = execute_code(code + test)
print("pattern1 (correct fn):", r.success, repr(r.stdout), repr(r.error))

# --- pattern 2: wrong function must fail ---
code_bad = '''
def solve(s):
    return len(s)
'''
r2 = execute_code(code_bad + test)
print("pattern2 (wrong fn):  ", r2.success, repr(r2.error))

# --- pattern 3: syntax error must fail cleanly ---
r3 = execute_code("def solve(:\n  pass\n")
print("pattern3 (syntax err):", r3.success, repr(r3.error))

# --- pattern 4: model emits prose around the code (extract like HumanEval) ---
import re
completion = "Here is my solution:\n```python\ndef solve(s):\n    return s.count(\"a\")\n```\nDone."
m = re.findall(r"```(?:python)?\s*\n(.*?)\n```", completion, re.DOTALL)
extracted = m[0].strip() if m else completion
r4 = execute_code(extracted + test)
print("pattern4 (extract):   ", r4.success, repr(r4.stdout))

# --- pattern 5: multi-turn style — model uses the calculator tool loop ---
# (what the Engine already forces; verify use_calculator's reach)
from nanochat.engine import use_calculator
print("pattern5 calc 2+2*10:", use_calculator("2+2*10"))
print("pattern5 calc 2**10 :", use_calculator("**" in "2**10" and "2**10" or "2+3"))
print("pattern5 count      :", use_calculator("'banana'.count('a')"))