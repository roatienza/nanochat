"""Rewrite line 240 (0-based 239) of base_train_long.py unconditionally.

No conditionals, no subscripts in the source below except via chr()-built
strings assigned through a helper, so nothing can be stripped in transit.
"""
p = '/data/rowel/nanochat/scripts/base_train_long.py'
f = open(p)
lines = f.readlines()
f.close()
i = 239
cur = lines[i].strip()
assert cur == "nb = token_bytes", repr(cur)
LB = chr(91)
RB = chr(93)
idx_expr = "vy_safe"
new_line = "                nb = token_bytes" + LB + idx_expr + RB + "\n"
lines[i] = new_line
g = open(p, "w")
g.writelines(lines)
g.close()
import ast
ast.parse("".join(lines))
print("line 240 rewritten to:", repr(new_line))
print("syntax OK")