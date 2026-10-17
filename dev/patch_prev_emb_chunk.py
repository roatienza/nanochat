"""Patch _prev_embeddings (lines 343-344, 0-based 342/343) to chunked form.

Replaces the two lines unconditionally (verified visually: they are
    cur_norm = rms_norm(self.wte(input_ids))
    buf[cur_slot][valid] = cur_norm[valid].to(buf.dtype)
The transport strips bracketed subscripts, so no content asserts.
"""
p = '/data/rowel/nanochat_store/v2/vllm_d28b/nanochat_vllm_plugin/modeling_nanochat_d28.py'
lines = open(p).readlines()
i_norm, i_write = 342, 343
assert "cur_norm = rms_norm(self.wte(input_ids))" in lines[i_norm], repr(lines[i_norm])

LB, RB = chr(91), chr(93)
indent = lines[i_norm][:len(lines[i_norm]) - len(lines[i_norm].lstrip())]
new_block = (
    indent + "# Chunked: rms_norm in fp32 over 107k+ tokens is a 0.77 GB fp32\n"
    + indent + "# copy per tensor; with the KV cache sized for 131k there is no\n"
    + indent + "# headroom, so compute + write in 16384-token slices.\n"
    + indent + "M = 16384\n"
    + indent + "for s in range(0, num_tokens, M):\n"
    + indent + "    e = min(s + M, num_tokens)\n"
    + indent + "    cur_norm = rms_norm(self.wte(input_ids" + LB + "s:e" + RB + "))\n"
    + indent + "    buf" + LB + "cur_slot" + LB + "s:e" + RB + RB + " = cur_norm.to(buf.dtype)\n"
)
lines[i_norm:i_write + 1] = [new_block]
open(p, 'w').writelines(lines)
import ast
ast.parse("".join(lines))
print("patched _prev_embeddings, syntax OK")