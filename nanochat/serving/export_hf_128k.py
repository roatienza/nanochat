#!/usr/bin/env python
"""Export the Stage 7 long-context checkpoint to a vLLM model dir.

Adapted from export_hf.py (the 2048 rl100 exporter). Differences:
- source: base_checkpoints/d28b-128k (Stage 7a) or chatsft_checkpoints/
  d28b-128k-sft (Stage 7b), chosen by --phase/--tag/--step
- config: sequence_len/max_position_embeddings 131072, short_window 512
  (S layers keep the trained window; the vLLM plugin reads it), rope_base
  100000, torch_dtype bfloat16 (serving dtype)

Usage:
  python3 export_hf_128k.py --phase base --tag d28b-128k [--step 2000] \
      [--out /data/rowel/nanochat_store/v2/vllm_d28b/nanochat-d28b-rl100-128k]
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--phase", default="base", choices=["base", "sft", "rl"])
ap.add_argument("--tag", default="d28b-128k")
ap.add_argument("--step", type=int, default=None, help="default: last model_*.pt in the dir")
ap.add_argument("--base-dir", default=os.environ.get("NANOCHAT_BASE_DIR", "/data/rowel/nanochat_store/v2"))
ap.add_argument("--out", default=os.path.join(os.environ.get("NANOCHAT_BASE_DIR", "/data/rowel/nanochat_store/v2"), "vllm_d28b", "nanochat-d28b-rl100-128k"))
a = ap.parse_args()

phase_dir = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[a.phase]
src_dir = os.path.join(a.base_dir, phase_dir, a.tag)
if a.step is None:
    steps = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(src_dir) if f.startswith("model_")]
    a.step = max(steps)
ckpt_path = os.path.join(src_dir, f"model_{a.step:06d}.pt")
meta_path = os.path.join(src_dir, f"meta_{a.step:06d}.json")
print(f"source: {ckpt_path}")

with open(meta_path) as f:
    meta = json.load(f)
mc = meta["model_config"]
print("model_config:", mc)

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
assert isinstance(ckpt, dict), type(ckpt)
ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}

expected_nonlayer = {
    "resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda",
    "transformer.wte.weight", "lm_head.weight", "smear_gate.weight",
}
keys = list(ckpt.keys())
nonlayer = [k for k in keys if not k.startswith("transformer.h.")]
missing = expected_nonlayer - set(nonlayer)
assert not missing, f"unexpected checkpoint layout, missing {missing}"

tensors = {}
for k, v in ckpt.items():
    assert isinstance(v, torch.Tensor), (k, type(v))
    tensors[k] = v.contiguous().to(torch.float32)

n = sum(t.numel() for t in tensors.values())
print(f"tensors: {len(tensors)}, total params: {n / 1e9:.3f}B")

os.makedirs(a.out, exist_ok=True)
save_file(tensors, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})

config = {
    "model_type": "nanochat_d28",
    "architectures": ["NanochatD28ForCausalLM"],
    "auto_map": {
        "AutoConfig": "configuration_nanochat.NanochatD28Config",
        "AutoModelForCausalLM": "modeling_nanochat_d28.NanochatD28ForCausalLM",
    },
    "sequence_len": mc["sequence_len"],
    "max_position_embeddings": mc["sequence_len"],
    "vocab_size": mc["vocab_size"],
    "n_layer": mc["n_layer"],
    "n_head": mc["n_head"],
    "n_kv_head": mc["n_kv_head"],
    "n_embd": mc["n_embd"],
    "window_pattern": mc["window_pattern"],
    "short_window": mc.get("short_window", 512),
    "rotary_base": mc.get("rope_base", 100000),
    "qk_scale": 1.2,
    "logit_softcap": 15.0,
    "mlp_hidden_mult": 4,
    "ve_gate_channels": 12,
    "backout_layer_div": 2,
    "torch_dtype": "bfloat16",
    "bos_token_id": 32759,
    "eos_token_id": 32763,
    "transformers_version": "4.0.0",
}
with open(os.path.join(a.out, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print("wrote", a.out)
print("next: copy tokenizer.json / tokenizer_config.json / chat_template.jinja and the")
print("plugin files from nanochat-d28b-rl100/ (they are model-independent), then serve with")
print("--max-model-len 131072 --max-num-batched-tokens 131072 --max-num-seqs 1")