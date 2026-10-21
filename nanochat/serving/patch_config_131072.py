import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
changed = {}
for k in ("max_position_embeddings", "model_max_length", "max_seq_len", "sequence_len"):
    if k in cfg and cfg[k] != 131072:
        cfg[k] = 131072
        changed[k] = 131072
json.dump(cfg, open(p, "w"), indent=2)
print("config patched:", changed if changed else "already 131072")
