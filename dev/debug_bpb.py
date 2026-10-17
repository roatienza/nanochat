"""Debug: print every line containing 'token_bytes' with repr."""
p = '/data/rowel/nanochat/scripts/base_train_long.py'
import os
print("cwd:", os.getcwd())
print("exists:", os.path.exists(p))
lines = open(p).readlines()
print("total lines:", len(lines))
for i, l in enumerate(lines):
    if "token_bytes" in l and i > 100:
        print(i + 1, repr(l))