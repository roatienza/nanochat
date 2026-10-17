"""Debug the fixer's loop: print candidates near line 240."""
p = '/data/rowel/nanochat/scripts/base_train_long.py'
lines = open(p).readlines()
print("len:", len(lines))
for i in range(235, 245):
    l = lines[i]
    s = l.strip()
    print(i, repr(s), s.startswith("nb = token_bytes"), i > 200)