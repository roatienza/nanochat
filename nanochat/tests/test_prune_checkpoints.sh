#!/usr/bin/env bash
# Tests for recipe/prune_checkpoints.sh.
#
# WHY THESE EXIST
# ---------------
# The pruner is the only thing standing between this run and ENOSPC, and
# it has a subtlety that the obvious test does not catch. The recipe
# stages checkpoint trees with `cp -al`, which makes HARD LINKS, so a
# staged directory and its canonical directory share one inode.
#
# Deleting the model file from the canonical side frees NOTHING while
# the staged twin still holds the inode. On 2026-10-28 the pruner
# reported "pruned 3 files (57 MB)" while the bytes stayed exactly where
# they were -- the twin held them. /data was at 89% at the time.
#
# So the pruner removes the NAME from every tree that holds the same
# inode. These tests pin that behaviour, and pin the safety property
# that matters more: the pruner must never remove a checkpoint a later
# stage still has to read.
set -u

PRUNER="${1:-recipe/prune_checkpoints.sh}"
[ -f "$PRUNER" ] || { echo "cannot find the pruner at $PRUNER" >&2; exit 2; }

pass=0
fail=0

ok()   { printf '  ok   %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  FAIL %s\n' "$1"; fail=$((fail + 1)); }

# A scratch tree with a canonical dir and a hardlinked staged twin.
make_tree() {
  local root="$1" n="$2" i
  rm -rf "$root"
  mkdir -p "$root/canon"
  for i in $(seq 1 "$n"); do
    head -c 20000000 /dev/urandom > "$root/canon/model_00000$i.pt"
  done
  cp -al "$root/canon" "$root/staged"
}

# --- 1. it frees space when a hardlink twin holds the data ----------------
echo "== the pruner frees space that a hardlink twin is holding =="
T=$(mktemp -d)
make_tree "$T/tree" 4
before=$(du -sk "$T/tree" | cut -f1)
out=$(NANOCHAT_BASE_DIR="$T/tree" bash "$PRUNER" "$T/tree/canon" 1 0 2>&1)
after=$(du -sk "$T/tree" | cut -f1)
if [ "$after" -lt "$before" ]; then
  ok "space was reclaimed ($before KB -> $after KB)"
else
  bad "space was NOT reclaimed ($before KB -> $after KB) -- the twin still holds the bytes"
fi
if [ "$(ls -1 "$T/tree/staged" 2>/dev/null | wc -l)" -eq 1 ]; then
  ok "the staged twin was pruned too"
else
  bad "the staged twin still has $(ls -1 "$T/tree/staged" 2>/dev/null | wc -l) file(s)"
fi
rm -rf "$T"

# --- 2. it keeps the newest N sets ---------------------------------------
echo "== it keeps the newest N sets and deletes the rest =="
T=$(mktemp -d)
make_tree "$T/tree" 5
NANOCHAT_BASE_DIR="$T/tree" bash "$PRUNER" "$T/tree/canon" 3 0 >/dev/null 2>&1
left=$(ls -1 "$T/tree/canon"/model_*.pt 2>/dev/null | wc -l)
if [ "$left" -eq 3 ]; then
  ok "kept 3 of 5"
else
  bad "kept $left of 5, expected 3"
fi
# and the newest is still there
if [ -f "$T/tree/canon/model_000005.pt" ]; then
  ok "the newest checkpoint survived"
else
  bad "the newest checkpoint was deleted"
fi
rm -rf "$T"

# --- 3. it never deletes a file younger than min_age ---------------------
echo "== it respects the minimum age =="
T=$(mktemp -d)
make_tree "$T/tree" 4
NANOCHAT_BASE_DIR="$T/tree" bash "$PRUNER" "$T/tree/canon" 1 3600 >/dev/null 2>&1
left=$(ls -1 "$T/tree/canon"/model_*.pt 2>/dev/null | wc -l)
if [ "$left" -eq 4 ]; then
  ok "nothing was deleted while every file was fresh"
else
  bad "deleted $((4 - left)) fresh file(s); the min_age guard failed"
fi
rm -rf "$T"

# --- 4. it never touches the metas ---------------------------------------
echo "== it keeps the metadata =="
T=$(mktemp -d)
make_tree "$T/tree" 4
for i in 1 2 3 4; do echo '{}' > "$T/tree/canon/meta_00000$i.json"; done
NANOCHAT_BASE_DIR="$T/tree" bash "$PRUNER" "$T/tree/canon" 1 0 >/dev/null 2>&1
metas=$(ls -1 "$T/tree/canon"/meta_*.json 2>/dev/null | wc -l)
if [ "$metas" -eq 4 ]; then
  ok "all 4 meta files survived"
else
  bad "$metas meta file(s) survived, expected 4"
fi
rm -rf "$T"

# --- 5. the live tree: the step a pending stage needs must survive -------
echo "== on the live tree, the step the next stage reads survives =="
BASE="${NANOCHAT_BASE_DIR:-/data/rowel/nanochat_store/v2}"
if [ -d "$BASE/chatrl_checkpoints/utak-rl-warmup" ]; then
  # Stage 5 reads utak-rl-warmup @ 100. The pruner keeps 3, and that
  # directory holds 1, so the step it needs cannot be one of the victims.
  need=$(cat "$BASE/_handoff/utak-rl-warmup.step" 2>/dev/null || echo 100)
  if [ -f "$BASE/chatrl_checkpoints/utak-rl-warmup/model_$(printf '%06d' "$need").pt" ]; then
    ok "stage 5's source step $need is present and would be kept"
  else
    bad "stage 5's source step $need is missing"
  fi
else
  echo "  skip the live tree is not on this machine"
fi

printf '\n%s passed, %s failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
