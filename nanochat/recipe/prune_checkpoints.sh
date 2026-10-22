#!/usr/bin/env bash
# Keep only the newest checkpoints in a directory.
#
# WHY THIS EXISTS
# ---------------
# One checkpoint set is 14.63 GB: a 6.31 GB model file plus two 4.16 GB
# optimizer shards. Stage 1 saves one every 5,000 steps, and the pretrain is
# 277,760 steps, so stage 1 alone writes 56 of them -- 819 GB. The whole
# pipeline projects to 1.68 TB, on a filesystem with 4.0 TB free that a
# second tenant is also using.
#
# The ORIGINAL run did prune. base_checkpoints/d28b-v2 holds 61 files: 56
# meta files and exactly ONE model file plus its 4 optimizer shards. So the
# original kept the tiny metadata and deleted the big weights. This script
# does the same: the metas are 1.4 KB each and are the provenance record
# of what the run did, so they are always kept.
#
# The resume logic needs the NEWEST checkpoint (recipe/01_pretrain.sh picks
# the newest model_*.pt, and recipe/resume_point.sh needs the final one),
# so the pruner keeps the newest few sets and deletes the rest.
#
# SAFETY
#   - it only deletes model_*.pt and optim_*_rank*.pt; metas are kept;
#   - a file must be older than min_age_seconds, so it can never delete a
#     file that torch.save is still writing;
#   - the newest set is never deleted, whatever the settings.
#
# Usage:
#   ./recipe/prune_checkpoints.sh <dir> [keep_n] [min_age_seconds]
#
#   keep_n          how many of the newest checkpoint sets to keep (default 3)
#   min_age_seconds how old a file must be before it may be deleted
#                   (default 600)
set -u

DIR="${1:-}"
KEEP="${2:-3}"
MIN_AGE="${3:-600}"

if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
  echo "usage: $0 <dir> [keep_n] [min_age_seconds]" >&2
  exit 2
fi

now=$(date +%s)

# The checkpoint steps, NEWEST first.
steps=$(ls -1 "$DIR"/model_*.pt 2>/dev/null \
        | sed -n 's/.*model_\([0-9]\{6\}\)\.pt/\1/p' | sort -rn)
[ -n "$steps" ] || { echo "nothing to prune in $DIR"; exit 0; }

total=$(echo "$steps" | wc -l)
if [ "$total" -le "$KEEP" ]; then
  echo "nothing to prune in $DIR ($total checkpoints, keeping $KEEP)"
  exit 0
fi

# The steps to DELETE: everything below the newest $KEEP.
# rank is 1 for the newest, 2 for the next, ...
rank=0
freed=0
removed=0

for step in $steps; do
  rank=$((rank + 1))
  [ "$rank" -le "$KEEP" ] && continue

  for f in "$DIR"/model_"$step".pt "$DIR"/optim_"$step"_rank*.pt; do
    [ -e "$f" ] || continue
    mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    age=$((now - mtime))
    [ "$age" -lt "$MIN_AGE" ] && continue
    size=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if rm -f "$f"; then
      removed=$((removed + 1))
      freed=$((freed + size))
    fi
  done
done

if [ "$removed" -gt 0 ]; then
  printf 'pruned %d files (%d MB) from %s\n' "$removed" "$((freed / 1024 / 1024))" "$DIR"
else
  echo "nothing to prune in $DIR ($total checkpoints, keeping $KEEP)"
fi
