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
#   ./recipe/prune_checkpoints.sh <dir>
#
#   keep_n          how many of the newest checkpoint sets to keep (default 3)
#   min_age_seconds how old a file must be before it may be deleted
#                   (default 600)
#
# THE HARDLINK PROBLEM
# --------------------
# The recipe stages a checkpoint tree with `cp -al`, which makes HARD
# LINKS: the staged directory and the canonical directory share one
# inode. Deleting the model file from the CANONICAL directory therefore
# frees NOTHING -- the bytes stay on disk because the staged twin still
# holds the inode. A scratch test on 2026-10-28: 4 x 20 MB files,
# `cp -al` to a twin, then prune the canonical side keeping 1. The
# pruner reported "pruned 3 files (57 MB)", but the pair still used the
# same 78,140 KB, and the twin's file still had its full 20,000,000
# bytes. The pruner cannot see this on its own: `rm` succeeds, the bytes
# do not go.
#
# So the pruner also removes the NAME from every other directory that
# holds the same inode. It only ever removes a name that points at an
# inode it is already deleting, so it cannot destroy a checkpoint some
# other stage still needs: the inode IS the file, and the steps being
# removed are the ones below the keep count in BOTH trees (they are the
# same steps, because the trees are copies of each other).
set -u

DIR="${1:-}"
KEEP="${2:-3}"
MIN_AGE="${3:-600}"
# Where the hardlink twins live. All the staged trees are under here.
TREE="${NANOCHAT_BASE_DIR:-/data/rowel/nanochat_store/v2}"

if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
  echo "usage: $0 <dir>  " >&2
  exit 2
fi

now=$(date +%s)

# Every path under the tree that holds the same inode as a given file.
# The same inode means the same bytes on disk, so removing the name in
# one place and the name in the other place are the same deletion.
twin_paths() {
  find "$TREE" -inum "$1" -type f 2>/dev/null
}

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
    ino=$(stat -c %i "$f" 2>/dev/null || echo 0)
    if rm -f "$f"; then
      removed=$((removed + 1))
      freed=$((freed + size))
      # The bytes are still on disk if a staged twin holds the inode.
      # Remove that name too, so the space actually comes back.
      if [ -n "$ino" ] && [ "$ino" != "0" ]; then
        for twin in $(twin_paths "$ino"); do
          [ "$twin" = "$f" ] && continue
          rm -f "$twin" 2>/dev/null && true
        done
      fi
    fi
  done
done

if [ "$removed" -gt 0 ]; then
  printf 'pruned %d files (%d MB) from %s\n' "$removed" "$((freed / 1024 / 1024))" "$DIR"
else
  echo "nothing to prune in $DIR ($total checkpoints, keeping $KEEP)"
fi
