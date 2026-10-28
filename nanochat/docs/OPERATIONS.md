# utak — operational notes for a shared node

**Measured 2026-10-25 on `lambda3-hyperplane` (8x A100-SXM4-40GB, PCIe, no NVLink).**

This file records what the measurements actually showed, because two of
the results are counter-intuitive and one of them cost the run a day.

## 1. The resume is NOT slow. The node is.

The live 2-GPU run does 1,200 ms/step. The 2-GPU reference measured on
2026-10-22 (an idle node) was 965 ms/step. That looks like a 24%
regression in the resume path, and it is not.

Controlled A/B, same two cards, same code, back to back, depth 20 so
both probes fit alongside another tenant's jobs:

| run | ms/step |
|---|---|
| 2 GPUs, FRESH | 652 |
| 2 GPUs, RESUMED from a 2-rank checkpoint | 666 |

**The resume costs 2 %.** Everything else measured equal as well:

| component | fresh | resumed |
|---|---|---|
| model fwd+bwd (eager) | 156.5 ms | 156.5 ms |
| model fwd+bwd (compiled) | 161 ms | 180 ms |
| `optimizer.step()` | 298.44 ms | 298.43 ms |
| dataloader, per batch | 0.8 ms | 0.6 ms |
| parameter dtypes changed on resume | - | 0 |
| optimizer param groups | 10, same order | 10, same order |
| optimizer state keys | 25, `Parameter`-keyed | 25, `Parameter`-keyed |
| dynamo recompiles | 30 (warmup) | 30 (warmup) |

The remaining 22 % is the node: on 2026-10-22 the reference probe ran
with every other card idle. Now the neighbouring cards carry another
tenant's jobs.

**The one GPU case is at reference speed today** (1,854 ms measured vs
1,870 ms recorded on 2026-10-22) because one GPU does no collectives.

**Conclusion: do not "optimise" the resume path. The 24 % is contention.**

## 2. Pick GPU pairs on the SAME host bridge

`nvidia-smi topo -m` on this node:

```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0      X   PXB  NODE  NODE   SYS   SYS   SYS   SYS
GPU1    PXB    X  NODE  NODE   SYS   SYS   SYS   SYS
GPU2   NODE  NODE    X   PXB   SYS   SYS   SYS   SYS
GPU3   NODE  NODE  PXB    X   SYS   SYS   SYS   SYS
GPU4    SYS   SYS   SYS   SYS    X   PXB  NODE  NODE
...
```

`NODE` = same NUMA node, `SYS` = crosses the interconnect between the
two sockets. A 2-GPU run on **GPUs 0 and 2** (`NODE`, one bridge) is
fine. The same run on **GPUs 3 and 4** is a `SYS` pair -- one hop
across the socket interconnect -- and it measured **2,069 ms/step
against 652 ms** for the `NODE` pair: a **4x** penalty for a pair that
is otherwise identical.

**Rule: for a multi-GPU run, choose cards whose mutual link is `NODE`
or `NV#`, never `SYS`.** On this node that means the pairs
`0,2` `0,3` `1,2` `1,3` (bridge 0) and `4,6` `4,7` `5,6` `5,7` (bridge 1).

The recipe's default pair is `0,2`, which is correct. The `pick_free_gpus`
helper in `recipe/config.sh` does **not** check this: it returns the
lowest-indexed free cards, which can be a `SYS` pair.

## 3. CPU pinning helps only when the cards are quiet

Pinning the training ranks to the NUMA node that owns their GPUs
(`taskset -c 0-63` for GPUs 0-3) took a 2-GPU probe from 652 ms to
**523 ms** -- a 20 % gain -- on cards whose neighbours were only
lightly loaded.

Applied to the live run it gained nothing (1,200 ms -> 1,216 ms),
because that run's cards are already contended at the PCIe level.

**Rule: pin to the owning NUMA node; expect the gain only when the
neighbouring cards are quiet.**

## 4. What the node's CPU load does and does not do

The node runs at load 240-320 on 128 cores while another tenant's jobs
run. Three jobs in that tenant each hold **8,593 threads** and 7.2 GB
RSS, and together they demand about 128 cores.

Measured effects on our run:

* `renice` on the competing jobs (including every thread, not just the
  leader): **no change** to our step time (1,205 -> 1,200 ms).
* `SIGSTOP` on the competing jobs: **1.8 %** (1,205 -> 1,184 ms). Note
  that `SIGSTOP` does not stop their GPU work, so this is not a fair
  test of GPU contention -- but it does show the CPU side is not the
  bottleneck.
* The dataloader needs **1.3 ms** per batch against a 1,200 ms step.

**Conclusion: the CPU is not the bottleneck. The PCIe fabric is.**
The 2-GPU run moves roughly 5.6 GB per step per rank through the host
bridge (16 large AdamW parameters, each a `reduce_scatter` plus an
`all_gather`), and the neighbouring cards are pushing their own traffic
through the same bridge.

## 5. The node reboot, and what to do about it

On 2026-10-25 15:48 the node rebooted. The pretrain was at step
205,000 of 277,760 and the watcher died with it, so nothing restarted
the run: seven GPUs sat idle for twenty minutes.

A watcher cannot survive a reboot, because a reboot kills the watcher
too. The thing that has to survive it is configuration:

* systemd `--user` is **not** available on this node (no user bus,
  `Linger=no`),
* `/etc/cron.d` is **not** writable,
* the user crontab **is** available and cron **is** running.

So the mechanism is a user `@reboot` entry:

    @reboot /bin/bash /data/rowel/utak_smoke/utak_reboot_resume.sh

`recipe/ops/utak_reboot_resume.sh` is that script. It refuses to start
on a faulted driver, and refuses to start a second pipeline.

## 6. The 2026-10-24 driver fault, and the CPU fallback

The node's NVIDIA driver faulted at 2026-10-24 23:27: all 8 A100s
reported *"Unable to determine the device handle ... Unknown Error"*.
`torch.cuda.is_available()` returned False, the trainer silently fell
back to the CPU, and the run spent **15.6 hours** grinding through the
remaining 72,760 steps on 128 CPU cores at a speed that would have
taken about six months, printing not one step line, while the supervisor
watched a live pid and saw nothing wrong.

Four defects, fixed in commit `c222ea3`:

1. `autodetect_device_type()` treated "no CUDA" and "broken driver" as
   the same thing. It now distinguishes them and dies with a diagnosis.
2. `gpu_free_mib()` only checked for empty `nvidia-smi` output, but a
   faulted driver prints the literal string *"No devices were found"*
   to stdout and exits 6. It now requires both values to be integers.
3. Nothing checked driver health before starting a stage.
4. The supervisor had no stall detection -- it only asked "is the pid
   alive?".

## 7. Choosing the cards: a recipe-level check

`pick_free_gpus` in `recipe/config.sh` should not return a `SYS` pair.
The topology check is one `nvidia-smi topo -m` parse; without it, a
free-node moment can silently cost 4x.

## 8. The disk guard, and the hardlink trap

`recipe/prune_checkpoints.sh` is the only thing between this run and
ENOSPC, and it has a trap that the obvious test does not catch.

The recipe stages a checkpoint tree with `cp -al`, which makes **hard
links**: the staged directory and the canonical directory share one
inode. Deleting the model file from the canonical side therefore frees
**nothing** — the bytes stay on disk because the staged twin still holds
the inode. Measured on 2026-10-28 with a scratch tree of 4 × 20 MB
files: the pruner reported `pruned 3 files (57 MB)`, the pair still used
the same 78,140 KB, and the twin's file still had its full 20,000,000
bytes. `rm` succeeded; the bytes did not go.

Two things follow, and both are now done:

1. The pruner removes the **name** from every tree that holds the same
   inode, so the space actually comes back. It only ever removes a name
   that points at an inode it is already deleting, so it cannot destroy
   a checkpoint a later stage still needs: the inode *is* the file.
2. The watcher's prune list must include the **staged** trees, not only
   the canonical ones. Without them the pruner frees nothing at all.

`tests/test_prune_checkpoints.sh` covers the hardlink case, the keep-N
rule, the minimum age, the metadata, and the live tree.

**How much it matters.** `base_checkpoints/utak-base` alone was 736 GB:
54 pretrain checkpoints, none of which any stage after stage 2 needs.
Pruning it to the newest 3 reclaimed **711 GB** and took `/data` from
89 % to 84 %. The resume point (step 277,760) is in the kept set.

## References

* `recipe/ops/start_pipeline.sh` -- resume from the stage contract
* `recipe/ops/utak_reboot_resume.sh` -- the `@reboot` entry
* `recipe/ops/watch_utak.sh` -- the supervisor
* `recipe/resume_point.sh` -- the stage contract
* `recipe/prune_checkpoints.sh` -- the disk guard
* `docs/ETA_2GPU.md` -- the stage-hour table
* `recipe/logs/stage1_pretrain.log` -- the live run
