"""
One-off driver: replicate the in-training ChatCORE eval loop (scripts/chat_sft.py:577-606)
exactly, on a given checkpoint. Used to fill the missing Stage-3-final ChatCORE number.

In-training settings being mirrored:
  - all_tasks  = ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval
  - categorical (ARC-Easy, ARC-Challenge, MMLU): max_problems = None  (full test set, chatcore_max_cat=-1)
  - generative (GSM8K, HumanEval):               max_problems = 24    (chatcore_max_sample=24)
  - batch_size = device_batch_size (2), num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50
  - ONE Engine, reused across all tasks, built on the uncompiled (orig) model
  - ChatCORE = centered_mean(all_tasks); ChatCORE_cat = centered_mean(categorical_tasks)
"""
import os, sys, time
sys.path.insert(0, "/data/rowel/nanochat")

from nanochat.common import compute_init, compute_cleanup, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

MODEL_TAG = os.environ.get("MODEL_TAG", "d28b-v2")
STEP      = int(os.environ.get("STEP", "15028"))

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=MODEL_TAG, step=STEP)
print(f"[driver] loaded {MODEL_TAG} step {STEP} from {meta.get('val_bpb')}", flush=True)

model.eval()
engine = Engine(model, tokenizer)  # uncompiled model, single engine reused (matches in-training)

all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval']
categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
baseline_accuracies = {'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25, 'GSM8K': 0.0, 'HumanEval': 0.0}
chatcore_max_cat = -1    # full test set
chatcore_max_sample = 24

task_results = {}
for task_name in all_tasks:
    t0 = time.time()
    limit = chatcore_max_cat if task_name in categorical_tasks else chatcore_max_sample
    max_problems = None if limit < 0 else limit
    acc = run_chat_eval(task_name, model, tokenizer, engine,
                        batch_size=2, max_problems=max_problems)
    task_results[task_name] = acc
    print(f"[driver] {task_name}: {100*acc:.2f}%  ({time.time()-t0:.1f}s, max_problems={max_problems})", flush=True)

def centered_mean(tasks):
    return sum((task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)

chatcore = centered_mean(all_tasks)
chatcore_cat = centered_mean(categorical_tasks)
print("=" * 60, flush=True)
print(f"[driver] ChatCORE: {chatcore:.4f}", flush=True)
print(f"[driver] ChatCORE_cat: {chatcore_cat:.4f}", flush=True)
for t in all_tasks:
    print(f"[driver]   {t}: {100*task_results[t]:.2f}%", flush=True)
print("=" * 60, flush=True)
compute_cleanup()
