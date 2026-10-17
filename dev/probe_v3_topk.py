# Top-10 next tokens after the training prompt (target 889064), and after a
# fresh-code short prompt (target 424242). Run on GPU 7.
import os, sys, random, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.insert(0, "/data/rowel/nanochat")
from tasks.longctx_sft import LongContextSFT, _WORDS
from nanochat.tokenizer import RustBPETokenizer
from nanochat.checkpoint_manager import load_model_from_dir

tok = RustBPETokenizer.from_directory("/data/rowel/nanochat_store/tokenizer")
ckdir = "/data/rowel/nanochat_store/v2/chatsft_checkpoints"
model, t2, meta = load_model_from_dir(ckdir, device=torch.device("cuda"), phase="eval", model_tag="d28b-128k-sft-v3", step=400)
model.eval()

def show(prompt_ids, label, target):
    x = torch.tensor([list(prompt_ids)], dtype=torch.long, device="cuda")
    with torch.no_grad():
        logits = model(x)
        top10 = logits[0, -1].topk(10)
    parts = []
    for v, i in zip(top10.values.tolist(), top10.indices.tolist()):
        parts.append(f"{i}:{v:.2f}:{tok.decode([i])!r}")
    print(label, "target", repr(target))
    print("   ", " | ".join(parts))
    print("   target ids:", tok.encode(target))

ds = LongContextSFT()
conv = ds.get_example(0)
ids, mask = tok.render_conversation(conv, max_tokens=131072)
n_pre = len(ids) - sum(mask)
show(ids[:n_pre], "TRAIN niah L1 i0", "889064")

def filler(rng, n):
    out = []
    while len(out) < n:
        out += rng.sample(_WORDS, min(len(_WORDS), 40))
    return out[:n]

rng = random.Random(555)
words = filler(rng, int(4000 / 1.3))
idx = int(len(words) * 0.5)
words[idx] = "TIDEGLASS"
words[idx + 1] = "424242"
prompt = ("Transcript of a scanned document follows. Read it carefully.\n\n" + " ".join(words) +
          "\n\nQuestion: The report above mentions the code word TIDEGLASS exactly once. "
          "What 6-digit number follows it? Answer with only the number.")
show(tok.encode(prompt), "SHORT 4k fresh", "424242")