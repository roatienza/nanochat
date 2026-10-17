from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import GSM8K
from tasks.toolcalling import ToolCalling
import statistics

t = get_tokenizer()
g = GSM8K(subset="main", split="train")
n = g.num_examples()
lens = [len(t.render_conversation(g.get_example(i))[0]) for i in range(0, n, 25)]
gsm_avg = statistics.mean(lens)
print(f"GSM8K rows={n} avg_tokens/row={gsm_avg:.1f}")

tc = ToolCalling(split="train")
m = tc.num_examples()
tlens = [len(t.render_conversation(tc.get_example(i))[0]) for i in range(0, m, 10)]
tc_avg = statistics.mean(tlens)
print(f"ToolCalling rows={m} avg_tokens/row={tc_avg:.1f}")

gsm_rows = 10 * n
tc_rows = 2 * m
total_tokens = gsm_rows * gsm_avg + tc_rows * tc_avg
steps = total_tokens / 32768
print(f"MIXTURE: GSM8K x10 ({gsm_rows:,}) + ToolCalling x2 ({tc_rows:,}) = {gsm_rows+tc_rows:,} rows")
print(f"~{total_tokens/1e6:.1f}M tokens -> ~{steps:.0f} steps (32768 tok/step)")
