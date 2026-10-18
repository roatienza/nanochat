"""
Long-context SFT data tasks (Stage 7b).

Synthetic conversations that TEACH long-context skills at 128k, rendered in
nanochat chat format. Each task builds a long user turn (filler document +
question/instruction) and a short assistant answer, so the model learns to
attend across tens of thousands of tokens and to answer from distant content.

Families:
  niah        — needle-in-a-haystack: magic word + code buried in filler;
                levels vary needle depth and count (1-3 needles).
  doc_qa      — a structured "report" (sections with facts); question about a
                fact stated once, deep in the document.
  multi_tool  — tool schemas + a long document; the user asks something that
                requires ONE tool call whose arguments come from the document
                (keeps the Stage-4 CALL format alive at long context).
  long_summary— "summarize the report above in 3 bullet points" (long
                generation target, teaches not to EOS immediately).

All families are deterministic given (family, level, i, seed) so train/val
splits are reproducible and never overlap.
"""
import json
import random
import string

from tasks.common import Task

# ---------------------------------------------------------------------------
# Filler generation
# ---------------------------------------------------------------------------

_WORDS = (
    "meadow harbor copper signal orchid quartz ember willow cobalt tundra "
    "falcon marble cedar lantern ripple summit glacier nectar bramble thistle "
    "kestrel saffron basalt juniper meridian flint hollow amber lattice corsair "
    "vellum tundra zephyr cobble damson garnet hearth ingle mica nook opal "
    "pyre quill rune slate tarn umbel vitreous wattle yarrow zephyr anchor"
).split()


def _filler(rng, n_words):
    out = []
    while len(out) < n_words:
        out += rng.sample(_WORDS, min(len(_WORDS), 40))
    return out[:n_words]


def _words_to_tokens_estimate(n_words):
    return int(n_words * 1.3)  # ~1.3 tokens/word for this vocab


# ---------------------------------------------------------------------------
# Conversation helpers (nanochat format: user/assistant alternating, trailing
# assistant turn REQUIRED for render_for_completion; for SFT the assistant
# turn holds the target).
# ---------------------------------------------------------------------------

def _conv(user, assistant):
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def _magic(rng):
    word = rng.choice(["ZEPHYRQUILL", "MOONSTONE", "VERDANTKITE", "OBSIDIANFINCH",
                       "LUMENCRAG", "TIDEGLASS", "EMBERFOXGLOVE", "STARWROUGHT"])
    code = rng.randint(100000, 999999)
    return word, code


# ---------------------------------------------------------------------------
# Family generators: (prompt, target) at (level, i, rng)
# ---------------------------------------------------------------------------

def _gen_niah(rng, level, i):
    """Needle retrieval. Level 0: 1 needle @50%, 4k tokens (short-range copy
    circuit bootstrapping). Level 1: 1 needle @50%, 24k. L2: 1 needle @ random
    depth, 64k tokens. L3: 3 needles, answer asks for the SECOND one."""
    target_tokens = {0: 4000, 1: 24000, 2: 64000, 3: 100000}[level]
    n_words = int(target_tokens / 1.3)
    words = _filler(rng, n_words)
    needles = []
    n_needles = 1 if level <= 2 else 3
    for _ in range(n_needles):
        word, code = _magic(rng)
        needles.append((word, code))
    if level <= 1:
        frac = 0.5
    elif level == 2:
        frac = rng.uniform(0.15, 0.85)
    else:
        frac = rng.uniform(0.1, 0.9)
    idx = int(len(words) * frac)
    for j, (word, code) in enumerate(needles):
        pos = min(len(words) - 2, idx + j * 512)
        words[pos] = word
        words[pos + 1] = str(code)
    text = " ".join(words)
    if level <= 2:
        q = (f"The report above mentions the code word {needles[0][0]} exactly once. "
             f"What 6-digit number follows it? Answer with only the number.")
        a = str(needles[0][1])
    else:
        want = needles[1]
        q = (f"The report above mentions three code words, each followed by a "
             f"6-digit number. What number follows {want[0]}? Answer with only the number.")
        a = str(want[1])
    # DISTINCT prefix per family: the long_summary family must not share this
    # phrasing, or its near-identical prefix hijacks every long prompt
    # (observed in the first Stage 7b run: every "Below is a long report..."
    # prompt produced the summary boilerplate instead of retrieval).
    prompt = ("Transcript of a scanned document follows. Read it carefully.\n\n" + text +
              "\n\nQuestion: " + q)
    return prompt, a


_FACT_TOPICS = [
    ("water regulation", "Reservoir Ordinance 47-B caps withdrawals at 12,000 acre-feet per season."),
    ("rail scheduling", "Freight line 9 runs on Tuesdays and Fridays only, departing at 06:40."),
    ("archive access", "Reading room passes are issued at the north desk and expire after 14 days."),
    ("sensor calibration", "Barometric units are recalibrated every 240 hours of operation."),
    ("vendor payments", "Invoices under code 77 are settled net-45; all others net-30."),
    ("greenhouse zones", "Zone 4 holds humidity at 62 percent and lights run 16 hours daily."),
    ("ferry tickets", "Deck passage for vehicles taller than 2.1 meters costs double."),
    ("library fines", "Overdue maps accrue 20 cents per day up to a 15 dollar cap."),
    ("compost schedule", "Green bins are collected on the first and third Wednesday."),
    ("antenna permits", "Rooftop arrays above 3 meters need a countersigned form 12-K."),
]


def _gen_doc_qa(rng, level, i):
    """A sectioned report; the question targets one fact stated once.
    Level 2/3 make the document longer and the fact deeper."""
    target_tokens = {0: 4000, 1: 24000, 2: 64000, 3: 100000}[level]
    n_words = int(target_tokens / 1.3)
    n_sections = {0: 4, 1: 12, 2: 24, 3: 36}[level]
    topic_idx = rng.randrange(len(_FACT_TOPICS))
    section_no = rng.randrange(2, n_sections - 1)
    topic, fact = _FACT_TOPICS[topic_idx]
    words = _filler(rng, n_words)
    per_sec = len(words) // n_sections
    body = []
    for s in range(n_sections):
        body.append(f"Section {s + 1}. " + " ".join(words[s * per_sec:(s + 1) * per_sec]))
        if s == section_no:
            body.append(f"Section {s + 1} (continued). Note on {topic}: {fact}")
    text = "\n\n".join(body)
    # distinct prefix (doc_qa family): "archival record" phrasing
    prompt = ("The following is an archival record with numbered sections. Read it carefully.\n\n" + text +
              f"\n\nQuestion: What does the record say about {topic}? "
              "Answer in one sentence, quoting the exact rule.")
    return prompt, fact


_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
        "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_web", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "send_email", "parameters": {
        "type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}},
        "required": ["to", "subject"]}}},
    {"type": "function", "function": {"name": "create_ticket", "parameters": {
        "type": "object", "properties": {"title": {"type": "string"}, "priority": {"type": "string"}},
        "required": ["title", "priority"]}}},
]

_CITIES = ["Tokyo", "Oslo", "Lima", "Cairo", "Sydney", "Toronto", "Marseille", "Nairobi"]


def _gen_multi_tool(rng, level, i):
    """Long document + tool schemas; the call arguments live in the document.
    Target keeps the trained CALL format: CALL name {compact json}."""
    target_tokens = {0: 4000, 1: 24000, 2: 64000, 3: 100000}[level]
    n_words = int(target_tokens / 1.3)
    words = _filler(rng, n_words)
    city = rng.choice(_CITIES)
    unit = rng.choice(["Celsius", "Fahrenheit"])
    tool_idx = rng.randrange(2)  # weather or search for args-in-doc
    if tool_idx == 0:
        name, args = "get_weather", {"city": city, "unit": unit}
        mention = f"Operational note: the field office for this quarter is based in {city}, and all meteorological readings must be reported in {unit}."
    else:
        name, args = "search_web", {"query": f"{city} logistics bulletin"}
        mention = f"Operational note: before any shipment decision, staff must pull the latest logistics bulletin for {city}."
    pos = int(len(words) * rng.uniform(0.3, 0.8))
    words[pos] = mention
    text = " ".join(words)
    tools_json = json.dumps(_TOOLS, separators=(",", ":"))
    prompt = ("You are a function calling AI model. You are provided with function "
              "signatures within " + chr(60) + "tools" + chr(62) + tools_json + chr(60) + "/tools" + chr(62) +
              ". You may call one or more functions to assist with the user query.\n\n"
              "Below is a long operational report. Read it carefully.\n\n" + text +
              "\n\nBased on the report, make the single tool call that the operational note requires. "
              "Reply with one line of the form CALL name {json arguments}. "
              "Use the exact tool names given in the function signatures above.")
    target = "CALL " + name + " " + json.dumps(args, separators=(",", ":"))
    return prompt, target


def _gen_multi_tool_weather(rng, level, i):
    """multi_tool with the weather branch FORCED (tool_idx=0). Used by the
    --mt-extra-epochs re-anchor copies: v5j/v5n regressions pick search_web
    ("{city} logistics bulletin" is a memorized training target); oversampling
    the weather branch restores the get_weather mapping without new shapes."""
    saved = random.Random(rng.random())
    # _gen_multi_tool picks the branch with rng.randrange(2); force weather by
    # temporarily pinning randrange to 0
    orig = random.Random.randrange
    class _PinRng(random.Random):
        def randrange(self, *a):
            if a == (2,):
                return 0
            return orig(self, *a)
    pin = _PinRng(rng.random())
    return _gen_multi_tool(pin, level, i)


def _gen_long_summary(rng, level, i):
    """Teaches long GENERATION: summarize a long report; target is a real
    multi-sentence summary so the model does not EOS after a few tokens."""
    target_tokens = {0: 4000, 1: 24000, 2: 64000, 3: 100000}[level]
    n_words = int(target_tokens / 1.3)
    words = _filler(rng, n_words)
    n_sections = {0: 4, 1: 8, 2: 16, 3: 24}[level]
    per_sec = len(words) // n_sections
    themes = rng.sample(["logistics", "maintenance", "budget", "staffing",
                         "safety", "training", "inventory", "compliance"], 4)
    body = []
    for s in range(n_sections):
        theme = themes[s % 4]
        body.append(f"Section {s + 1} ({theme}). " + " ".join(words[s * per_sec:(s + 1) * per_sec]))
    text = "\n\n".join(body)
    # distinct prefix (long_summary family): imperative task framing, no
    # "report ... read carefully" phrasing shared with the other families
    prompt = ("Summarize the following document in 3 bullet points.\n\n" + text +
              "\n\nEach bullet must be one full sentence and mention its theme by name "
              "(logistics, maintenance, budget, staffing - pick the three that appear).")
    picks = themes[:3]
    a = ("\n".join(f"- The report discusses {t}: the relevant sections outline "
                   f"current practice and note where oversight applies." for t in picks))
    return prompt, a


GENERATORS = {
    "niah": _gen_niah,
    "doc_qa": _gen_doc_qa,
    "multi_tool": _gen_multi_tool,
    "multi_tool_weather": _gen_multi_tool_weather,
    "long_summary": _gen_long_summary,
}


class LongContextSFT(Task):
    """Deterministic long-context SFT conversations.

    Index layout: family_idx * levels * instances_per_level + level_slot * instances + i
    (same convention as tasks/agentic.py AgenticSandbox).
    """

    def __init__(self, families=("niah", "doc_qa", "multi_tool", "long_summary"),
                 levels=(1, 2, 3), instances_per_level=40, seed=4242, **kwargs):
        super().__init__(**kwargs)
        self.families = tuple(families)
        self.levels = tuple(sorted(levels))
        # instances_per_level: int (same count for every level) or a sequence
        # aligned with self.levels (e.g. (2000, 1000, 500, 250) for short-heavy
        # mixtures). Short rows carry the same ~8 supervised tokens as long
        # rows at a fraction of the tokens, so per-level weighting is how you
        # control supervised-signal density per unit of compute.
        if isinstance(instances_per_level, int):
            self.instances = [instances_per_level] * len(self.levels)
        else:
            assert len(instances_per_level) == len(self.levels), \
                f"instances_per_level {instances_per_level} must align with levels {self.levels}"
            self.instances = [int(n) for n in instances_per_level]
        self.instances_per_level = self.instances[0]
        self.seed = seed

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.families) * sum(self.instances)

    def _decompose(self, index):
        n_per_family = sum(self.instances)
        family_idx = index // n_per_family
        rem = index % n_per_family
        # walk level slots, each with its own instance count
        level_slot, i, acc = 0, rem, 0
        for slot, n in enumerate(self.instances):
            if i < acc + n:
                level_slot, i = slot, i - acc
                break
            acc += n
        else:  # pragma: no cover - index always < n_per_family
            level_slot, i = len(self.instances) - 1, i - acc
        family = self.families[family_idx % len(self.families)]
        level = self.levels[level_slot % len(self.levels)]
        return family, level, i

    def get_example(self, index):
        family, level, i = self._decompose(index)
        rng = random.Random(f"{self.seed}:{family}:{level}:{i}")
        prompt, target = GENERATORS[family](rng, level, i)
        return _conv(prompt, target)

    def evaluate(self, conversation, completion):
        raise NotImplementedError("SFT data task; scoring not used in training")