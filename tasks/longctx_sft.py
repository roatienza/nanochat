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


# ---------------------------------------------------------------------------
# Generated fact material
# ---------------------------------------------------------------------------
# The original _FACT_TOPICS pool held 10 (topic, fact) pairs and the question
# named the topic, so the target was a 10-entry lookup the model could memorize
# without reading the document. Measured on the old code: 300/300 doc_qa rows
# were answerable from the question text alone, and 200 sampled rows produced
# 10 distinct targets. Facts are now GENERATED per instance, so every target is
# a function of document content and the mapping cannot be memorized.

_TOPIC_STEMS = (
    "water regulation", "rail scheduling", "archive access", "sensor calibration",
    "vendor payments", "greenhouse zones", "ferry tickets", "library fines",
    "compost schedule", "antenna permits", "fleet servicing", "cold storage",
    "irrigation quotas", "noise abatement", "harbour dredging", "turbine output",
    "seed stock", "customs bonding", "radio licensing", "waste transfer",
)

_QUALIFIERS = ("north", "south", "east", "west", "central", "coastal", "upland",
               "harbour", "inland", "border")

_UNITS = ("acre-feet", "kilowatt-hours", "litres", "tonnes", "hours", "meters",
          "days", "credits", "permits", "crates")


def _clause(rng):
    """A clause code, e.g. 'QP-47C'. Fresh per instance."""
    return (rng.choice(string.ascii_uppercase) + rng.choice(string.ascii_uppercase) +
            f"-{rng.randint(10, 99)}" + rng.choice(string.ascii_uppercase))


def _gen_topic(rng):
    return f"{rng.choice(_QUALIFIERS)} {rng.choice(_TOPIC_STEMS)}"


def _gen_fact(rng, topic):
    """A fact that is a function of generated numbers, never a fixed string."""
    n = rng.randint(11, 9999)
    m = rng.randint(2, 24)
    unit = rng.choice(_UNITS)
    code = _clause(rng)
    return rng.choice([
        f"{topic.capitalize()} is capped at {n:,} {unit} per season under clause {code}.",
        f"Under clause {code}, {topic} must be reviewed every {m} {unit} and logged in register {n}.",
        f"{topic.capitalize()} requires a countersigned form {code} whenever it exceeds {n:,} {unit}.",
        f"Clause {code} sets the {topic} threshold at {n:,} {unit}, reviewed every {m} {unit}.",
        f"Where {topic} applies, record {n:,} {unit} against clause {code} before release.",
    ])


def _sections(rng, n_words, n_sections):
    """Split filler into numbered sections, returning (body_list, per_sec_words).

    body_list holds the section paragraphs; callers insert their own marked
    sentences after the paragraph they want to target.
    """
    words = _filler(rng, n_words)
    per_sec = max(1, len(words) // n_sections)
    body = []
    for s in range(n_sections):
        body.append(f"Section {s + 1}. " + " ".join(words[s * per_sec:(s + 1) * per_sec]))
    return body, words


def _gen_doc_qa(rng, level, i):
    """A sectioned report; the question targets one GENERATED fact stated once.
    Level 2/3 make the document longer and the fact deeper."""
    target_tokens = {0: 4000, 1: 24000, 2: 64000, 3: 100000}
    n_words = int(target_tokens[level] / 1.3)
    n_sections = {0: 4, 1: 12, 2: 24, 3: 36}[level]
    topic = _gen_topic(rng)
    fact = _gen_fact(rng, topic)
    section_no = rng.randrange(2, n_sections - 1)
    body, _ = _sections(rng, n_words, n_sections)
    body.insert(section_no + 1, f"Section {section_no + 1} (continued). Note on {topic}: {fact}")
    text = "\n\n".join(body)
    # distinct prefix (doc_qa family): "archival record" phrasing
    prompt = ("The following is an archival record with numbered sections. Read it carefully.\n\n" + text +
              f"\n\nQuestion: What does the record say about {topic}? "
              "Answer in one sentence, quoting the exact rule.")
    return prompt, fact


def _gen_doc_arith(rng, level, i):
    """The answer is a SUM of line items buried in one section. No single span
    of the document contains the target, so retrieval alone cannot answer it."""
    target_tokens = {0: 4000, 1: 24000, 2: 64000, 3: 100000}
    n_words = int(target_tokens[level] / 1.3)
    n_sections = {0: 4, 1: 12, 2: 24, 3: 36}[level]
    unit = rng.choice(_UNITS)
    n_items = {0: 3, 1: 4, 2: 5, 3: 6}[level]
    items = [rng.randint(11, 999) for _ in range(n_items)]
    total = sum(items)
    section_no = rng.randrange(2, n_sections - 1)
    body, _ = _sections(rng, n_words, n_sections)
    listing = "; ".join(f"{v:,} {unit}" for v in items)
    body.insert(section_no + 1,
                f"Section {section_no + 1} (continued). Line items: {listing}.")
    text = "\n\n".join(body)
    prompt = ("The following is a procurement record with numbered sections. Read it carefully.\n\n" + text +
              f"\n\nQuestion: What is the total of the line items listed in section "
              f"{section_no + 1}, in {unit}? Answer with only the number, no separators.")
    return prompt, str(total)


def _gen_doc_join(rng, level, i):
    """Two facts in DIFFERENT sections must be combined to answer."""
    target_tokens = {0: 4000, 1: 24000, 2: 64000, 3: 100000}
    n_words = int(target_tokens[level] / 1.3)
    n_sections = {0: 4, 1: 12, 2: 24, 3: 36}[level]
    entity = rng.choice(["copper", "cobalt", "cedar", "granite", "saffron", "graphite",
                         "juniper", "basalt", "amber", "opal"])
    depart = rng.randint(1, 25)
    transit = rng.randint(2, 21)
    arrive = depart + transit
    a_no = rng.randrange(1, n_sections // 2)
    b_no = rng.randrange(n_sections // 2, n_sections - 1)
    body, _ = _sections(rng, n_words, n_sections)
    # insert at the HIGHER index first so the lower insertion does not shift it
    body.insert(b_no + 1, f"Section {b_no + 1} (continued). Transit for {entity} cargo "
                          f"takes {transit} days from the loading date.")
    body.insert(a_no + 1, f"Section {a_no + 1} (continued). The {entity} cargo loads on "
                          f"day {depart} of the month.")
    text = "\n\n".join(body)
    prompt = ("The following is a shipping dossier with numbered sections. Read it carefully.\n\n" + text +
              f"\n\nQuestion: On which day of the month does the {entity} cargo arrive? "
              "Combine the loading day with the transit time. Answer with only the number.")
    return prompt, str(arrive)


def _gen_doc_contradict(rng, level, i):
    """The same attribute appears twice with different values; the instruction
    says the higher revision governs. The model must prefer the stated rule
    over the first value it meets."""
    target_tokens = {0: 4000, 1: 24000, 2: 64000, 3: 100000}
    n_words = int(target_tokens[level] / 1.3)
    n_sections = {0: 4, 1: 12, 2: 24, 3: 36}[level]
    topic = _gen_topic(rng)
    unit = rng.choice(_UNITS)
    old_val = rng.randint(11, 999)
    new_val = rng.randint(11, 999)
    while new_val == old_val:
        new_val = rng.randint(11, 999)
    old_rev, new_rev = rng.randint(2, 5), rng.randint(6, 9)
    a_no = rng.randrange(1, n_sections // 2)
    b_no = rng.randrange(n_sections // 2, n_sections - 1)
    body, _ = _sections(rng, n_words, n_sections)
    # Which revision sits in which section is RANDOM. Pinning the higher
    # revision to the later section would let the model answer with a
    # positional shortcut ("last value wins") and never read the revision
    # number, which is the whole skill this family is meant to teach.
    hi_no, lo_no = (a_no, b_no) if rng.random() < 0.5 else (b_no, a_no)
    for sec_no, rev, val in ((hi_no, new_rev, new_val), (lo_no, old_rev, old_val)):
        body.insert(sec_no + 1, f"Section {sec_no + 1} (revision {rev}). {topic.capitalize()} "
                                f"is set to {val:,} {unit}.")
    text = "\n\n".join(body)
    prompt = ("The following is a controlled record with numbered sections. Read it carefully.\n\n" + text +
              f"\n\nQuestion: Two sections give a value for {topic}. Where sections disagree, "
              "the higher revision number governs. What is the current value? "
              f"Answer with only the number and unit, e.g. 42 {unit}.")
    return prompt, f"{new_val:,} {unit}"


def _gen_doc_track(rng, level, i):
    """A quantity mutates across several sections; ask for the FINAL state.
    Every intermediate value appears in the text, so returning the first or
    the largest value is wrong — the model must apply all deltas in order."""
    target_tokens = {0: 4000, 1: 24000, 2: 64000, 3: 100000}
    n_words = int(target_tokens[level] / 1.3)
    n_sections = {0: 4, 1: 12, 2: 24, 3: 36}[level]
    item = rng.choice(["stock", "reserve", "balance", "inventory", "allocation"])
    start = rng.randint(200, 900)
    n_events = {0: 2, 1: 3, 2: 4, 3: 5}[level]
    slots = sorted(rng.sample(range(1, n_sections - 1), n_events))
    state = start
    events = []
    for s in slots:
        delta = rng.randint(11, 120)
        sign = rng.choice([+1, -1])
        if sign < 0 and state - delta < 0:
            sign = +1
        state += sign * delta
        events.append((s, sign, delta))
    body, _ = _sections(rng, n_words, n_sections)
    for s, sign, delta in sorted(events, reverse=True):
        verb = "increased by" if sign > 0 else "decreased by"
        body.insert(s + 1, f"Section {s + 1} (continued). The {item} {verb} {delta} units.")
    body.insert(1, f"Section 1 (continued). The {item} opens at {start} units.")
    text = "\n\n".join(body)
    prompt = ("The following is a running ledger with numbered sections, in order. "
              "Read it carefully.\n\n" + text +
              f"\n\nQuestion: Apply every change in section order. What is the final {item}, "
              "in units? Answer with only the number.")
    return prompt, str(state)

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
    multi-sentence summary so the model does not EOS after a few tokens.

    Two defects fixed here, both measured on the previous version:
      (1) the prompt hardcoded a 4-name theme list while the target named
          themes from an 8-name pool -> 285/300 rows named a theme the prompt
          never offered, i.e. the family taught the model to ignore an
          explicit constraint in the instruction.
      (2) every bullet ended with the same sentence ("outline current practice
          and note where oversight applies"), so the target was reproducible
          from format alone without reading the document.
    Themes are now drawn from the document that was actually built, and each
    bullet must carry that section's recorded figures.
    """
    target_tokens = {0: 4000, 1: 24000, 2: 64000, 3: 100000}
    n_words = int(target_tokens[level] / 1.3)
    words = _filler(rng, n_words)
    n_sections = {0: 4, 1: 8, 2: 16, 3: 24}[level]
    per_sec = max(1, len(words) // n_sections)
    # themes actually present in THIS document, in document order
    pool = ("logistics", "maintenance", "budget", "staffing", "safety",
            "training", "inventory", "compliance", "procurement", "outreach",
            "insurance", "utilities")
    k = min(len(pool), max(3, n_sections // 2))
    themes = rng.sample(pool, k)
    sec_theme, sec_figs = [], []
    for s in range(n_sections):
        t = themes[s % k]
        entries = rng.randint(11, 999)
        flagged = rng.randint(1, max(1, entries // 4))
        sec_theme.append(t)
        sec_figs.append((entries, flagged))
    body = []
    for s in range(n_sections):
        entries, flagged = sec_figs[s]
        body.append(f"Section {s + 1} ({sec_theme[s]}). " + " ".join(words[s * per_sec:(s + 1) * per_sec]) +
                    f" The register records {entries} entries, of which {flagged} were flagged.")
    text = "\n\n".join(body)
    # distinct prefix (long_summary family): imperative task framing, no
    # "report ... read carefully" phrasing shared with the other families
    first_seen = []
    for t in sec_theme:
        if t not in [x for x, _ in first_seen]:
            first_seen.append((t, sec_figs[sec_theme.index(t)]))
    picks = first_seen[:3]
    offered = ", ".join(t for t, _ in picks)
    prompt = ("Summarize the following document in 3 bullet points.\n\n" + text +
              f"\n\nEach bullet must name one of these themes and state that theme's figures "
              f"({offered}). Do not invent a theme that is not listed.")
    a = "\n".join(f"- {t}: {entries} entries recorded, {flagged} flagged."
                  for t, (entries, flagged) in picks)
    return prompt, a


GENERATORS = {
    "niah": _gen_niah,
    "doc_qa": _gen_doc_qa,
    "doc_arith": _gen_doc_arith,
    "doc_join": _gen_doc_join,
    "doc_contradict": _gen_doc_contradict,
    "doc_track": _gen_doc_track,
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