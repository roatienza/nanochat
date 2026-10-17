"""
ToolStop task — teaches the model when to STOP calling tools (v5f touch-up).

Why this file exists
--------------------
Live probes on the served v5e (Oct 17, probe_overcall2.py) found the exact
trigger of the agent-harness over-call: the BARE-CHAIN shape

    [assistant CALL] [tool result] [assistant CALL] [tool result] -> ???

with NO prose answer and NO user turn between the rounds. When the model
emits a second call without an intervening user question (its own choice, or
the harness feeding back results without asking anything), it is
off-distribution — no v4-shape training row has [user "Tool result: ..."]
followed by [assistant CALL] — and it then keeps chaining (10/10 at temp 0),
even when the original question was already answered by round 1 (5/5).

Data census (why no row covers it):
  * toolfollowup (hermes): every [user Tool result] is followed by PROSE. 0 rows
    of result->CALL.
  * glaive_multiround: calls only ever follow a real user question; results are
    always followed by prose. 0 rows of result->CALL.
  * toolcalling/toolace (Stage-4 shape): results ride INSIDE the assistant turn
    as python_output parts, and result->CALL IS supervised there (612 hermes +
    650 toolace rows) — but in the OLD rendering, not the v4 serving shape.
  * glaive prose-only rounds (tools present, no call): 1,650 of 33,484 rows
    (4.9%) — swamped by the call rows, hence the over-trigger on easy questions.

This task generates synthetic conversations in the EXACT v4 serving shape that
teach all four missing transitions:

  A) bare-chain-then-STOP:  user q -> CALL -> [Tool result] -> CALL ->
     [Tool result] -> PROSE (final answer covering all rounds)
  B) spurious-extra-round:  user q (single-part) -> CALL -> [Tool result] ->
     CALL (unprompted) -> [Tool result] -> PROSE answering ONLY the question
  C) no-tool-needed:        user easy q + tools present -> PROSE directly
     (fixes the P5 over-trigger)
  D) 1-round stop:          user q -> CALL -> [Tool result] -> PROSE
     (reinforces the working P1/P3 behavior)

Semantics for the assistant's final prose:
  * In (A) the question is multi-part ("weather in Tokyo and then Paris") and
    the answer covers every round's result.
  * In (B) the question asks about ONE city; the model over-called Paris
    anyway; the answer must cover ONLY the asked city (Tokyo), never the
    spurious round's data — the honest behavior an agent harness needs.
  * In (C) the answer is plain knowledge; tools are present but unused.

Rounds: 1-3 calls, same tool repeated AND different tools, 4 tool families
(weather, time, search, stock) x novel tool names, plain-text AND JSON result
bodies, a broad city list (including the harness's London/Paris/Lima/Nairobi
probes) so the model cannot key on the city list.

Rendered with the same renderer as training (strict alternation, python parts
for calls) so train==serve at the token level.
"""

import json
import random

from tasks.common import Task
from tasks.toolcalling import render_tools_prompt

# Tool schemas: the 4 Stage-4 tools plus 2 novel ones (novel names break any
# "weather = call" shortcut; the model must read the schema).
_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
        "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_web", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}},
        "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_stock_price", "parameters": {
        "type": "object", "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"]}}},
    {"type": "function", "function": {"name": "get_air_quality", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"]}}},
    {"type": "function", "function": {"name": "convert_currency", "parameters": {
        "type": "object", "properties": {"amount": {"type": "number"}, "from_": {"type": "string"}, "to": {"type": "string"}},
        "required": ["amount", "from_", "to"]}}},
]

_CITIES = ["Tokyo", "Oslo", "Lima", "Cairo", "Sydney", "Toronto", "Marseille",
           "Nairobi", "London", "Paris", "Berlin", "Madrid", "Seoul", "Dubai",
           "Singapore", "Bogota", "Helsinki", "Lisbon", "Prague", "Vienna",
           "Warsaw", "Budapest", "Athens", "Dublin", "Manila", "Jakarta",
           "Buenos Aires", "Santiago", "Casablanca", "Addis Ababa"]

_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA", "NFLX",
           "AMD", "INTC", "ORCL", "CRM", "ADBE", "PYPL", "UBER", "ABNB"]

_QUESTIONS = {
    "get_weather": [("What is the weather in {a} right now?", "{a}"),
                    ("What's the current weather in {a}?", "{a}"),
                    ("Can you check the weather in {a} for me?", "{a}")],
    "get_time": [("What time is it in {a}?", "{a}"),
                 ("Can you tell me the current time in {a}?", "{a}")],
    "search_web": [("Search the web for {a}.", "{a}"),
                   ("Look up {a} online.", "{a}")],
    "get_stock_price": [("What is the current price of {a} stock?", "{a}"),
                        ("How is {a} trading right now?", "{a}")],
    "get_air_quality": [("How is the air quality in {a} today?", "{a}")],
    "convert_currency": [("Convert 100 {a} to {b}.", "{a}|{b}")],
}

_EASY_QS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Who wrote Pride and Prejudice?",
    "What is the largest planet in the solar system?",
    "Can you explain what photosynthesis is in one sentence?",
    "What year did the Berlin Wall fall?",
    "Spell the word 'necessary'.",
    "What is 15 percent of 80?",
    "Name three primary colors.",
    "What is the boiling point of water in Fahrenheit?",
    "Who was the first person to walk on the moon?",
    "What does 'UBER' stand for as a word? Nothing—it's a brand name. Just tell me if the word is a noun.",
    "Is a tomato a fruit or a vegetable?",
    "How many continents are there?",
    "What language is spoken in Brazil?",
    "Count from 1 to 5.",
    "What is the square root of 144?",
    "Which is heavier, a kilogram of feathers or a kilogram of lead?",
    "What is the chemical symbol for gold?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
]

_THANKS = ["Thanks, that's all I needed.", "Great, thank you!",
           "Perfect, that answers my question.", "Got it, thanks!"]


def _fmt_temp(rng):
    return rng.choice(["Celsius", "Fahrenheit"])


def _call_for(rng, name, city=None):
    """(name, args, result_text, question_template_fill) for one round."""
    if name == "get_weather":
        city = city or rng.choice(_CITIES)
        unit = _fmt_temp(rng)
        temp = rng.randint(-5, 35)
        cond = rng.choice(["sunny", "cloudy", "rain", "light wind", "overcast",
                           "partly cloudy", "clear skies", "humid"])
        args = {"city": city, "unit": unit}
        result = f"{city}: {temp}{'C' if unit == 'Celsius' else 'F'}, {cond}"
        q = rng.choice(_QUESTIONS[name])[0].format(a=city)
        ans = f"It is currently {temp} degrees {'Celsius' if unit == 'Celsius' else 'Fahrenheit'} in {city} with {cond} conditions."
        return name, args, result, q, ans
    if name == "get_time":
        city = city or rng.choice(_CITIES)
        hh = rng.randint(0, 23)
        mm = rng.randint(0, 59)
        args = {"city": city}
        result = f"{city}: {hh:02d}:{mm:02d} local time"
        q = rng.choice(_QUESTIONS[name])[0].format(a=city)
        ans = f"The current local time in {city} is {hh:02d}:{mm:02d}."
        return name, args, result, q, ans
    if name == "search_web":
        topic = rng.choice(["Tokyo travel news", "EU carbon policy update",
                            "global shipping delays", "renewable energy trends",
                            "AI chip supply", "urban air mobility"])
        args = {"query": topic}
        result = f"Top result: {topic} — latest updates summarized in 3 articles."
        q = rng.choice(_QUESTIONS[name])[0].format(a=topic)
        ans = f"I found recent coverage on {topic}; the top sources agree on the key points."
        return name, args, result, q, ans
    if name == "get_stock_price":
        sym = rng.choice(_STOCKS)
        price = rng.uniform(10, 500)
        args = {"symbol": sym}
        result = json.dumps({"symbol": sym, "price": round(price, 2)})
        q = rng.choice(_QUESTIONS[name])[0].format(a=sym)
        ans = f"{sym} is trading at ${round(price, 2):.2f} right now."
        return name, args, result, q, ans
    if name == "get_air_quality":
        city = city or rng.choice(_CITIES)
        aqi = rng.randint(10, 160)
        args = {"city": city}
        result = json.dumps({"city": city, "aqi": aqi})
        q = rng.choice(_QUESTIONS[name])[0].format(a=city)
        ans = f"The air quality index in {city} is {aqi} today."
        return name, args, result, q, ans
    if name == "convert_currency":
        a, b = rng.sample(["USD", "EUR", "JPY", "GBP", "CHF", "CAD"], 2)
        rate = rng.uniform(0.5, 150)
        args = {"amount": 100, "from_": a, "to": b}
        result = json.dumps({"result": round(100 * rate, 2), "rate": round(rate, 4)})
        q = rng.choice(_QUESTIONS[name])[0].format(a=a, b=b)
        ans = f"100 {a} converts to about {round(100 * rate, 2):.2f} {b} at the current rate."
        return name, args, result, q, ans
    raise ValueError(name)


class ToolStop(Task):
    """Synthetic conversations teaching call->result->(call|prose)->stop."""

    def __init__(self, n_rows=6000, seed=777, max_tokens=6500, **kwargs):
        super().__init__(**kwargs)
        rng = random.Random(seed)
        self.examples = []
        # distribution over shapes (v5n = exact v5j dist; multi_tool re-anchored
        #  via --mt-extra-epochs instead of the E rows, which re-flipped P7):
        #  C no-tool-needed (easy q, tools present): 35%
        #  C' toolish-no-tool (boundary rows): 35%
        #  D 1-round stop: 30%
        for i in range(n_rows):
            u = rng.random()
            if u < 0.35:
                ex = self._gen_no_tool(rng, i)
            elif u < 0.70:
                ex = self._gen_notool_toolish(rng, i)
            else:
                ex = self._gen_one_round(rng, i)
            if ex is not None:
                self.examples.append(ex)
        self.length = len(self.examples)

    # ------------------------------------------------------------------
    def _mk_tools(self, rng, names):
        """The used tools plus a random number of decoys (harnesses pass the
        full tool list; training must not key on 'only the used tool is
        listed')."""
        used = [t for t in _TOOLS if t["function"]["name"] in names]
        decoys = [t for t in _TOOLS if t["function"]["name"] not in names]
        k = rng.choice([0, 0, 1, 1, 2, len(decoys)]) if decoys else 0
        extra = rng.sample(decoys, k)
        out = used + extra
        rng.shuffle(out)
        return out

    def _append_round(self, messages, name, args, result):
        messages.append({"role": "assistant", "content": [
            {"type": "python", "text": f"CALL {name} {json.dumps(args, separators=(',', ':'))}"}]})
        messages.append({"role": "user", "content": f"Tool result: {result}"})

    def _gen_bare_chain(self, rng, i, mixed):
        """(A) 2-3 rounds, NO prose between rounds, then final prose."""
        n_rounds = rng.choice([2, 2, 2, 3])
        same_tool = not mixed
        tool_name = rng.choice(["get_weather", "get_time", "get_stock_price",
                                "get_air_quality"])
        names = {tool_name} if same_tool else set(rng.sample(
            [t["function"]["name"] for t in _TOOLS], n_rounds))
        tools = self._mk_tools(rng, names)
        # multi-part question: asks about n_rounds entities up front
        rounds = []
        cities = rng.sample(_CITIES, n_rounds)
        for r in range(n_rounds):
            nm = tool_name if same_tool else sorted(names)[r]
            rounds.append(_call_for(rng, nm, city=cities[r] if nm in
                                    ("get_weather", "get_time", "get_air_quality") else None))
        # question mentions every entity (multi-part), so chaining is LEGIT
        ents = [r[3].split("?")[0].split(".")[-1].strip() for r in rounds]
        if same_tool and tool_name in ("get_weather", "get_time", "get_air_quality"):
            q = f"What is the {'weather' if tool_name == 'get_weather' else ('time' if tool_name == 'get_time' else 'air quality')} in {' and then '.join(cities)}?"
            parts = [r[4] for r in rounds]
            final = " ".join(parts)
        elif same_tool and tool_name == "get_stock_price":
            syms = [r[2] and json.loads(r[2])["symbol"] for r in rounds]
            q = f"What are the current prices of {' and '.join(syms)}?"
            final = " ".join(r[4] for r in rounds)
        else:
            q = "I need a few things: " + " Also, ".join(r[3] for r in rounds)
            final = " ".join(r[4] for r in rounds)
        messages = [{"role": "user",
                     "content": q + "\n\n" + render_tools_prompt(tools)}]
        for nm, args, result, _, _ in rounds:
            self._append_round(messages, nm, args, result)
        messages.append({"role": "assistant", "content": final})
        return {"messages": messages}

    def _gen_spurious(self, rng, i):
        """(B) single-part question; the model over-calls a 2nd round anyway;
        the final answer covers ONLY the asked entity."""
        tool_name = rng.choice(["get_weather", "get_time", "get_stock_price",
                                "get_air_quality"])
        tools = self._mk_tools(rng, {tool_name})
        nm, args, result, q, ans = _call_for(rng, tool_name)
        # spurious round: 50% same tool w/ different entity, 50% a DIFFERENT
        # tool entirely — either way it was never asked about
        if rng.random() < 0.5:
            other = rng.choice([t["function"]["name"] for t in _TOOLS
                                if t["function"]["name"] != tool_name])
            nm2, args2, result2, _, ans2 = _call_for(rng, other)
        else:
            nm2, args2, result2, _, ans2 = _call_for(rng, tool_name)
        if args2 == args:
            return None
        messages = [{"role": "user",
                     "content": q + "\n\n" + render_tools_prompt(tools)}]
        self._append_round(messages, nm, args, result)
        self._append_round(messages, nm2, args2, result2)
        messages.append({"role": "assistant", "content": ans})
        return {"messages": messages}

    def _gen_no_tool(self, rng, i):
        """(C) easy question + tools present -> prose, no call."""
        tools = self._mk_tools(rng, {t["function"]["name"] for t in _TOOLS})
        q = rng.choice(_EASY_QS)
        a = rng.choice([
            "2+2 is 4.",
            "The capital of France is Paris.",
            "Pride and Prejudice was written by Jane Austen.",
            "Jupiter is the largest planet in the solar system.",
            "Photosynthesis is how plants convert sunlight, water and CO2 into energy and oxygen.",
            "The Berlin Wall fell in 1989.",
            "N-E-C-E-S-S-A-R-Y.",
            "15 percent of 80 is 12.",
            "Red, yellow and blue are the three primary colors.",
            "Water boils at 212 degrees Fahrenheit.",
            "Neil Armstrong was the first person to walk on the moon.",
            "A tomato is botanically a fruit.",
            "There are seven continents.",
            "Portuguese is the official language of Brazil.",
            "1, 2, 3, 4, 5.",
            "The square root of 144 is 12.",
            "They weigh exactly the same: one kilogram.",
            "The chemical symbol for gold is Au.",
            "Romeo and Juliet: two feuding families' children fall in love and both die by suicide, reconciling the families.",
        ])
        messages = [{"role": "user",
                     "content": q + "\n\n" + render_tools_prompt(tools)},
                    {"role": "assistant", "content": a}]
        return {"messages": messages}

    def _gen_notool_toolish(self, rng, i):
        """(C') tool-LOOKING question answerable from knowledge -> prose.
        Attacks the exact over-trigger boundary: the question mentions
        weather/time/cities but asks for general knowledge, not live data."""
        tools = self._mk_tools(rng, {t["function"]["name"] for t in _TOOLS})
        qa = rng.choice([
            ("What time zone is Cairo in?",
             "Cairo is in the Eastern European Time zone (UTC+2)."),
            ("How do stock splits work?",
             "A stock split divides each share into multiple ones, lowering the per-share price while keeping total value unchanged."),
            ("What does GDP stand for and measure?",
             "Gross Domestic Product measures the total value of goods and services an economy produces in a period."),
            ("What is the difference between a stock and a bond?",
             "A stock is ownership in a company; a bond is a loan to it that pays fixed interest."),
            ("How do time zones offset from UTC?",
             "Time zones offset from UTC by whole or half hours, usually aligned with country or region borders."),
            ("What is inflation in simple terms?",
             "Inflation is the general rise in prices over time, which lowers what one unit of money buys."),
            ("What's the stock market's role in an economy?",
             "Stock markets let companies raise capital and let investors buy ownership shares, providing liquidity and price discovery."),
            ("How do currencies get their exchange rates?",
             "Exchange rates float based on supply, demand, interest rates and trade balances between the two currencies."),
            ("Is the air quality usually better in cities with more trees?",
             "Generally yes — urban trees filter particulates and cool the air, improving measured air quality."),
            ("What is compound interest in simple terms?",
             "Compound interest is interest earned on both the original amount and the interest already added, so growth accelerates."),
            ("Do stock prices trade on weekends?",
             "No, major stock exchanges are closed on weekends; prices only move when markets are open."),
        ])
        messages = [{"role": "user",
                     "content": qa[0] + "\n\n" + render_tools_prompt(tools)},
                    {"role": "assistant", "content": qa[1]}]
        return {"messages": messages}

    def _gen_doc_call(self, rng, i):
        """(E) doc-reading anchor (multi_tool re-anchor, v5l): a short
        operational report with an Operational note naming the field office
        (city) and reporting unit; tools present; the CORRECT behavior is to
        read the note and emit CALL get_weather {city, unit}. Re-anchors the
        multi_tool family (v5j regressed it to search_web) WITHOUT adding any
        result->CALL rows (which would flip P1) and without longctx copies
        (which re-dominate and re-flip P7)."""
        tools = self._mk_tools(rng, {"get_weather", "search_web"})
        city = rng.choice(_CITIES)
        unit = _fmt_temp(rng)
        n_words = rng.randint(120, 400)
        words = []
        vocab = ("quarterly logistics maintenance staffing budget safety "
                 "inventory compliance training operations schedule review "
                 "shipment warehouse distribution procurement audit regional "
                 "coordinator facility throughput seasonal demand forecast").split()
        while len(words) < n_words:
            words += rng.sample(vocab, min(len(vocab), 12))
        pos = int(n_words * rng.uniform(0.3, 0.8))
        words.insert(pos, "Operational note: the field office for this quarter is based in "
                          f"{city}, and all meteorological readings must be reported in {unit}.")
        text = " ".join(words)
        OB, CB = chr(91), chr(93)
        tools_json = json.dumps(tools, separators=(",", ":"))
        prompt = ("You are a function calling AI model. You are provided with function "
                  "signatures within " + OB + "tools" + CB + tools_json + OB + "/tools" + CB +
                  ". You may call one or more functions to assist with the user query.\n\n"
                  "Below is a short operational report. Read it carefully.\n\n" + text +
                  "\n\nBased on the report, make the single tool call that the operational "
                  "note requires. Reply with one line of the form CALL name {json arguments}.")
        target = "CALL get_weather " + json.dumps({"city": city, "unit": unit}, separators=(",", ":"))
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": target}]
        return {"messages": messages}

    def _gen_one_round(self, rng, i):
        """(D) 1 round then stop (reinforce the working behavior)."""
        tool_name = rng.choice([t["function"]["name"] for t in _TOOLS])
        tools = self._mk_tools(rng, {tool_name})
        nm, args, result, q, ans = _call_for(rng, tool_name)
        messages = [{"role": "user",
                     "content": q + "\n\n" + render_tools_prompt(tools)}]
        self._append_round(messages, nm, args, result)
        messages.append({"role": "assistant", "content": ans})
        return {"messages": messages}

    # ------------------------------------------------------------------
    def num_examples(self):
        return self.length

    def get_example(self, index):
        return self.examples[index]

    def evaluate(self, conversation, assistant_response):
        raise NotImplementedError("SFT data task; scoring not used in training")