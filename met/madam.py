"""MADAM-RAG baseline (Wang, Prasad, Stengel-Eskin & Bansal, 2025, arXiv:2504.13079).

Port of the authors' reference implementation (github.com/HanNight/RAMDocs, run_madam_rag.py):
  * one agent per retrieved document; round 1 sees only its own document
  * later rounds: each agent additionally sees the OTHER agents' full previous-round responses
    (the reference code does this; the paper's prose describes passing the aggregator's summary)
  * an aggregator with the authors' one-shot prompt outputs "All Correct Answers: [...]"
  * at most 3 rounds; the debate stops early when every agent's normalised answer is contained in,
    or contains, its previous-round answer, and the PREVIOUS round's aggregation is then final
  * greedy decoding, 128 new tokens per call, prompts passed as a single user turn
Prompt strings are copied verbatim, including the authors' typos, so that behaviour matches.
Changes: (1) the aggregator's reply is prefilled with "All Correct Answers: [" (see AGG_PREFILL);
(2) batching: the agents of one round are generated together, which is numerically
equivalent under greedy decoding up to padding effects.
"""
import json, re, string
import torch

AGENT_FIRST = """You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""

AGENT_LATER = """You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following reponses are from other agents as additional information.
{history}
Answer the question based on the document and other agents' response. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""

AGGREGATOR = """You are an aggregator reading answers from multiple agents.

If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

The following are examples:
Question: In which year was Michael Jordan born?
Agent responses:
Agent 1: Answer: 1963. Explanation: The document clearly states that Michael Jeffrey Jordan was born on February 17, 1963.
Agent 2: Answer: 1956. Explanation: The document states that Michael Irwin Jordan was born on February 25, 1956. However, it's important to note that this document seems to be about a different Michael Jordan, who is an American scientist, not the basketball player. The other agents' responses do not align with the information provided in the document.
Agent 3: Answer: 1998. Explanation: The According to the document provided, Michael Jeffrey Jordan was born on February 17, 1998.
Agent 4: Answer: Unknown. Explanation: The provided document focuses on Jordan's college and early professional career, mentioning his college championship in 1982 and his entry into the NBA in 1984, but it does not include information about his birth year.
All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on Februray 17, 1963, so 1963 is correct. Agent 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Agent 2 is also correct. Agent 3 provides an error stating Michael Jordan's birth year as 1998, which is incorrect. Based on the correct information from Agent 1, Michael Jeffrey Jordan was born on February 17, 1963. Agent 4 does not provide any useful information.

Question: {query}
Agent responses:
{joined}
"""


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _answer(resp):        # reference-code parsing, quirks included
    return resp[resp.find("Answer: ") + len("Answer: "):resp.find("Explanation")].strip()


@torch.no_grad()
def _generate(lm, prompts, max_new_tokens=128, prefill=""):
    tok = lm.tok
    texts = []
    for p in prompts:
        try:
            texts.append(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                 add_generation_prompt=True) + prefill)
        except Exception:
            texts.append(p + prefill)
    side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    tok.padding_side = side
    enc = {k: v.to(lm.dev) for k, v in enc.items()}
    out = lm.m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                        top_k=None, eos_token_id=sorted(lm.eos), pad_token_id=tok.pad_token_id)
    n = enc["input_ids"].shape[1]
    res = [tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]
    if any(r.count("!") > 20 for r in res):
        raise FloatingPointError("degenerate generation ('!!!'): numeric precision problem; use bfloat16")
    return res


# The reference aggregator prompt ends with the agents' responses and no closing instruction. Llama-3.1-8B
# then continues the input pattern ("Question: ... Agent responses: ...") instead of answering: in our first
# Llama run 83% of aggregations never produced "All Correct Answers". We therefore prefill the aggregator's
# reply with the answer header the prompt itself asks for. The prompt text is unchanged.
AGG_PREFILL = "All Correct Answers: ["


def _aggregate(lm, query, responses):
    joined = "\n".join(f"Agent {i+1}: {r}" for i, r in enumerate(responses))
    out = _generate(lm, [AGGREGATOR.format(query=query, joined=joined)], prefill=AGG_PREFILL)[0]
    return AGG_PREFILL + out


def parse_final(agg):
    """Return (all listed answers joined, first listed answer). 'unknown' if none."""
    if agg is None:
        return "unknown", "unknown"
    m = re.search(r"All Correct Answers:\s*\[(.*?)\]", agg, re.S)
    if not m:
        head = agg.split("Explanation")[0].strip()
        return head[:200] or "unknown", head[:200] or "unknown"
    inner = m.group(1).strip()
    try:
        items = [str(x).strip() for x in json.loads("[" + inner + "]")]
    except Exception:
        # unquoted list (the prefilled reply often omits quotes): split on commas, but never inside a number
        # such as 120,000 (comma between digits)
        items = [x.strip().strip("'\"") for x in re.split(r"\"\s*,\s*\"|',\s*'|(?<!\d),|,(?!\d)", inner)]
    items = [x for x in items if x and x.lower() != "unknown"]
    if not items:
        return "unknown", "unknown"
    return " ; ".join(items), items[0]


def madam_rag(lm, query, documents, num_rounds=3):
    rec = {"rounds": []}
    outs = _generate(lm, [AGENT_FIRST.format(query=query, document=d) for d in documents])
    agg = _aggregate(lm, query, outs)
    rec["rounds"].append({"answers": [_answer(o) for o in outs], "responses": outs, "aggregation": agg})
    final = agg
    for t in range(1, num_rounds):
        prompts = []
        for i, d in enumerate(documents):
            history = "\n".join(f"Agent {j+1}: {outs[j]}" for j in range(len(documents)) if j != i)
            # reference code: an empty history (single document) falls back to the round-1 prompt
            prompts.append(AGENT_LATER.format(query=query, document=d, history=history) if history
                           else AGENT_FIRST.format(query=query, document=d))
        new = _generate(lm, prompts)
        cur = [normalize_answer(_answer(o)) for o in new]
        prev = [normalize_answer(a) for a in rec["rounds"][-1]["answers"]]
        outs = new
        if all(c in p or p in c for c, p in zip(cur, prev)):       # converged: previous aggregation is final
            rec["rounds"].append({"answers": [_answer(o) for o in new], "responses": new, "aggregation": None})
            final = rec["rounds"][-2]["aggregation"]
            break
        agg = _aggregate(lm, query, new)
        rec["rounds"].append({"answers": [_answer(o) for o in new], "responses": new, "aggregation": agg})
        final = agg
    strict, first = parse_final(final)
    rec["final"] = final
    rec["n_rounds"] = len(rec["rounds"])
    return strict, first, rec
