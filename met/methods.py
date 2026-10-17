"""Decoding methods for mixed-evidence RAG. All CPU-friendly, training-free.

Token-level (single blended context):  rag, cad, adacad, arr
Attention-level:                        sdag  (block-diagonal attention between documents)
Answer-level over isolated documents:   vote  (isolate-then-aggregate, no prior)
                                        pvote (ours: prior as a bounded, confidence-weighted extra voter)
"""
import math
import re
import torch
import torch.nn.functional as F

from .text import norm, same_answer, is_nonanswer

SYS_CTX = ("You answer questions using the provided documents. "
           "Reply with only the short answer (a few words), no explanation.")
SYS_NOCTX = "Answer the question. Reply with only the short answer (a few words), no explanation."


class LM:
    def __init__(self, model, tok, max_new=12):
        self.m, self.tok, self.max_new = model, tok, max_new
        self.eos = set(x for x in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                                   tok.convert_tokens_to_ids("<|eot_id|>")] if isinstance(x, int) and x >= 0)
        self.dtype = next(model.parameters()).dtype
        self.dev = model.get_input_embeddings().weight.device  # works on CPU, single GPU, or device_map="auto"

    # ---------- prompts ----------
    def _chat(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{system}\n\n{user}\nAnswer:"

    def ids(self, text):
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids

    def prompt_noctx(self, q):
        return self.ids(self._chat(SYS_NOCTX, f"Question: {q}"))

    def prompt_ctx(self, q, docs):
        body = "".join(f"Document [{i+1}]: {d}\n\n" for i, d in enumerate(docs))
        return self.ids(self._chat(SYS_CTX, f"{body}Question: {q}"))

    def prompt_ctx_segments(self, q, docs):
        """Same text as prompt_ctx but returns ids plus per-token block labels (-1 = shared, i = doc i)."""
        marker = "\u0001DOCS\u0001"
        full = self._chat(SYS_CTX, f"{marker}Question: {q}")
        pre, post = full.split(marker)
        pieces = [(pre, -1)] + [(f"Document [{i+1}]: {d}\n\n", i) for i, d in enumerate(docs)] + [(post, -1)]
        ids, lab = [], []
        for text, b in pieces:
            t = self.tok(text, add_special_tokens=False).input_ids
            ids += t
            lab += [b] * len(t)
        return torch.tensor([ids]), torch.tensor(lab)

    # ---------- low-level ----------
    @torch.no_grad()
    def _step(self, ids, past=None, mask4d=None):
        kw = dict(input_ids=ids.to(self.dev), past_key_values=past, use_cache=True)
        if mask4d is not None:
            kw["attention_mask"] = mask4d.to(self.dev)
        out = self.m(**kw)
        logits = out.logits[0, -1].float()
        if not torch.isfinite(logits).all():  # fp16 overflow etc. -> fail fast instead of emitting junk
            raise FloatingPointError("Model produced NaN/inf logits. Use --dtype bfloat16 or float32.")
        return logits, out.past_key_values

    def _decode(self, tok_ids):
        return self.tok.decode(tok_ids, skip_special_tokens=True).strip().split("\n")[0].strip()

    @torch.no_grad()
    def greedy(self, ids):
        logits, past = self._step(ids)
        out = []
        for _ in range(self.max_new):
            t = int(torch.argmax(logits))
            if t in self.eos:
                break
            out.append(t)
            logits, past = self._step(torch.tensor([[t]]), past)
        return self._decode(out)

    @torch.no_grad()
    def greedy_conf(self, ids):
        """Greedy answer plus the model's confidence in it: exp(mean log-prob of the generated tokens)."""
        logits, past = self._step(ids)
        out, lps = [], []
        for _ in range(self.max_new):
            lp = F.log_softmax(logits, -1)
            t = int(torch.argmax(lp))
            if t in self.eos:
                break
            out.append(t); lps.append(float(lp[t]))
            logits, past = self._step(torch.tensor([[t]]), past)
        return self._decode(out), (math.exp(sum(lps) / len(lps)) if lps else 0.0)

    @torch.no_grad()
    def contrastive(self, ids_ctx, ids_pri, rule):
        """Step-wise combination of context and prior next-token distributions."""
        lc, pc_past = self._step(ids_ctx)
        lp, pp_past = self._step(ids_pri)
        out = []
        for _ in range(self.max_new):
            t = int(torch.argmax(rule(F.log_softmax(lc, -1), F.log_softmax(lp, -1))))
            if t in self.eos:
                break
            out.append(t)
            x = torch.tensor([[t]])
            lc, pc_past = self._step(x, pc_past)
            lp, pp_past = self._step(x, pp_past)
        return self._decode(out)

    @torch.no_grad()
    def sdag(self, q, docs):
        ids, lab = self.prompt_ctx_segments(q, docs)
        L = ids.shape[1]
        r = torch.arange(L)
        causal = r[:, None] >= r[None, :]
        rb, cb = lab[:, None], lab[None, :]
        allowed = causal & ((cb == -1) | (rb == -1) | (rb == cb))  # doc tokens can't see other docs
        neg = torch.finfo(self.dtype).min
        m4 = torch.zeros(1, 1, L, L, dtype=self.dtype).masked_fill(~allowed, neg)
        logits, past = self._step(ids, mask4d=m4)
        out, cur = [], L
        for _ in range(self.max_new):
            t = int(torch.argmax(logits))
            if t in self.eos:
                break
            out.append(t)
            cur += 1
            logits, past = self._step(torch.tensor([[t]]), past, torch.zeros(1, 1, 1, cur, dtype=self.dtype))
        return self._decode(out)

    @torch.no_grad()
    def answer_loglik(self, ids_prompt, answers):
        """Length-normalised log p(answer | prompt) for each candidate answer (one batched pass)."""
        if not answers:
            return []
        seqs, spans = [], []
        for a in answers:
            a_ids = self.tok(a, add_special_tokens=False).input_ids or [self.tok.eos_token_id]
            seqs.append(ids_prompt[0].tolist() + a_ids)
            spans.append(len(a_ids))
        P = ids_prompt.shape[1]
        W = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        x = torch.full((len(seqs), W), pad)
        att = torch.zeros(len(seqs), W, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        lp = F.log_softmax(self.m(input_ids=x.to(self.dev), attention_mask=att.to(self.dev)).logits.float(), -1).cpu()
        res = []
        for i, n in enumerate(spans):
            tgt = x[i, P:P + n]
            res.append(float(lp[i, P - 1:P - 1 + n].gather(-1, tgt[:, None]).mean()))
        return res


# ---------- token-level combination rules (log-prob inputs) ----------
def jsd(lc, lp):
    pc, pp = lc.exp(), lp.exp()
    m = 0.5 * (pc + pp)
    lm = torch.log(m.clamp_min(1e-12))
    return float(0.5 * (pc * (lc - lm)).sum() + 0.5 * (pp * (lp - lm)).sum()) / math.log(2)  # in [0,1]


def rule_cad(alpha=1.0):
    return lambda lc, lp: (1 + alpha) * lc - alpha * lp


def rule_adacad():
    def f(lc, lp):
        a = jsd(lc, lp)
        return (1 + a) * lc - a * lp
    return f


def rule_arr():
    """Adaptive Regime Routing (Jiang et al., 2026): tau = 1 + (2d-1)*JSD, d = [max p_ctx > max p_prior]."""
    def f(lc, lp):
        d = 1.0 if float(lc.max()) > float(lp.max()) else 0.0
        tau = 1 + (2 * d - 1) * jsd(lc, lp)
        return (1 - tau) * lp + tau * lc
    return f


# ---------- answer-level aggregation ----------
def cluster_votes(answers):
    """Group isolated-document answers into clusters of equivalent answers."""
    clusters = []  # [representative, count]
    for a in answers:
        if not norm(a) or is_nonanswer(a):  # refusals / "not mentioned" are abstentions, not votes
            continue
        for c in clusters:
            if same_answer(a, c[0]):
                c[1] += 1
                break
        else:
            clusters.append([a, 1])
    return clusters


def vote(iso_answers, fallback):
    cl = cluster_votes(iso_answers)
    if not cl:
        return fallback
    return max(cl, key=lambda c: c[1])[0]


def prior_vote(iso_answers, prior_answer, prior_conf, gamma, joint_answer, grounded=True):
    """Ours (Mixed-Evidence Trust vote).
    1. Each document is read in isolation; refusals ("not mentioned", "I don't know", ...) are abstentions.
    2. No document answers  -> the model's closed-book answer.
    3. All answering documents agree -> follow them (a unanimous context, e.g. a legitimate update, is never overruled).
    4. Otherwise each document casts one vote and the closed-book prior casts gamma * prior_conf votes for its answer.
       grounded=True: the prior can only reinforce an answer that at least one document gives (it arbitrates, it never
       introduces an unsupported answer). grounded=False: it may also win on its own if gamma * prior_conf exceeds the
       largest document cluster (more robust, less faithful; the high-robustness operating point).
    5. Ties are broken by the joint reading of all documents (plain RAG answer)."""
    cl = cluster_votes(iso_answers)
    if not cl:
        return prior_answer if norm(prior_answer) else joint_answer
    if len(cl) == 1:
        return cl[0][0]
    scores = [c[1] for c in cl]
    hit = [i for i, c in enumerate(cl) if same_answer(prior_answer, c[0])]
    if hit:
        scores[hit[0]] += gamma * prior_conf
    elif not grounded and gamma * prior_conf > max(scores):
        return prior_answer
    best = max(scores)
    tied = [i for i, s in enumerate(scores) if abs(s - best) < 1e-9]
    by_joint = [i for i in tied if same_answer(joint_answer, cl[i][0])]
    return cl[(by_joint or tied)[0]][0]


# ---------------------------------------------------------------------------
# RobustRAG baselines (Xiang et al., 2024, "Certifiably Robust RAG against Retrieval Corruption").
# Re-implemented from the paper and the official code (inspire-group/RobustRAG), QA setting:
#   * isolated per-passage prompts that may abstain with "I don't know"
#   * keyword aggregation: keep phrases occurring in >= min(beta, alpha * #non-abstained) responses,
#     then answer the question from the aggregated keyword hints (alpha=0.3, beta=3)
#   * decoding aggregation: average next-token distributions of the non-abstaining passages at every
#     step (eta=0, the QA setting); passages with P("I don't know") >= 0.99 are dropped
# ---------------------------------------------------------------------------
SYS_ISO_ABST = ("You answer a question using only the given document. Reply with only the short answer "
                "(a few words). If the document does not contain the answer, reply exactly: I don't know.")
SYS_HINTS = ("Answer the question. Suggested phrases were aggregated from several sources and are listed "
             "most specific first. Reply with only the short answer (a few words), no explanation.")
IDK = "I don't know"
_STOP = set("a an the of in on at to for from by with and or but is are was were be been being this that these "
            "those it its as into about over under than then so such not no yes do does did has have had".split())
_IGNORE_POS = {'VERB', 'INTJ', 'ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PRON', 'SCONJ', 'PUNCT', 'SPACE'}
_nlp = None


def _spacy():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = False  # fall back to a stopword-based extractor
    return _nlp


def extract_phrases(response):
    """Phrase set as in RobustRAG: the whole response, maximal runs of content tokens (lemmatised), and their words."""
    out = {response.strip()}
    nlp = _spacy()
    if nlp:
        run = []
        for t in list(nlp(response)) + [None]:
            if t is None or t.pos_ in _IGNORE_POS:
                if run:
                    out.add("".join(x.lemma_ + x.whitespace_ for x in run).strip())
                    out.update(x.lemma_ for x in run)
                    run = []
            else:
                run.append(t)
    else:
        run = []
        for w in re.findall(r"[\w'-]+|[^\w\s]", response) + [None]:
            if w is None or w.lower() in _STOP or not re.match(r"[\w]", w):
                if run:
                    out.add(" ".join(run)); out.update(run); run = []
            else:
                run.append(w)
    return {p for p in out if p}


def keyword_hints(responses, alpha=0.3, beta=3):
    kept = [r for r in responses if "I don't" not in r and "I do not" not in r and r.strip()]
    if not kept:
        return None
    cnt = {}
    for r in kept:
        for ph in extract_phrases(r):
            cnt[ph] = cnt.get(ph, 0) + 1
    thr = min(beta, alpha * len(kept))
    keep = [(p, c) for p, c in cnt.items() if c >= thr and p.lower() not in _STOP and re.search(r"\w", p)]
    keep.sort(key=lambda x: (len(x[0]), x[0]), reverse=True)
    return ", ".join(p for p, _ in keep)


def _lm_iso_abst(lm, q, doc):
    return lm.ids(lm._chat(SYS_ISO_ABST, f"Document: {doc}\n\nQuestion: {q}"))


def _lm_hints(lm, q, hints):
    return lm.ids(lm._chat(SYS_HINTS, f"Question: {q}\nSuggested phrases: [{hints}]"))


@torch.no_grad()
def seq_logprob(lm, ids_prompt, text):
    """Total log p(text | prompt)."""
    a_ids = lm.tok(text, add_special_tokens=False).input_ids
    x = torch.tensor([ids_prompt[0].tolist() + a_ids])
    lp = F.log_softmax(lm.m(input_ids=x.to(lm.dev)).logits.float(), -1).cpu()
    P = ids_prompt.shape[1]
    return float(lp[0, P - 1:P - 1 + len(a_ids)].gather(-1, torch.tensor(a_ids)[:, None]).sum())


def robustrag_keyword(lm, q, docs, iso_abst=None):
    iso_abst = iso_abst if iso_abst is not None else [lm.greedy(_lm_iso_abst(lm, q, d)) for d in docs]
    hints = keyword_hints(iso_abst)
    if hints is None:
        return IDK + ".", iso_abst
    return lm.greedy(_lm_hints(lm, q, hints)), iso_abst


@torch.no_grad()
def robustrag_decoding(lm, q, docs, abstain_prob=0.99):
    prompts = [_lm_iso_abst(lm, q, d) for d in docs]
    keep = [p for p in prompts if math.exp(seq_logprob(lm, p, IDK)) < abstain_prob]
    if not keep:
        return IDK + "."
    banned = set()
    for w in ["I", "Please", "\n"]:
        t = lm.tok(w, add_special_tokens=False).input_ids
        if t:
            banned.add(t[0])
    states = [lm._step(p) for p in keep]
    out = []
    for step in range(lm.max_new):
        probs = torch.stack([F.softmax(lg, -1) for lg, _ in states]).mean(0)
        if step == 0:
            for b in banned:
                probs[b] = 0
        t = int(torch.argmax(probs))
        if t in lm.eos:
            break
        out.append(t)
        x = torch.tensor([[t]])
        states = [lm._step(x, past) for _, past in states]
    return lm._decode(out)
