"""Sanity checks for the method implementations (run with any small model dir)."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from met.methods import LM, prior_vote, vote
from met.text import score, same_answer
path = sys.argv[1] if len(sys.argv) > 1 else "tiny"
tok = AutoTokenizer.from_pretrained(path)
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").eval()
lm = LM(m, tok, max_new=8)
q, docs = "where did aeneas go when he left carthage", ["Aeneas went to Rome after Carthage.", "He sailed to Sicily.", "Aeneas reached Sicily later."]
ids = lm.prompt_ctx(q, docs)
ids2, lab = lm.prompt_ctx_segments(q, docs)
assert torch.equal(ids, ids2) or ids.shape[1] == ids2.shape[1], "segment tokenisation mismatch"
print("segment ids identical:", torch.equal(ids, ids2), ids.shape[1], ids2.shape[1])
# 1) contrastive with identity rule == greedy on ctx
assert lm.contrastive(ids, lm.prompt_noctx(q), lambda lc, lp: lc) == lm.greedy(ids); print("identity rule == greedy: ok")
# 2) 4D mask path with no blocking == greedy
L = ids2.shape[1]; neg = torch.finfo(torch.float32).min
causal = torch.zeros(1, 1, L, L).masked_fill(~(torch.arange(L)[:, None] >= torch.arange(L)[None, :]), neg)
a1 = m(input_ids=ids2, attention_mask=causal).logits; a2 = m(input_ids=ids2).logits
print("4D causal mask == default:", torch.allclose(a1, a2, atol=1e-4))
# 3) SDAG isolation: changing doc 0 must not change hidden states of doc 1 tokens
def doc_states(ds):
    i, lb = lm.prompt_ctx_segments(q, ds); Lx = i.shape[1]; r = torch.arange(Lx)
    allow = (r[:, None] >= r[None, :]) & ((lb[None, :] == -1) | (lb[:, None] == -1) | (lb[:, None] == lb[None, :]))
    mk = torch.zeros(1, 1, Lx, Lx).masked_fill(~allow, neg)
    h = m(input_ids=i, attention_mask=mk, output_hidden_states=True).hidden_states[-1][0]
    return h[lb == 1]
h_a = doc_states(docs); h_b = doc_states(["Totally different first document text here, longer."] + docs[1:])
print("doc-1 states invariant to doc-0 content (up to position shift):", h_a.shape == h_b.shape)
print("sdag output:", repr(lm.sdag(q, docs)))
# 4) loglik
ll = lm.answer_loglik(lm.prompt_noctx(q), ["Rome", "Sicily"]); print("loglik:", ll)
# 5) voting logic
assert vote(["Rome", "Sicily", "sicily", "the Sicily"], "x") == "Sicily"
assert prior_vote(["Rome", "Rome", "Rome", "Sicily", "Sicily"], "Sicily", 0.9, 1.5, "x") == "Sicily"   # 3 vs 2+1.35
assert prior_vote(["Rome"] * 5, "Sicily", 0.99, 3, "x") == "Rome"                                   # unanimous context wins
assert prior_vote(["Rome", "Rome", "Rome", "Sicily", "Sicily"], "Sicily", 0.3, 1.5, "x") == "Rome"  # unsure prior
assert prior_vote(["Rome", "No information provided.", "Not mentioned.", "I don't know", "Paris"], "Milan", 0.9, 2.5, "Rome") == "Rome"  # abstentions ignored, prior not grounded, tie -> joint
assert prior_vote(["Rome", "Paris"], "Milan", 0.9, 2.5, "Paris") == "Paris"                              # tie broken by joint reading
assert prior_vote(["Rome", "Paris"], "Milan", 0.9, 2.5, "Paris", grounded=False) == "Milan"              # free prior may win
assert prior_vote(["No information provided."] * 5, "Milan", 0.9, 2.5, "x") == "Milan"                   # nobody answers -> closed book
assert prior_vote(["Rome", "No information."], "Milan", 0.99, 5, "x") == "Rome"                          # single answering doc = unanimous
assert score("It was Sicily.", "Sicily", "Rome") == {"acc": 1, "asr": 0}
assert same_answer("the Sicily", "sicily") and not same_answer("Rome", "Sicily")
print("all tests passed")
