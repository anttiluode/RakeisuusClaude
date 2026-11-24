"""RakeisuusClaude: the semantic grain of a frozen Qwen.

Blur one word's state toward its parent in the model's own concept tree
(oak -> tree -> plant) and watch which answers go first. See README.md.

    python rakeisuus.py --self-test
    python rakeisuus.py --device cuda
    python rakeisuus.py --device cuda --max-leaves 1      (quick look)

Gate 0  is Qwen's tree in its geometry at all? (nearest parent, orthogonality vs nulls)
Gate C  can a clamped state be read at all? (band chosen on train leaves by transplant)
Gate 1  the grain ladder: word-parent path, sibling-mean path, equal-size random,
        wrong parent
Gate 2  how small is the leaf: blur to "tree", put back a compressed residue
        (top-k PCs from train leaves, equal-size random k, 1 bit per dim), and
        transplant a sibling's compressed residue
"""
import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from statistics import mean

import torch

import concepts as C

MODEL = "Qwen/Qwen3-1.7B"     # GrainLensSol's checkpoint, so the two repos can be joined
YES, NO = ["Yes", "yes"], ["No", "no"]
USER = "Item: {w}\n\n{q} Answer yes or no."
PREFIX = "Answer:"
# Gate 3's open-ended readout never names a category in the prompt.
OPEN_USER = "Item: {w}\n\n{q}"
OPEN_Q = "What kind of thing is the item above? Answer with one word."
SEED = 20261124
# Candidate bands for a 28-layer model, inclusive, scaled for other depths.
# Layer -1 is the token embedding (the input to layer 0). A band starting at -1
# leaves no copy of the original word anywhere in the forward pass, so no later
# token can route around the clamp (Neon's bypass). Bands starting at 0 or later
# are kept to measure how much leaks around them.
# Qwenimagination V9: concept swaps steer only when the band covers layers 4-12.
BANDS28 = [(-1, 12), (-1, 8), (0, 12), (2, 12), (4, 12), (4, 16)]
TS = (0.25, 0.5, 0.75, 1.0)
# Neutral slot for calibration: with no item information the model still prefers
# some option names (smoke run, Nov 24: under the wrong-category clamp X1 the leaf
# question picked the test leaf 8/8). Every choice is made on evidence minus the
# evidence for "Item: thing"; raw choices are kept beside them.
NEUTRAL = ["thing", "something", "stuff"]
# Report-only locator: clamp starting at these layers (no embedding), 8 layers long,
# to find where the word stops mattering at its own position.
LOCATOR28 = (4, 6, 8, 10, 12, 14, 16, 20)
KS = (1, 2, 4, 8, 16)


def _hs(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _replace(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


class Track:
    """One word's states: track[-1] is the embedding, track[l] the output of layer l."""

    def __init__(self, t):
        self.t = t                      # [L + 1, d]
        self.shape = t.shape

    def __getitem__(self, l):
        return self.t[l + 1]


def _pre_hook(pos, v=None, store=None):
    """Read or overwrite the embedding at `pos` (the input of layer 0)."""
    def hook(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if store is not None:
            store[-1] = h[0, pos].detach().float().cpu()
            return None
        h = h.clone()
        h[:, pos] = v.to(h.device, h.dtype)
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = h
        return args, kwargs
    return hook


# ============================================================== model access
class Lm:
    def __init__(self, model, tok, device, name):
        self.model, self.tok, self.device, self.name = model.eval(), tok, device, name
        self.layers = model.model.layers
        self.n_layers = len(self.layers)
        self.d = model.config.hidden_size
        pad = getattr(tok, "pad_token_id", None)
        self.pad = pad if pad is not None else (getattr(tok, "eos_token_id", None) or 0)
        self.yes = self._ids(YES)
        self.no = self._ids(NO)
        if not self.yes or not self.no or set(self.yes) & set(self.no):
            raise SystemExit("STOP: yes/no answer words are not single tokens")

    def _ids(self, words):
        out = set()
        for w in words:
            for s in (" " + w, w):
                ids = self.tok(s, add_special_tokens=False)["input_ids"]
                if len(ids) == 1:
                    out.add(ids[0])
        return sorted(out)

    def chat_text(self, user):
        msgs = [{"role": "user", "content": user}]
        try:
            t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return t + PREFIX

    def prompt(self, word, q, template=USER):
        """Token ids, and the word's position if the word is exactly one token there."""
        text = self.chat_text(template.format(w=word, q=q))
        ws = text.index("Item: " + word) + len("Item: ")
        we = ws + len(word)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        hit = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < we and b > ws]
        ok = len(hit) == 1 and enc["offset_mapping"][hit[0]][0] >= ws - 1 \
            and enc["offset_mapping"][hit[0]][1] <= we
        return enc["input_ids"], (hit[0] if ok else None)

    @torch.no_grad()
    def states(self, word):
        """Embedding and residual stream after every layer at the word's position."""
        ids, pos = self.prompt(word, C.TOPS[0][1])
        store = {}
        hooks = [self.layers[l].register_forward_hook(
            lambda m, i, o, l=l: store.__setitem__(l, _hs(o)[0, pos].detach().float().cpu()))
            for l in range(self.n_layers)]
        hooks.append(self.layers[0].register_forward_pre_hook(_pre_hook(pos, store=store),
                                                              with_kwargs=True))
        try:
            self.model.model(input_ids=torch.tensor([ids], device=self.device))
        finally:
            for h in hooks:
                h.remove()
        return Track(torch.stack([store[l] for l in range(-1, self.n_layers)]))

    @torch.no_grad()
    def margins(self, word, questions, targets=None):
        """Yes-minus-no log evidence for each (key, question) about "Item: word".
        targets: {layer: [d] vector} written over the word's position after that layer."""
        encs = [self.prompt(word, q) for _, q in questions]
        pos = encs[0][1]
        assert pos is not None and all(p == pos for _, p in encs)
        n = max(len(ids) for ids, _ in encs)
        x = torch.full((len(encs), n), self.pad, dtype=torch.long)
        mask = torch.zeros((len(encs), n), dtype=torch.long)
        for b, (ids, _) in enumerate(encs):
            x[b, :len(ids)] = torch.tensor(ids)
            mask[b, :len(ids)] = 1
        last = mask.sum(1) - 1
        hooks = self._clamp_hooks(pos, targets)
        try:
            hs = self.model.model(input_ids=x.to(self.device),
                                  attention_mask=mask.to(self.device)).last_hidden_state
        finally:
            for h in hooks:
                h.remove()
        h_last = hs[torch.arange(len(encs), device=hs.device), last.to(hs.device)]
        lp = torch.log_softmax(self.model.lm_head(h_last).float(), -1)
        m = torch.logsumexp(lp[:, self.yes], -1) - torch.logsumexp(lp[:, self.no], -1)
        return {k: float(v) for (k, _), v in zip(questions, m.cpu())}

    def _clamp_hooks(self, pos, targets):
        hooks = []
        for l, v in (targets or {}).items():
            if l == -1:
                hooks.append(self.layers[0].register_forward_pre_hook(_pre_hook(pos, v),
                                                                      with_kwargs=True))
                continue

            def hook(mod, inp, out, v=v):
                h = _hs(out).clone()
                h[:, pos] = v.to(h.device, h.dtype)
                return _replace(out, h)
            hooks.append(self.layers[l].register_forward_hook(hook))
        return hooks

    @torch.no_grad()
    def open_logprobs(self, word, targets=None):
        """Next-token log-probs after an open question that names no category."""
        ids, pos = self.prompt(word, OPEN_Q, template=OPEN_USER)
        hooks = self._clamp_hooks(pos, targets)
        try:
            hs = self.model.model(input_ids=torch.tensor([ids], device=self.device)).last_hidden_state
        finally:
            for h in hooks:
                h.remove()
        return torch.log_softmax(self.model.lm_head(hs[0, -1]).float(), -1).cpu()

    def word_ids(self, word):
        out = set()
        for w in {word, word.capitalize()}:
            for s_ in (" " + w, w):
                ids = self.tok(s_, add_special_tokens=False)["input_ids"]
                if len(ids) == 1:
                    out.add(ids[0])
        return sorted(out)


def load_hf(name, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return Lm(model.to(device), tok, device, name)


# ------------------------------------------------ offline stand-in for tests
class StubTokenizer:
    """Word-level tokenizer with offsets and a Qwen-style chat template."""
    SPECIAL = ["<|im_start|>", "<|im_end|>", "<pad>", "<unk>"]
    PAT = re.compile(r"<\|im_start\|>|<\|im_end\|>| ?[A-Za-z']+| ?[^\sA-Za-z]|\n")
    pad_token_id = 2
    eos_token_id = 1

    def __init__(self, texts):
        self.vocab, seen = list(self.SPECIAL), set(self.SPECIAL)
        for t in texts:
            for p in self.PAT.findall(t):
                for q in (p, " " + p.strip(), p.strip()):
                    if q and q not in seen:
                        seen.add(q)
                        self.vocab.append(q)
        self.idx = {w: i for i, w in enumerate(self.vocab)}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ms = list(self.PAT.finditer(text))
        out = {"input_ids": [self.idx.get(m.group(), 3) for m in ms]}
        if return_offsets_mapping:
            out["offset_mapping"] = [m.span() for m in ms]
        return out

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def load_stub(device="cpu", layers=8, d=64, seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    texts = C.all_words() + C.all_questions() + [USER, PREFIX, "user assistant", OPEN_Q] + \
        YES + NO + NEUTRAL + [w.capitalize() for w in C.all_words()]
    tok = StubTokenizer(texts)
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tok.vocab), hidden_size=d, intermediate_size=2 * d,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True)
    return Lm(Qwen2ForCausalLM(cfg).to(device), tok, device, f"stub-L{layers}-d{d}")


# ================================================================ the tree
class Tree:
    """The concept tree, keeping only words that are one token in the prompt."""

    def __init__(self, lm, log=print):
        self.dropped = []

        def first(cands, q):
            for w in cands:
                if lm.prompt(w, q)[1] is not None:
                    return w
                self.dropped.append(w)
            return None

        self.tops = {}
        for key, q, cands in C.TOPS:
            w = first(cands, q)
            if w is None:
                raise SystemExit(f"STOP: no single-token word for top node {key}")
            self.tops[key] = dict(key=key, q=q, word=w)
        self.cats = {}
        for key, top, q, cands, leaves in C.CATS:
            w = first(cands, q)
            kept = []
            for lw in leaves:
                if len(kept) == C.LEAVES_PER_CAT:
                    break
                if lm.prompt(lw, C.leaf_question(lw))[1] is not None:
                    kept.append(lw)
                else:
                    self.dropped.append(lw)
            if w is None or len(kept) < C.MIN_LEAVES:
                log(f"  category {key} dropped (word {w}, {len(kept)} single-token leaves)")
                continue
            self.cats[key] = dict(key=key, top=top, q=q, word=w, leaves=kept)
        # alternate leaves within each category: even -> train, odd -> test
        self.leaves = []
        for c in self.cats.values():
            for i, w in enumerate(c["leaves"]):
                self.leaves.append(dict(word=w, cat=c["key"], top=c["top"],
                                        split="train" if i % 2 == 0 else "test"))
        self.by = {L["word"]: L for L in self.leaves}
        cat_keys = list(self.cats)
        for c in self.cats.values():
            same_top = [k for k in cat_keys if self.cats[k]["top"] == c["top"]]
            c["sib"] = same_top[(same_top.index(c["key"]) + 1) % len(same_top)]
            other = [k for k in cat_keys if self.cats[k]["top"] != c["top"]]
            c["cross"] = other[cat_keys.index(c["key"]) % len(other)] if other else None

    def questions(self, leaf):
        qs = [(f"top:{k}", t["q"]) for k, t in self.tops.items()]
        qs += [(f"cat:{k}", c["q"]) for k, c in self.cats.items()]
        qs += [(f"leaf:{w}", C.leaf_question(w)) for w in self.cats[leaf["cat"]]["leaves"]]
        return qs

    def words(self):
        return [t["word"] for t in self.tops.values()] + \
               [c["word"] for c in self.cats.values()] + [L["word"] for L in self.leaves]

    def chance(self, leaf):
        return dict(top=1 / len(self.tops), cat=1 / len(self.cats),
                    leaf=1 / len(self.cats[leaf["cat"]]["leaves"]))


def judge(tree, leaf, m, top=None, cat=None, lw=None):
    """Which option wins at each level, scored against the given targets.
    Choices use evidence minus the neutral slot's evidence when tree.cal is set."""
    raw = m
    cal = getattr(tree, "cal", {}).get(leaf["cat"])
    if cal:
        m = {k: v - cal[k] for k, v in m.items()}

    def pick(prefix, m=m):
        opts = {k.split(":", 1)[1]: v for k, v in m.items() if k.startswith(prefix)}
        return max(opts, key=opts.get), opts
    t, _ = pick("top:")
    c, _ = pick("cat:")
    l, lo = pick("leaf:")
    z = torch.tensor(list(lo.values()))
    p = torch.softmax(z, 0)
    H = float(-(p * p.clamp_min(1e-12).log()).sum() / math.log(len(lo)))
    tr, _ = pick("top:", raw)
    cr, _ = pick("cat:", raw)
    lr, _ = pick("leaf:", raw)
    return dict(top=float(t == (top or leaf["top"])), cat=float(c == (cat or leaf["cat"])),
                leaf=float(l == (lw or leaf["word"])), leaf_H=H, picks=[t, c, l],
                raw=dict(top=float(tr == (top or leaf["top"])), cat=float(cr == (cat or leaf["cat"])),
                         leaf=float(lr == (lw or leaf["word"])), picks=[tr, cr, lr]))


def boot(xs, n=2000, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor(xs, dtype=torch.float64)
    idx = torch.randint(len(x), (n, len(x)), generator=g)
    ms = x[idx].mean(1).sort().values
    return dict(mean=float(x.mean()), lo=float(ms[int(0.025 * n)]), hi=float(ms[int(0.975 * n) - 1]))


def unit(seed, d):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(d, generator=g, dtype=torch.float64)
    return (u / u.norm()).float()


# ================================================================ gates
def gate0(tree, S, n_layers):
    """Is the tree in the geometry, layer by layer?"""
    rows = []
    for l in range(-1, n_layers):
        leaves = [L["word"] for L in tree.leaves]
        mu = torch.stack([S[w][l] for w in leaves]).mean(0)

        def cosv(a, b):
            return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

        near_cat, near_top, near_loo, ct, cs, cx = [], [], [], [], [], []
        for L in tree.leaves:
            h = S[L["word"]][l]
            c, t = tree.cats[L["cat"]], tree.tops[L["top"]]
            cats = {k: S[v["word"]][l] for k, v in tree.cats.items()}
            tops = {k: S[v["word"]][l] for k, v in tree.tops.items()}
            near_cat.append(float(max(cats, key=lambda k: cosv(h - mu, cats[k] - mu)) == L["cat"]))
            near_top.append(float(max(tops, key=lambda k: cosv(h - mu, tops[k] - mu)) == L["top"]))
            loo = {k: torch.stack([S[w][l] for w in v["leaves"] if w != L["word"]]).mean(0)
                   for k, v in tree.cats.items()}
            near_loo.append(float(max(loo, key=lambda k: cosv(h - mu, loo[k] - mu)) == L["cat"]))
            p1, p2 = S[c["word"]][l], S[t["word"]][l]
            ct.append(cosv(h - p1, p1 - p2))
            s1 = S[tree.cats[c["sib"]]["word"]][l]
            cs.append(cosv(h - s1, s1 - p2))
            if c["cross"]:
                xc = tree.cats[c["cross"]]
                x1, x2 = S[xc["word"]][l], S[tree.tops[xc["top"]]["word"]][l]
                cx.append(cosv(h - x1, x1 - x2))
        # cos(a-b, b-c) is about -0.5 for unrelated equal-norm vectors (Neon), so the
        # true value is only meaningful against the nulls, not against 0.
        rows.append(dict(layer=l, true_minus_sib_null=mean(ct) - mean(cs),
                         true_minus_cross_null=(mean(ct) - mean(cx)) if cx else None,
                         nearest_cat=mean(near_cat), nearest_top=mean(near_top),
                         nearest_cat_loo_mean=mean(near_loo), cos_true=mean(ct),
                         abs_cos_true=mean(abs(v) for v in ct), cos_sibling_null=mean(cs),
                         cos_cross_null=mean(cx) if cx else None))
    return rows


def scale_bands(n_layers):
    out = []
    for a, b in BANDS28:
        a2 = a if a < 0 else round(a * n_layers / 28)
        b2 = min(n_layers - 1, round(b * n_layers / 28))
        if (a2, b2) not in out and a2 <= b2:
            out.append((a2, b2))
    return out


def clamp(S, word, band):
    return {l: S[word][l] for l in range(band[0], band[1] + 1)}


def donors(tree, L):
    c = tree.cats[L["cat"]]
    i = c["leaves"].index(L["word"])
    sib = c["leaves"][(i + 1) % len(c["leaves"])]
    cross = None
    if c["cross"]:
        xl = tree.cats[c["cross"]]["leaves"]
        cross = xl[i % len(xl)]
    return sib, cross


def transplant(lm, tree, S, leaves, band):
    sl, cc, ct = [], [], []
    for L in leaves:
        qs = tree.questions(L)
        sib, cross = donors(tree, L)
        m = lm.margins(L["word"], qs, clamp(S, sib, band))
        sl.append(judge(tree, L, m, lw=sib)["leaf"])
        if cross:
            X = tree.by[cross]
            j = judge(tree, L, lm.margins(L["word"], qs, clamp(S, cross, band)),
                      top=X["top"], cat=X["cat"])
            cc.append(j["cat"])
            ct.append(j["top"])
    return dict(sibling_leaf=mean(sl), cross_cat=mean(cc) if cc else None,
                cross_top=mean(ct) if ct else None,
                score=mean([mean(sl)] + ([mean(cc), mean(ct)] if cc else [])))


def gate1_conditions(tree, S, L, band, seed):
    """{name: {layer: target}} for one leaf. All targets are built per layer."""
    c, t = tree.cats[L["cat"]], tree.tops[L["top"]]
    w = L["word"]
    sib = tree.cats[c["sib"]]
    mates = [x for x in c["leaves"] if x != w]
    topmates = [x["word"] for x in tree.leaves if x["top"] == L["top"] and x["word"] != w]
    conds = {"W0": {}}
    layers = range(band[0], band[1] + 1)
    bsib = donors(tree, L)[0]
    conds["B0.5"] = {l: 0.5 * (S[w][l] + S[bsib][l]) for l in layers}   # oak + one sibling, 50/50
    conds["B1"] = {l: S[bsib][l] for l in layers}                       # the sibling itself
    for tt in TS:
        W, Cp, M, Mt, R, RC, X = {}, {}, {}, {}, {}, {}, {}
        for l in layers:
            h, p1, p2 = S[w][l], S[c["word"]][l], S[t["word"]][l]
            m1 = torch.stack([S[x][l] for x in mates]).mean(0)
            m2 = torch.stack([S[x][l] for x in topmates]).mean(0)
            u = unit(seed * 1000 + l, h.numel())
            W[l] = h - tt * (h - p1)                          # oak -> "tree"
            Cp[l] = p1 - tt * (p1 - p2)                       # "tree" -> "plant"
            M[l] = h - tt * (h - m1)                          # oak -> mean of the other trees
            Mt[l] = m1 - tt * (m1 - m2)                       # mean of trees -> mean of plants
            R[l] = h + tt * (h - p1).norm() * u               # same size, random direction
            RC[l] = p1 + tt * (p1 - p2).norm() * u
            X[l] = h - tt * (h - S[sib["word"]][l])            # oak -> "flower" (wrong parent)
        for name, v in (("W", W), ("C", Cp), ("M", M), ("MT", Mt), ("R", R), ("RC", RC), ("X", X)):
            conds[f"{name}{tt:g}"] = v
    return conds


def gate2_conditions(tree, S, L, band, pcs, seed):
    c = tree.cats[L["cat"]]
    w = L["word"]
    sib = [x for x in c["leaves"] if x != w and tree.by[x]["split"] == "test"]
    sib = sib[0] if sib else [x for x in c["leaves"] if x != w][0]
    conds = {}
    layers = range(band[0], band[1] + 1)
    d = S[w].shape[1]
    for l in layers:
        h, p1 = S[w][l], S[c["word"]][l]
        o, U = pcs[l]                       # shared offset, PCs of centred train residues
        r, rs = h - p1 - o, S[sib][l] - p1 - o
        base = p1 + o                       # "a specific leaf of this kind, unspecified"
        conds.setdefault("K0", {})[l] = base
        for k in KS:
            if k > U.shape[0]:
                continue
            Uk = U[:k]
            kr = Uk.T @ (Uk @ r)
            g = torch.Generator().manual_seed(seed * 1000 + l * 31 + k)
            Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
            Q = Q.float().T
            qr = Q.T @ (Q @ r)
            qr = qr * (kr.norm() / qr.norm().clamp_min(1e-12))   # equal size
            conds.setdefault(f"K{k}", {})[l] = base + kr
            conds.setdefault(f"Q{k}", {})[l] = base + qr
            conds.setdefault(f"T{k}", {})[l] = base + Uk.T @ (Uk @ rs)
        conds.setdefault("P", {})[l] = p1
        conds.setdefault("S1bit", {})[l] = p1 + r.norm() * torch.sign(r) / math.sqrt(d)
        conds.setdefault("Tfull", {})[l] = S[sib][l]
    return conds, sib


def fit_pcs(tree, S, band):
    pcs = {}
    train = [L for L in tree.leaves if L["split"] == "train"]
    for l in range(band[0], band[1] + 1):
        R = torch.stack([S[L["word"]][l] - S[tree.cats[L["cat"]]["word"]][l] for L in train])
        o = R.mean(0)
        # Centred, so PC1 is not the shared "specific leaf vs the word" offset fused with
        # one train leaf's identity (the smoke run aliased maple -> oak that way).
        _, _, Vh = torch.linalg.svd((R - o).double(), full_matrices=False)
        pcs[l] = (o, Vh.float())
    return pcs


def fmt(x, spec=".2f"):
    return "-" if x is None else format(x, spec)


def first_drop(seq, base, chance):
    """Index of the first point below halfway between baseline and chance, or None."""
    th = (base + chance) / 2
    for i, v in enumerate(seq):
        if v < th:
            return i
    return None


def run(lm, out_dir, max_leaves=None, log=print):
    t0 = time.time()
    report = dict(model=lm.name, n_layers=lm.n_layers, seed=SEED, user_template=USER, prefix=PREFIX)
    tree = Tree(lm, log)
    if max_leaves:
        keep = []
        for split in ("train", "test"):
            for c in tree.cats:
                keep += [L for L in tree.leaves if L["cat"] == c and L["split"] == split][:max_leaves]
        tree.leaves = keep
        tree.by = {L["word"]: L for L in keep}
        for c in tree.cats.values():
            c["leaves"] = [w for w in c["leaves"] if w in tree.by]
    report["tree"] = dict(tops=tree.tops, cats=tree.cats, leaves=tree.leaves, dropped=tree.dropped)
    log(f"tree: {len(tree.tops)} tops, {len(tree.cats)} categories, {len(tree.leaves)} leaves "
        f"({sum(L['split'] == 'test' for L in tree.leaves)} test); dropped {len(tree.dropped)} "
        f"multi-token words")

    S = {w: lm.states(w) for w in tree.words()}
    neutral = next((w for w in NEUTRAL if lm.prompt(w, C.TOPS[0][1])[1] is not None), None)
    tree.cal = {}
    if neutral:
        for k, c in tree.cats.items():
            tree.cal[k] = lm.margins(neutral, tree.questions(tree.by[c["leaves"][0]]))
    report["neutral_slot"] = neutral
    report["neutral_choices"] = {k: dict(
        top=max((x for x in m if x.startswith("top:")), key=m.get),
        cat=max((x for x in m if x.startswith("cat:")), key=m.get),
        leaf=max((x for x in m if x.startswith("leaf:")), key=m.get)) for k, m in tree.cal.items()}
    log(f"calibration slot: 'Item: {neutral}'; what an empty slot gets called, per category: "
        f"{ {k: v['leaf'][5:] for k, v in report['neutral_choices'].items()} }")

    def save():
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f"summary.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(report, indent=1, allow_nan=False))
        tmp.replace(out_dir / "summary.json")

    # ---- Gate 0
    report["gate0"] = gate0(tree, S, lm.n_layers)
    log("\nGate 0: is the tree in the geometry? (chance: cat %.2f, top %.2f)"
        % (1 / len(tree.cats), 1 / len(tree.tops)))
    log(" layer  near-cat  near-cat(sibling mean)  near-top  cos(true)  cos(sib null)  cos(cross null)")
    for r in report["gate0"]:
        log(f"  {r['layer']:3d}    {r['nearest_cat']:.2f}          {r['nearest_cat_loo_mean']:.2f}"
            f"              {r['nearest_top']:.2f}     {r['cos_true']:+.3f}      "
            f"{r['cos_sibling_null']:+.3f}         {fmt(r['cos_cross_null'], '+.3f')}"
            f"        true-null {fmt(r['true_minus_cross_null'], '+.3f')}")
    save()

    # ---- Gate C: printed baseline, then the band by transplant on train leaves
    base = {}
    for L in tree.leaves:
        base[L["word"]] = judge(tree, L, lm.margins(L["word"], tree.questions(L)))
    test = [L for L in tree.leaves if L["split"] == "test"]
    train = [L for L in tree.leaves if L["split"] == "train"]
    chance = {k: mean(tree.chance(L)[k] for L in test) for k in ("top", "cat", "leaf")}
    b_test = {k: mean(base[L["word"]][k] for L in test) for k in ("top", "cat", "leaf")}
    report["gateC"] = dict(printed_all={k: mean(base[L["word"]][k] for L in tree.leaves)
                                        for k in ("top", "cat", "leaf")},
                           printed_test=b_test, chance_test=chance,
                           printed_ok=b_test["top"] >= 0.9 and b_test["cat"] >= 0.8 and b_test["leaf"] >= 0.8)
    log(f"\nGate C: printed word, test leaves: top {b_test['top']:.2f}  cat {b_test['cat']:.2f}  "
        f"leaf {b_test['leaf']:.2f}  (chance {chance['top']:.2f} / {chance['cat']:.2f} / {chance['leaf']:.2f})")
    bands, band_of = {}, {}
    for band in scale_bands(lm.n_layers):
        band_of[f"L{band[0]}-{band[1]}"] = band
        bands[f"L{band[0]}-{band[1]}"] = r = transplant(lm, tree, S, train, band)
        log(f"  band L{band[0]}-{band[1]} transplant on train: sibling leaf {r['sibling_leaf']:.2f}  "
            f"cross cat {fmt(r['cross_cat'])}  cross top {fmt(r['cross_top'])}")
    e, z = scale_bands(lm.n_layers)[0], scale_bands(lm.n_layers)[2]
    leak = {k: bands[f"L{z[0]}-{z[1]}"][k] - bands[f"L{e[0]}-{e[1]}"][k]
            for k in ("sibling_leaf", "score")}
    report_leak = dict(with_embedding=f"L{e[0]}-{e[1]}", without=f"L{z[0]}-{z[1]}",
                       change_when_embedding_left_out=leak)
    log(f"  leaving the embedding out of the clamp changes sibling-leaf transplant by "
        f"{leak['sibling_leaf']:+.2f} (negative = the original word leaks around the clamp)")
    key = max(bands, key=lambda k: bands[k]["score"])
    band = band_of[key]
    tt = transplant(lm, tree, S, test, band)
    report["gateC"].update(bands_train=bands, chosen_band=list(band), transplant_test=tt,
                           embedding_leak=report_leak,
                           pass_leaf=tt["sibling_leaf"] >= 0.5)
    log(f"  chosen {key}; on test leaves: sibling leaf {tt['sibling_leaf']:.2f}  "
        f"cross cat {fmt(tt['cross_cat'])}  cross top {fmt(tt['cross_top'])}"
        f"  -> leaf-level readable: {'yes' if tt['sibling_leaf'] >= 0.5 else 'NO'}")
    # report-only: where does the word stop mattering at its own position?
    window = {}
    for s0 in sorted({round(x * lm.n_layers / 28) for x in LOCATOR28}):
        b = (s0, min(lm.n_layers - 1, s0 + round(8 * lm.n_layers / 28)))
        window[f"L{b[0]}-{b[1]}"] = w = transplant(lm, tree, S, test, b)
        log(f"  window L{b[0]}-{b[1]} (no embedding), test: sibling leaf {w['sibling_leaf']:.2f}  "
            f"cross cat {fmt(w['cross_cat'])}  cross top {fmt(w['cross_top'])}")
    report["gateC"]["window"] = window
    save()

    # ---- Gate 1
    rows1 = {}
    for i, L in enumerate(test):
        for name, tg in gate1_conditions(tree, S, L, band, SEED + i).items():
            m = lm.margins(L["word"], tree.questions(L), tg or None)
            j = judge(tree, L, m)
            j["margins"] = m
            if name.startswith("X"):
                j["cat_wrong"] = float(j["picks"][1] == tree.cats[L["cat"]]["sib"])
            rows1.setdefault(name, []).append(dict(item=L["word"], **j))
    g1 = {n: {k: mean(r[k] for r in rs) for k in ("top", "cat", "leaf", "leaf_H")}
          for n, rs in rows1.items()}
    for n in g1:
        if n.startswith("X"):
            g1[n]["cat_wrong"] = mean(r["cat_wrong"] for r in rows1[n])
    path = ["W0"] + [f"W{t:g}" for t in TS] + [f"C{t:g}" for t in TS]
    drops = {k: first_drop([g1[p][k] for p in path], g1["W0"][k], chance[k]) for k in ("leaf", "cat", "top")}

    def before(a, b):
        return a is not None and (b is None or a < b)
    # a level can only "drop" if the unblurred clamp reads it well above chance
    valid = {k: g1["W0"][k] >= chance[k] + 0.25 for k in ("leaf", "cat", "top")}
    sel = []
    for i, L in enumerate(test):
        v = []
        for t in (0.5, 0.75):
            w_, r_ = rows1[f"W{t:g}"][i], rows1[f"R{t:g}"][i]
            v.append((w_["cat"] - w_["leaf"]) - (r_["cat"] - r_["leaf"]))
        sel.append(mean(v))
    selb = boot(sel)
    m1 = g1["M1"]
    x1 = g1["X1"]
    report["gate1"] = dict(
        table=g1, rows=rows1, path=path, first_drop=drops,
        readings=dict(
            levels_readable=valid,
            order_leaf_cat_top=(before(drops["leaf"], drops["cat"]) and
                                (drops["top"] is None or before(drops["cat"], drops["top"])))
            if valid["leaf"] and valid["cat"] else None,
            selectivity_vs_random=selb, selective=selb["lo"] > 0,
            sum_is_parent=m1["cat"] >= 0.8 and m1["leaf"] <= chance["leaf"] + 0.15 and m1["leaf_H"] >= 0.8,
            wrong_parent_decides=x1["cat_wrong"] >= 0.8 and x1["top"] >= 0.9))
    log("\nGate 1: accuracy on test leaves (W oak->'tree', C 'tree'->'plant', M oak->mean of other "
        "trees, MT mean trees->mean plants, R/RC same-size random, X oak->a sibling category word)")
    log("  cond    top   cat   leaf  leaf-flatness   (uncalibrated: top cat leaf)")
    for n in ["W0"] + [f"{p}{t:g}" for p in ("W", "C", "M", "MT", "R", "RC", "X") for t in TS]:
        r = g1[n]
        rw = {k: mean(x["raw"][k] for x in rows1[n]) for k in ("top", "cat", "leaf")}
        extra = f"  chose wrong cat {r['cat_wrong']:.2f}" if "cat_wrong" in r else ""
        log(f"  {n:6s} {r['top']:.2f}  {r['cat']:.2f}  {r['leaf']:.2f}   {r['leaf_H']:.2f}"
            f"            ({rw['top']:.2f} {rw['cat']:.2f} {rw['leaf']:.2f}){extra}")
    log(f"  first drop along W then C (index 0-8): {drops}")
    # Prototype drift (Neon, from Sihti's eye curve): as identity fades, does the
    # absolute evidence for the item's own category hold or rise, or does the
    # category only "survive" because its rivals fell too?
    def ev(row, L):
        m = row["margins"]
        own = m[f"leaf:{L['word']}"]
        rest = [v for k, v in m.items() if k.startswith("leaf:") and k != f"leaf:{L['word']}"]
        return m[f"cat:{L['cat']}"], m[f"top:{L['top']}"], own - mean(rest)
    drift = {}
    for p in ("W", "M", "R"):
        for t in (0.5, 0.75):
            dc, dt, ds = [], [], []
            for i, L in enumerate(test):
                c0, t0_, s0 = ev(rows1["W0"][i], L)
                c1, t1, s1 = ev(rows1[f"{p}{t:g}"][i], L)
                dc.append(c1 - c0)
                dt.append(t1 - t0_)
                ds.append(s1 - s0)
            drift[f"{p}{t:g}"] = dict(d_cat_evidence=boot(dc), d_top_evidence=boot(dt),
                                      d_leaf_specificity=boot(ds), _dc=dc)
    readings = report["gate1"]["readings"]
    for p in ("W", "M"):
        for t in (0.5, 0.75):
            k = f"{p}{t:g}"
            diff = boot([a - b for a, b in zip(drift[k]["_dc"], drift[f"R{t:g}"]["_dc"])])
            drift[k]["d_cat_vs_random"] = diff
            drift[k]["generic_rise"] = drift[k]["d_cat_evidence"]["lo"] > 0 and \
                drift[k]["d_leaf_specificity"]["hi"] < 0
            drift[k]["concept_specific"] = diff["lo"] > 0
    for v in drift.values():
        v.pop("_dc")
    report["gate1"]["prototype_drift"] = drift
    bl = []
    for i, L in enumerate(test):
        b5, b1, w0 = rows1["B0.5"][i]["margins"], rows1["B1"][i]["margins"], rows1["W0"][i]["margins"]
        sib = donors(tree, L)[0]
        c = f"cat:{L['cat']}"
        bl.append(dict(item=L["word"], sib=sib,
                       both_rejected=float(b5[f"leaf:{L['word']}"] < 0 and b5[f"leaf:{sib}"] < 0),
                       cat_overshoot=float(b5[c] > max(w0[c], b1[c])),
                       d_cat_vs_better_end=b5[c] - max(w0[c], b1[c])))
    report["gate1"]["sibling_blend"] = dict(
        rows=bl, both_rejected=mean(x["both_rejected"] for x in bl),
        cat_overshoot=mean(x["cat_overshoot"] for x in bl),
        d_cat_vs_better_end=boot([x["d_cat_vs_better_end"] for x in bl]))
    sb = report["gate1"]["sibling_blend"]
    log(f"  oak+sibling 50/50 (B0.5): both names rejected {sb['both_rejected']:.2f}, category above "
        f"both ends {sb['cat_overshoot']:.2f}, category minus better end "
        f"{sb['d_cat_vs_better_end']['mean']:+.2f} [{sb['d_cat_vs_better_end']['lo']:+.2f},"
        f"{sb['d_cat_vs_better_end']['hi']:+.2f}]")
    readings["generic_rise_toward_word_parent"] = any(drift[f"W{t:g}"]["generic_rise"] for t in (0.5, 0.75))
    readings["generic_rise_toward_sibling_mean"] = any(drift[f"M{t:g}"]["generic_rise"] for t in (0.5, 0.75))
    log("  prototype drift vs unblurred (yes-minus-no nats; mean [95% CI]):")
    for k, v in drift.items():
        f = lambda b: f"{b['mean']:+.2f} [{b['lo']:+.2f},{b['hi']:+.2f}]"
        log(f"    {k:5s} own-category {f(v['d_cat_evidence'])}  own-top {f(v['d_top_evidence'])}  "
            f"leaf specificity {f(v['d_leaf_specificity'])}")
    log(f"  readings: {readings}")
    save()

    # ---- Gate 2
    pcs = fit_pcs(tree, S, band)
    rows2 = {}
    for i, L in enumerate(test):
        conds, sib = gate2_conditions(tree, S, L, band, pcs, SEED + 7 * i)
        for name, tg in conds.items():
            target = sib if name.startswith("T") else None
            j = judge(tree, L, lm.margins(L["word"], tree.questions(L), tg), lw=target)
            j["alias_train"] = float(tree.by[j["picks"][2]]["split"] == "train")
            rows2.setdefault(name, []).append(dict(item=L["word"], sib=sib, **j))
    g2 = {n: mean(r["leaf"] for r in rs) for n, rs in rows2.items()}
    gap = b_test["leaf"] - g2["P"]

    def rec(n):
        return (g2[n] - g2["P"]) / gap if gap >= 0.25 and n in g2 else None
    ks = [k for k in KS if f"K{k}" in g2]
    small = [k for k in ks if k <= 8 and rec(f"K{k}") is not None
             and rec(f"K{k}") >= 0.8 and rec(f"Q{k}") < 0.4]
    report["gate2"] = dict(
        leaf_acc=g2, rows=rows2, gap=gap, recovery={n: rec(n) for n in g2 if not n.startswith("T")},
        readings=dict(small_residue_k=small[0] if small else None,
                      gap_big_enough=gap >= 0.25,
                      one_bit=(rec("S1bit") >= 0.8) if rec("S1bit") is not None else None,
                      transplant_k=next((k for k in ks if g2.get(f"T{k}", 0) >= 0.5), None)))
    log("\nGate 2: blur to 'tree' (P), put back part of the residue; leaf accuracy on test leaves")
    alias = {n: mean(r["alias_train"] for r in rs) for n, rs in rows2.items()}
    report["gate2"]["alias_to_train_leaf"] = alias
    log(f"  printed {b_test['leaf']:.2f}   P {g2['P']:.2f}   1 bit/dim {g2['S1bit']:.2f}   "
        f"K0 (tree + shared leaf offset) {g2['K0']:.2f}")
    for k in ks:
        log(f"  k={k:2d}: top-k PCs {g2[f'K{k}']:.2f} (named a train leaf {alias[f'K{k}']:.2f})   "
            f"equal-size random {g2[f'Q{k}']:.2f}   sibling's residue -> says sibling {g2[f'T{k}']:.2f}")
    log(f"  sibling full state -> says sibling {g2['Tfull']:.2f}")
    log(f"  readings: {report['gate2']['readings']}")

    # ---- Gate 3 (report-only): what does averaging siblings make?
    # Neon/Sol, after the smoke run: does the category sharpen as more siblings are
    # averaged, is the rise just RMSNorm gain on a shorter vector (norm-matched arm),
    # does an open question that never names the category produce it, and what
    # does an intruder from the sibling category do?
    neu = report.get("neutral_slot") or tree.cats[next(iter(tree.cats))]["leaves"][0]
    cat_ids = {k: lm.word_ids(c["word"]) for k, c in tree.cats.items()}
    open_neu = lm.open_logprobs(neu)

    def open_read(lp):
        sc = {k: float(torch.logsumexp(lp[ids], 0) - torch.logsumexp(open_neu[ids], 0))
              for k, ids in cat_ids.items() if ids}
        top5 = [lm.tok.decode([int(i)]) for i in lp.topk(5).indices]
        return sc, top5

    def matched(vecs, members):
        out = {}
        for l, v in vecs.items():
            n = mean(float(S[w][l].norm()) for w in members)
            out[l] = v * (n / float(v.norm()))
        return out
    rows3 = {}
    lay = range(band[0], band[1] + 1)
    for k, c in tree.cats.items():
        leaves = c["leaves"]
        qs = tree.questions(tree.by[leaves[0]])
        cal = tree.cal.get(k) if getattr(tree, "cal", None) else None
        intr = tree.cats[c["sib"]]["leaves"][0]
        states = {}
        for n in range(1, len(leaves) + 1):
            A = {l: torch.stack([S[w][l] for w in leaves[:n]]).mean(0) for l in lay}
            states[f"A{n}"] = (A, leaves[:n])
            states[f"N{n}"] = (matched(A, leaves[:n]), leaves[:n])
        I = {l: torch.stack([S[w][l] for w in leaves + [intr]]).mean(0) for l in lay}
        states["I"] = (I, leaves + [intr])
        states["IN"] = (matched(I, leaves + [intr]), leaves + [intr])
        for name, (tg, members) in states.items():
            m = lm.margins(neu, qs, tg)
            mc = {q: v - (cal[q] if cal else 0.0) for q, v in m.items()}
            sc, top5 = open_read(lm.open_logprobs(neu, tg))
            others = [v for q, v in mc.items() if q.startswith("cat:") and q != f"cat:{k}"]
            rows3.setdefault(k, {})[name] = dict(
                members=members,
                cat=mc[f"cat:{k}"], top=mc[f"top:{c['top']}"], cat_lead=mc[f"cat:{k}"] - max(others),
                members_rejected=mean(float(m[f"leaf:{w}"] < 0) for w in members if f"leaf:{w}" in m),
                open_choice=max(sc, key=sc.get), open_own=sc.get(k), top5=top5)
    nmax = {k: len(c["leaves"]) for k, c in tree.cats.items()}

    def across(fn):
        return boot([fn(k) for k in rows3])
    g3 = dict(
        rows=rows3,
        rise_all_vs_one=across(lambda k: rows3[k][f"A{nmax[k]}"]["cat"] - rows3[k]["A1"]["cat"]),
        rise_all_vs_one_norm_matched=across(lambda k: rows3[k][f"N{nmax[k]}"]["cat"] - rows3[k]["N1"]["cat"]),
        intruder_cat=across(lambda k: rows3[k]["I"]["cat"] - rows3[k][f"A{nmax[k]}"]["cat"]),
        intruder_top=across(lambda k: rows3[k]["I"]["top"] - rows3[k][f"A{nmax[k]}"]["top"]),
        open_right_one=mean(float(rows3[k]["A1"]["open_choice"] == k) for k in rows3),
        open_right_all=mean(float(rows3[k][f"A{nmax[k]}"]["open_choice"] == k) for k in rows3),
        open_right_all_norm_matched=mean(float(rows3[k][f"N{nmax[k]}"]["open_choice"] == k) for k in rows3))
    report["gate3"] = g3
    log("\nGate 3 (report-only): average the first n leaves of each category into an empty slot "
        f"('Item: {neu}'); calibrated evidence, N = same mean rescaled to the members' norm")
    for k, rr in rows3.items():
        line = "  ".join(f"n{n}: cat {rr[f'A{n}']['cat']:+.1f}/{rr[f'N{n}']['cat']:+.1f} "
                         f"rej {rr[f'A{n}']['members_rejected']:.2f} -> {rr[f'A{n}']['open_choice']}"
                         for n in range(1, nmax[k] + 1))
        log(f"  {k:10s} {line}")
        log(f"  {'':10s} +intruder {tree.cats[tree.cats[k]['sib']]['leaves'][0]}: cat {rr['I']['cat']:+.1f} "
            f"top {rr['I']['top']:+.1f} -> {rr['I']['open_choice']}   open top-5 at n={nmax[k]}: "
            f"{rr[f'A{nmax[k]}']['top5']}")
    f = lambda b: f"{b['mean']:+.2f} [{b['lo']:+.2f},{b['hi']:+.2f}]"
    log(f"  category rise, all vs one: {f(g3['rise_all_vs_one'])}; norm-matched: "
        f"{f(g3['rise_all_vs_one_norm_matched'])}")
    log(f"  intruder: category {f(g3['intruder_cat'])}, top {f(g3['intruder_top'])}")
    log(f"  open question names the right category: one leaf {g3['open_right_one']:.2f}, "
        f"all leaves {g3['open_right_all']:.2f}, all leaves norm-matched {g3['open_right_all_norm_matched']:.2f}")
    report["seconds"] = round(time.time() - t0, 1)
    save()
    return report


# ================================================================ self-test
def self_test():
    lm = load_stub()
    tree = Tree(lm, log=lambda *a: None)
    L = tree.leaves[0]
    qs = tree.questions(L)
    S = {w: lm.states(w) for w in tree.words()}
    band = scale_bands(lm.n_layers)[0]
    # 1. clamping a word to its own clean state changes nothing
    a, b = lm.margins(L["word"], qs), lm.margins(L["word"], qs, clamp(S, L["word"], band))
    assert max(abs(a[k] - b[k]) for k in a) < 1e-4, "identity clamp changed the answer"
    # 2. a foreign clamp changes the answer, and only from the word position on
    other = tree.leaves[-1]["word"]
    c = lm.margins(L["word"], qs, clamp(S, other, band))
    assert max(abs(a[k] - c[k]) for k in a) > 1e-4, "clamp had no effect"
    ids, pos = lm.prompt(L["word"], qs[0][1])
    seen = {}
    top = lm.n_layers - 1
    hk = lm.layers[top].register_forward_hook(lambda m, i, o: seen.__setitem__("h", _hs(o).clone()))
    lm.margins(L["word"], qs[:1])
    clean = seen["h"]
    lm.margins(L["word"], qs[:1], clamp(S, other, band))
    hk.remove()
    assert torch.allclose(clean[0, :pos], seen["h"][0, :pos]), "positions before the word moved"
    assert not torch.allclose(clean[0, pos + 1:], seen["h"][0, pos + 1:]), "later positions did not move"
    # 2b. no bypass: clamping another word from the embedding through every layer
    #     gives exactly that word's own answers, so nothing of the original survives
    full = (-1, lm.n_layers - 1)
    own = lm.margins(other, [(k, q) for k, q in qs if not k.startswith("leaf:")])
    via = lm.margins(L["word"], [(k, q) for k, q in qs if not k.startswith("leaf:")],
                     clamp(S, other, full))
    assert max(abs(own[k] - via[k]) for k in own) < 1e-4, "the original word leaked around the clamp"
    assert band[0] == -1, "first candidate band should start at the embedding"
    # 3. batched margins equal one-at-a-time margins (padding is harmless)
    one = {k: lm.margins(L["word"], [(k, q)])[k] for k, q in qs[:3]}
    assert all(abs(one[k] - a[k]) < 1e-4 for k in one), "padding changed the answer"
    # 4. the conditions: W0 is identity, W1 is the parent word, M excludes self, R is equal size
    cond = gate1_conditions(tree, S, L, band, 1)
    l = band[0]
    p1 = S[tree.cats[L["cat"]]["word"]][l]
    assert torch.allclose(cond["W1"][l], p1)
    mates = [x for x in tree.cats[L["cat"]]["leaves"] if x != L["word"]]
    assert torch.allclose(cond["M1"][l], torch.stack([S[x][l] for x in mates]).mean(0))
    h = S[L["word"]][l]
    assert abs(float((cond["R0.5"][l] - h).norm() - (cond["W0.5"][l] - h).norm())) < 1e-4
    # 5. Gate 2: PCs come from train leaves only; random arm has equal size; full residue = leaf
    test = [x for x in tree.leaves if x["split"] == "test"][0]
    pcs = fit_pcs(tree, S, band)
    g2, sib = gate2_conditions(tree, S, test, band, pcs, 1)
    p1t = S[tree.cats[test["cat"]]["word"]][l] + pcs[l][0]
    assert abs(float((g2["K4"][l] - p1t).norm() - (g2["Q4"][l] - p1t).norm())) < 1e-3
    assert torch.allclose(g2["K0"][l], p1t)
    assert torch.allclose(g2["Tfull"][l], S[sib][l])
    assert all(tree.by[x["word"]]["split"] == "train" for x in tree.leaves if x["split"] == "train")
    # 6. the whole run, end to end, on a 1-leaf-per-split tree
    out = Path("runs/selftest")
    rep = run(lm, out, max_leaves=1, log=lambda *a: None)
    json.loads((out / "summary.json").read_text())
    assert "gate2" in rep and "readings" in rep["gate1"]
    print(f"self-test passed ({len(tree.leaves)} stub leaves, band {band}, "
          f"run took {rep['seconds']} s); numbers from a random model mean nothing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--max-leaves", type=int, default=None,
                    help="leaves per category per split, for a quick look")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    lm = load_hf(a.model, a.device, getattr(torch, a.dtype))
    tag = a.model.replace("/", "_") + ("" if a.max_leaves is None else f"_smoke{a.max_leaves}")
    run(lm, a.out or Path("runs") / tag, a.max_leaves)


if __name__ == "__main__":
    main()
