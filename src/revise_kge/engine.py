"""Shared implementation for the REVISE continual KGE experiments."""
import os, time, math, random, copy, json, glob, collections, urllib.request, warnings
from types import SimpleNamespace as NS
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared implementation section 2
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("REVISE_DATA_DIR", ROOT / "data" / "raw"))
PRESET = os.environ.get("REVISE_PRESET", "paper")
RES = str(Path(os.environ.get("REVISE_RESULTS_DIR", ROOT / "results")))
os.makedirs(f"{RES}/base", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


# ======================================================================= PRESET
# 'smoke' : tiny induced subgraphs, 1 seed, minutes on CPU. Use it to check the pipeline.
# 'paper' : full benchmarks, 5 seeds, all models. Needs a GPU; every run is cached, so it can be
#           split over many sessions by setting REVISE_RESULTS_DIR.

EPOCHS_PAPER = {"FB15k-237": 150, "WN18RR": 200, "CoDEx-S": 200, "CoDEx-M": 100}
PRESETS = {
  "smoke": dict(max_entities=800, dim=32, epochs={"default": 30}, bs=512, K=16, lr=3e-3, upd_steps=120, lr_upd=2e-3,
                eval_cap=500, T=2, bs_new=128, bs_rep=128, bs_tomb=128, seeds=[0], seeds_aux=[0], tune_seed=1000,
                main=dict(datasets=["FB15k-237", "WN18RR"], models=["TransE", "RotatE", "ComplEx"]),
                scen=dict(datasets=["FB15k-237"], models=["TransE"]),
                abl=dict(datasets=["FB15k-237"], models=["TransE"]),
                sens=dict(datasets=["FB15k-237"], models=["TransE"])),
  "paper": dict(max_entities=None, dim=200, epochs=EPOCHS_PAPER, bs=1024, K=64, lr=1e-3, upd_steps=600, lr_upd=5e-4,
                eval_cap=5000, T=3, bs_new=512, bs_rep=512, bs_tomb=512, seeds=[0, 1, 2, 3, 4], seeds_aux=[0, 1, 2], tune_seed=1000,
                main=dict(datasets=["FB15k-237", "WN18RR", "CoDEx-S", "CoDEx-M"], models=["TransE", "RotatE", "ComplEx"]),
                scen=dict(datasets=["FB15k-237", "WN18RR"], models=["TransE", "RotatE", "ComplEx"]),
                abl=dict(datasets=["FB15k-237", "WN18RR"], models=["TransE", "RotatE"]),
                sens=dict(datasets=["FB15k-237", "WN18RR"], models=["TransE", "RotatE"])),
}
P = NS(**PRESETS[PRESET])
GAMMA = {("WN18RR", "TransE"): 6.0, ("WN18RR", "RotatE"): 6.0}          # default 9.0 elsewhere (RotatE paper style)
CFG = NS(dataset=None, model=None, gamma=9.0, alpha=1.0, base_epochs=None, dim=P.dim, max_entities=P.max_entities,
         new_ent_frac=0.05, hold_frac=0.05,        # entities / triples held back and ADDED over the stream
         rem_frac=0.05, sub_frac=0.03,             # fraction of the current graph deleted / substituted per step
         ent_del_n=15, ent_del_deg=(5, 40))        # entity_deletion: number of entities and degree window
SCENARIOS = ["mixed", "remove", "substitute", "entity_deletion"]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Shared implementation section 5
# ---------------------------------------------------------------------------
SOURCES = {
  "FB15k-237": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/{sp}.txt",
  "WN18RR":    "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/{sp}.txt",
  "CoDEx-S":   "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/{sp}.txt",
  "CoDEx-M":   "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/{sp}.txt",
}
def _read(ds, sp):
    path = DATA_DIR / ds / f"{sp}.txt"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True); urllib.request.urlretrieve(SOURCES[ds].format(sp=sp), path)
    return [l.strip().split("\t") for l in open(path, encoding="utf8") if l.strip()]

def load_kg(ds, max_entities=None):
    raw = {sp: _read(ds, sp) for sp in ("train", "valid", "test")}
    cnt = collections.Counter(x for h, r, t in raw["train"] for x in (h, t))
    ents = [e for e, _ in cnt.most_common(max_entities)]; keep = set(ents)
    e2i = {e: i for i, e in enumerate(ents)}
    rels = sorted({r for h, r, t in raw["train"] if h in keep and t in keep}); r2i = {r: i for i, r in enumerate(rels)}
    out = {sp: np.array([(e2i[h], r2i[r], e2i[t]) for h, r, t in tr if h in keep and t in keep and r in r2i], dtype=np.int64)
           for sp, tr in raw.items()}
    perm = np.random.default_rng(0).permutation(len(out["valid"]))
    out["valid_cal"], out["valid_ev"] = out["valid"][perm[: len(perm) // 2]], out["valid"][perm[len(perm) // 2:]]
    out.update(n_ent=len(ents), n_rel=len(rels)); return out

KG, _KG_CACHE = None, {}
def configure(ds, model):
    # select dataset / model; downstream functions read the globals KG and CFG
    global KG
    if ds not in _KG_CACHE: _KG_CACHE[ds] = load_kg(ds, P.max_entities)
    KG = _KG_CACHE[ds]; CFG.dataset, CFG.model = ds, model; CFG.gamma = GAMMA.get((ds, model), 9.0)
    CFG.base_epochs = P.epochs.get(ds, P.epochs.get("default", 100))
    return KG



# ---------------------------------------------------------------------------
# Shared implementation section 6
# ---------------------------------------------------------------------------
def keys(x, n_rel, n_ent):  return (x[:, 0] * n_rel + x[:, 1]) * n_ent + x[:, 2]
def setdiff(a, b, n_rel, n_ent):
    return a if len(b) == 0 else a[~np.isin(keys(a, n_rel, n_ent), keys(b, n_rel, n_ent))]
EMPTY = np.zeros((0, 3), dtype=np.int64)

def group_index(a, b, c, n_b):
    """dict: a*n_b+b -> array of c (used for filtered ranking)."""
    k = a * n_b + b; o = np.argsort(k, kind="stable"); k, c = k[o], c[o]
    uk, st = np.unique(k, return_index=True)
    return dict(zip(uk.tolist(), np.split(c, st[1:])))

class Filter:
    def __init__(self, triples, n_ent, n_rel):
        h, r, t = triples.T
        self.tails = group_index(h, r, t, n_rel)       # key h*n_rel+r
        self.heads = group_index(t, r, h, n_rel)       # key t*n_rel+r

def build_stream(kg, scenario, T, rng, c):
    n_ent, n_rel, train = kg["n_ent"], kg["n_rel"], kg["train"]
    known = np.concatenate([train, kg["valid"], kg["test"]])
    deg = np.bincount(train[:, [0, 2]].ravel(), minlength=n_ent)
    has_add = scenario in ("add", "mixed", "entity_deletion")
    has_rem = scenario in ("remove", "mixed", "entity_deletion")
    has_sub = scenario in ("substitute", "mixed")
    step_of = np.zeros(len(train), dtype=int)
    if has_add:
        cand = np.where(deg >= 3)[0]
        new_e = rng.choice(cand, size=int(c.new_ent_frac * n_ent), replace=False)
        is_new = np.zeros(n_ent, bool); is_new[new_e] = True
        e_step = np.zeros(n_ent, int); e_step[new_e] = rng.integers(1, T + 1, size=len(new_e))
        h, t = train[:, 0], train[:, 2]
        touch = is_new[h] | is_new[t]
        step_of[touch] = np.maximum(e_step[h], e_step[t])[touch]
        rest = np.where(~touch)[0]
        hold = rest[rng.random(len(rest)) < c.hold_frac]
        step_of[hold] = rng.integers(1, T + 1, size=len(hold))
    G = train[step_of == 0]
    chunks = [train[step_of == s] for s in range(1, T + 1)]
    steps, present = [], np.zeros(n_ent, bool)
    present[np.unique(G[:, [0, 2]])] = True
    g0 = dict(G=G, present=present.copy(), new_ents=np.where(present)[0])
    cur = G; all_removed = EMPTY
    for s in range(T):
        A = chunks[s]; D = EMPTY; S_old = EMPTY; S_new = EMPTY
        if has_rem:
            if scenario == "entity_deletion":
                dcur = np.bincount(cur[:, [0, 2]].ravel(), minlength=n_ent)
                lo, hi = c.ent_del_deg
                cand = np.where((dcur >= lo) & (dcur <= hi))[0]
                ents = rng.choice(cand, size=min(c.ent_del_n, len(cand)), replace=False)
                D = cur[np.isin(cur[:, 0], ents) | np.isin(cur[:, 2], ents)]
            else:
                D = cur[rng.choice(len(cur), size=int(c.rem_frac * len(cur)), replace=False)]
        if has_sub:
            pool = setdiff(cur, D, n_rel, n_ent)
            idx = rng.choice(len(pool), size=int(c.sub_frac * len(pool)), replace=False)
            S_old = pool[idx].copy()
            tails_of = {r: np.unique(cur[cur[:, 1] == r, 2]) for r in np.unique(S_old[:, 1])}
            new_t = np.array([rng.choice(tails_of[r]) for r in S_old[:, 1]])
            S_new = S_old.copy(); S_new[:, 2] = new_t
            ok = ~np.isin(keys(S_new, n_rel, n_ent), keys(known, n_rel, n_ent)) & (S_new[:, 2] != S_old[:, 2])
            ok &= ~np.isin(keys(S_new, n_rel, n_ent), keys(cur, n_rel, n_ent))
            S_old, S_new = S_old[ok], S_new[ok]
        nxt = setdiff(setdiff(cur, D, n_rel, n_ent), S_old, n_rel, n_ent)
        nxt = np.concatenate([nxt, A, S_new])
        prev_present = present.copy()
        present[np.unique(nxt[:, [0, 2]])] = True           # universe grows monotonically
        all_removed = np.concatenate([all_removed, D, S_old])
        steps.append(dict(G_prev=cur, G=nxt, A=A, D=D, S_old=S_old, S_new=S_new,
                          present_prev=prev_present, present=present.copy(), all_removed=all_removed.copy()))
        cur = nxt
    return g0, steps



# ---------------------------------------------------------------------------
# Shared implementation section 8
# ---------------------------------------------------------------------------
class KGE(nn.Module):
    def __init__(self, n_ent, n_rel, dim, kind, gamma, reg=0.0):
        super().__init__()
        self.kind, self.dim, self.gamma, self.reg = kind, dim, gamma, reg
        ed = dim if kind == "TransE" else 2 * dim
        rd = {"TransE": dim, "RotatE": dim, "ComplEx": 2 * dim}[kind]
        self.ent, self.rel = nn.Embedding(n_ent, ed), nn.Embedding(n_rel, rd)
        rng = (gamma + 2.0) / dim
        if kind == "ComplEx":
            nn.init.normal_(self.ent.weight, std=0.1); nn.init.normal_(self.rel.weight, std=0.1); self.reg = reg or 1e-3
        else:
            nn.init.uniform_(self.ent.weight, -rng, rng)
            if kind == "TransE": nn.init.uniform_(self.rel.weight, -rng, rng)
            else: nn.init.uniform_(self.rel.weight, -math.pi, math.pi)

    @staticmethod
    def _rot(h, ph):
        re, im = h.chunk(2, -1); c, s = torch.cos(ph), torch.sin(ph)
        return torch.cat([re * c - im * s, re * s + im * c], -1)

    def score_emb(self, H, R, T):                       # broadcasting scorer
        if self.kind == "TransE": return self.gamma - (H + R - T).abs().sum(-1)
        if self.kind == "RotatE": return self.gamma - ((self._rot(H, R) - T).pow(2).sum(-1) + 1e-9).sqrt()
        hr, hi = H.chunk(2, -1); rr, ri = R.chunk(2, -1); tr, ti = T.chunk(2, -1)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)

    def score(self, h, r, t): return self.score_emb(self.ent(h), self.rel(r), self.ent(t))

    def score_tails(self, h, r):
        H, R, E = self.ent(h), self.rel(r), self.ent.weight
        if self.kind == "TransE": return self.gamma - torch.cdist(H + R, E, p=1)
        if self.kind == "RotatE": return self.gamma - torch.cdist(self._rot(H, R), E, p=2)
        hr, hi = H.chunk(2, -1); rr, ri = R.chunk(2, -1)
        return torch.cat([hr * rr - hi * ri, hr * ri + hi * rr], -1) @ E.T

    def score_heads(self, r, t):
        T, R, E = self.ent(t), self.rel(r), self.ent.weight
        if self.kind == "TransE": return self.gamma - torch.cdist(T - R, E, p=1)
        if self.kind == "RotatE": return self.gamma - torch.cdist(self._rot(T, -R), E, p=2)
        tr, ti = T.chunk(2, -1); rr, ri = R.chunk(2, -1)
        wr, wi = rr * tr + ri * ti, ri * tr - rr * ti
        return torch.cat([wr, -wi], -1) @ E.T

def new_model(): return KGE(KG["n_ent"], KG["n_rel"], CFG.dim, CFG.model, CFG.gamma).to(device)


# ---------------------------------------------------------------------------
# Shared implementation section 10
# ---------------------------------------------------------------------------
def kge_loss(model, pos, pool, K, alpha, w=None):
    h, r, t = pos.T; B = len(h)
    sp = model.score(h, r, t)
    neg = pool[torch.randint(len(pool), (B, K), device=pos.device)]
    side = torch.rand(B, 1, device=pos.device) < 0.5
    hn = torch.where(side, neg, h[:, None]); tn = torch.where(side, t[:, None], neg)
    H, R, T = model.ent(hn), model.rel(r)[:, None], model.ent(tn)
    sn = model.score_emb(H, R, T)
    wn = torch.softmax(alpha * sn, 1).detach() if alpha > 0 else torch.full_like(sn, 1.0 / K)
    l = -(F.logsigmoid(sp) + (wn * F.logsigmoid(-sn)).sum(1)) / 2
    if model.reg > 0:
        l = l + model.reg * (model.ent(h).pow(2).sum(-1) + model.ent(t).pow(2).sum(-1) + model.rel(r).pow(2).sum(-1)) / 3
    return (l if w is None else l * w).mean()

def sync():
    if device.type == "cuda": torch.cuda.synchronize()

def train_model(model, triples, pool_ids, epochs, bs, lr, K, alpha, desc="train", verbose=True):
    X = torch.as_tensor(triples, device=device); pool = torch.as_tensor(pool_ids, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    it = tqdm(range(epochs), desc=desc, leave=False, disable=not verbose)
    for ep in it:
        model.train(); perm = torch.randperm(len(X), device=device); tot = 0.0
        for i in range(0, len(X), bs):
            loss = kge_loss(model, X[perm[i:i + bs]], pool, K, alpha)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += loss.item()
        sched.step(); it.set_postfix(loss=f"{tot / max(1, len(X) // bs):.4f}")
    model.eval(); return model


# ---------------------------------------------------------------------------
# Shared implementation section 12
# ---------------------------------------------------------------------------
@torch.no_grad()
def rank_triples(model, triples, filt, univ, sides=("t", "h"), bs=256):
    model.eval(); nr = KG["n_rel"]; out = []
    X = torch.as_tensor(triples, device=device); neg_inf = float("-inf")
    for i in range(0, len(X), bs):
        b = X[i:i + bs]; h, r, t = b.T; bk = b.cpu().numpy()
        for side in sides:
            if side == "t": sc, tgt, kk, idx = model.score_tails(h, r), t, bk[:, 0] * nr + bk[:, 1], filt.tails
            else:           sc, tgt, kk, idx = model.score_heads(r, t), h, bk[:, 2] * nr + bk[:, 1], filt.heads
            sc = sc.masked_fill(~univ, neg_inf)
            tsc = model.score(h, r, t)
            rows, cols = [], []
            for j, k in enumerate(kk.tolist()):
                a = idx.get(k)
                if a is not None: rows.append(np.full(len(a), j)); cols.append(a)
            if rows:
                sc[torch.as_tensor(np.concatenate(rows), device=device), torch.as_tensor(np.concatenate(cols), device=device)] = neg_inf
            sc.scatter_(1, tgt[:, None], tsc[:, None])
            out.append((1 + (sc > tsc[:, None]).sum(1) + 0.5 * ((sc == tsc[:, None]).sum(1) - 1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

def mstats(r):
    if len(r) == 0: return dict(mrr=np.nan, h10=np.nan, n=0)
    return dict(mrr=float((1 / r).mean()), h10=float((r <= 10).mean()), n=len(r))

def cap(x, n, rng):
    return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]

def make_eval_sets(step, kg, rng, ecap, test_triples=None):
    nr, ne = kg["n_rel"], kg["n_ent"]; univ = step["present"]
    inu = lambda x: x[univ[x[:, 0]] & univ[x[:, 2]]] if len(x) else x
    G = step["G"]; changed = np.concatenate([step["D"], step["S_old"]])
    ret = setdiff(G, np.concatenate([step["A"], step["S_new"]]), nr, ne)
    touched = np.unique(changed[:, [0, 2]]) if len(changed) else np.zeros(0, int)
    ret_nb = ret[np.isin(ret[:, 0], touched) | np.isin(ret[:, 2], touched)] if len(touched) else EMPTY
    return dict(
        test=cap(inu(kg["test"] if test_triples is None else test_triples), ecap, rng),
        added=cap(np.concatenate([step["A"], step["S_new"]]), ecap, rng),
        removed=cap(step["D"], ecap, rng),
        removed_cum=cap(step["all_removed"], ecap, rng),
        ret_nb=cap(ret_nb, ecap, rng),
        ret_rand=cap(ret, ecap, rng),
        S_old=step["S_old"], S_new=step["S_new"]), univ

@torch.no_grad()
def evaluate(model, sets, filt, univ_t, valid_scores):
    res = {}
    for name in ("test", "added", "removed", "removed_cum", "ret_nb", "ret_rand"):
        s = mstats(rank_triples(model, sets[name], filt, univ_t)) if len(sets[name]) else mstats([])
        res[f"{name}_mrr"], res[f"{name}_h10"] = s["mrr"], s["h10"]
    if len(sets["S_new"]):
        rn = rank_triples(model, sets["S_new"], filt, univ_t, sides=("t",))
        ro = rank_triples(model, sets["S_old"], filt, univ_t, sides=("t",))
        sn = model.score(*torch.as_tensor(sets["S_new"], device=device).T)
        so = model.score(*torch.as_tensor(sets["S_old"], device=device).T)
        res.update(sub_new_mrr=float((1 / rn).mean()), sub_old_mrr=float((1 / ro).mean()), sub_acc=float((sn > so).float().mean()))
    else: res.update(sub_new_mrr=np.nan, sub_old_mrr=np.nan, sub_acc=np.nan)
    if len(sets["removed"]):
        sd = model.score(*torch.as_tensor(sets["removed"], device=device).T)
        res["rem_z"] = float((sd.mean() - valid_scores.mean()) / (valid_scores.std() + 1e-9))
    else: res["rem_z"] = np.nan
    return res


# ---------------------------------------------------------------------------
# Shared implementation section 14
# ---------------------------------------------------------------------------
HP = NS(upd_steps=P.upd_steps, lr=P.lr_upd, bs_new=P.bs_new, bs_rep=P.bs_rep, bs_tomb=P.bs_tomb,
        K=P.K, alpha=CFG.alpha, init="role", init_noise=0.01,
        w_tomb=1.0, w_couple=1.0, w_dist=1.0, margin=1.0, f_min=0.2,
        tomb_q=0.5, tomb_shrink=5, lam_core=0.01, lam_buf=1.0, lam_rel=0.1, hops=1, rep_nb_cap=200000, rep_global=0.25)

REVISE_FLAGS = dict(replay="retained_nb", tomb="calib", couple=True, distill=True, anchor=True, local=True)
CORE_METHODS = {
  "Stale":            dict(stale=True),
  "FT-Add (blind)":   dict(replay="old_graph",        tomb="none",   couple=False, distill=False, anchor=False, local=False),
  "FT-Snapshot":      dict(replay="retained_uniform", tomb="none",   couple=False, distill=False, anchor=False, local=False),
  "Tombstone-neg":    dict(replay="retained_uniform", tomb="neg",    couple=False, distill=False, anchor=False, local=False),
  "NegGrad+":         dict(replay="retained_uniform", tomb="ascent", couple=False, distill=False, anchor=False, local=False),
  "REVISE":           REVISE_FLAGS,
}
ORDER = ["Stale", "FT-Add (blind)", "FT-Snapshot", "Tombstone-neg", "NegGrad+", "REVISE", "Retrain (oracle)"]
ABL_SPEC = {
  "REVISE w/o contraction":    dict(tomb="none"),
  "REVISE ascent (not calib)": dict(tomb="ascent"),
  "REVISE w/o coupling":       dict(couple=False),
  "REVISE w/o distill":        dict(distill=False),
  "REVISE w/o anchor":         dict(anchor=False),
  "REVISE w/o locality":       dict(local=False),
  "REVISE w/o nb-replay":      dict(replay="retained_uniform"),
  "REVISE random-init":        dict(init="random"),
}
SENS_SPEC = {}
for _q in (0.1, 0.25, 0.5, 0.75, 0.9): SENS_SPEC[f"REVISE[q={_q}]"] = dict(hp=dict(tomb_q=_q))
for _v in (0.0, 0.1, 1.0, 10.0):       SENS_SPEC[f"REVISE[lam_buf={_v}]"] = dict(hp=dict(lam_buf=_v))
for _v in (0, 1, 2):                    SENS_SPEC[f"REVISE[hops={_v}]"] = dict(hp=dict(hops=_v))
for _v in (0.3, 1.0, 3.0):              SENS_SPEC[f"REVISE[w_tomb={_v}]"] = dict(hp=dict(w_tomb=_v))
TUNE_GRID = [0.5, 0.75, 0.9, 0.95]
TUNED = {}          # (dataset, model) -> tomb_q chosen on the separate tuning stream
_tp = f"{RES}/tuned_{PRESET}.json"
if os.path.exists(_tp): TUNED = {tuple(k.split("|")): v for k, v in json.load(open(_tp)).items()}

def methods_for(kind):
    """Method dictionaries for the current (dataset, model); REVISE uses the tuned contraction quantile."""
    q = TUNED.get((CFG.dataset, CFG.model), HP.tomb_q)
    def rev(**o):
        f = dict(REVISE_FLAGS, **o); f["hp"] = {"tomb_q": q, **o.get("hp", {})}; return f
    if kind == "main": return {**CORE_METHODS, "REVISE": rev()}
    if kind == "abl":  return {"Stale": CORE_METHODS["Stale"], "FT-Snapshot": CORE_METHODS["FT-Snapshot"], "REVISE": rev(),
                               **{n: rev(**o) for n, o in ABL_SPEC.items()}}
    if kind == "sens": return {"Stale": CORE_METHODS["Stale"], **{n: rev(**o) for n, o in SENS_SPEC.items()}}
    if kind == "tune": return {"Stale": CORE_METHODS["Stale"], **{f"q={x}": dict(REVISE_FLAGS, hp=dict(tomb_q=x)) for x in TUNE_GRID}}

@torch.no_grad()
def role_init(model, model_old, G_prev, new_ids, G_new, noise):
    """Model-agnostic informed init: new entity <- mean role prototype of its incident (relation, role) pairs."""
    nr = KG["n_rel"]; E = model_old.ent.weight; d = E.shape[1]
    hp = torch.zeros(nr, d, device=device); tp = torch.zeros(nr, d, device=device)
    hc = torch.zeros(nr, device=device); tc = torch.zeros(nr, device=device)
    Gp = torch.as_tensor(G_prev, device=device); one = torch.ones(len(Gp), device=device)
    hp.index_add_(0, Gp[:, 1], E[Gp[:, 0]]); hc.index_add_(0, Gp[:, 1], one)
    tp.index_add_(0, Gp[:, 1], E[Gp[:, 2]]); tc.index_add_(0, Gp[:, 1], one)
    hp /= hc.clamp(min=1)[:, None]; tp /= tc.clamp(min=1)[:, None]
    is_new = torch.zeros(KG["n_ent"], dtype=torch.bool, device=device); is_new[torch.as_tensor(new_ids, device=device)] = True
    Gn = torch.as_tensor(G_new, device=device)
    acc = torch.zeros_like(model.ent.weight); cnt = torch.zeros(KG["n_ent"], device=device)
    for col, proto, c in ((0, hp, hc), (2, tp, tc)):
        m = is_new[Gn[:, col]] & (c[Gn[:, 1]] > 0)
        acc.index_add_(0, Gn[m, col], proto[Gn[m, 1]]); cnt.index_add_(0, Gn[m, col], torch.ones(int(m.sum()), device=device))
    ok = cnt > 0
    model.ent.weight[ok] = acc[ok] / cnt[ok][:, None] + noise * torch.randn_like(acc[ok]) * model.ent.weight.std()

@torch.no_grad()
def calib_targets(model, valid, univ, q, shrink):
    v = torch.as_tensor(valid[univ[valid[:, 0]] & univ[valid[:, 2]]], device=device)
    s = model.score(*v.T); glob = torch.quantile(s, q); tgt = torch.full((KG["n_rel"],), float(glob), device=device)
    for r in v[:, 1].unique():
        m = v[:, 1] == r; n = int(m.sum()); med = torch.quantile(s[m], q)
        tgt[r] = (n * med + shrink * glob) / (n + shrink)
    return tgt

def functionality(G):
    f = np.ones(KG["n_rel"])
    for r in np.unique(G[:, 1]):
        g = G[G[:, 1] == r]; f[r] = len(np.unique(g[:, 0])) / len(g)       # 1 / avg #tails per head
    return f

def neighbourhood(G, core, n_ent, hops):
    mask = np.zeros(n_ent, bool); mask[core] = True; cur = mask.copy()
    for _ in range(hops):
        m = cur[G[:, 0]] | cur[G[:, 2]]
        nb = np.zeros(n_ent, bool); nb[G[m][:, [0, 2]].ravel()] = True; cur = nb & ~mask; mask |= nb
    return mask

def update_model(flags, model_old, step, valid_cal, hp=HP, seed=0):
    t0 = time.time(); model = copy.deepcopy(model_old).to(device); model_old.eval()
    if flags.get("stale"): return model, 0.0
    hp = copy.copy(hp)
    for k, v in flags.get("hp", {}).items(): setattr(hp, k, v)
    nr, ne = KG["n_rel"], KG["n_ent"]; rng = np.random.default_rng(seed)
    G_prev, G_new = step["G_prev"], step["G"]
    A_all = np.concatenate([step["A"], step["S_new"]]); tomb = np.concatenate([step["D"], step["S_old"]])
    univ = torch.as_tensor(step["present"], device=device); pool = torch.where(univ)[0]
    new_ids = np.where(step["present"] & ~step["present_prev"])[0]
    if flags.get("init", hp.init) == "role" and len(new_ids): role_init(model, model_old, G_prev, new_ids, G_new, hp.init_noise)
    # --------------------------------------------------- replay pools and locality
    retained = setdiff(G_new, A_all, nr, ne)
    core_ids = np.unique(np.concatenate([A_all[:, [0, 2]].ravel(), tomb[:, [0, 2]].ravel()])) if len(A_all) + len(tomb) else np.zeros(0, int)
    nbmask = neighbourhood(retained, core_ids, ne, hp.hops)
    coremask = np.zeros(ne, bool); coremask[core_ids] = True
    if flags["replay"] == "old_graph": rep = G_prev                        # still contains removed facts: a removal-blind model cannot know
    elif flags["replay"] == "retained_uniform": rep = retained
    else:
        loc = retained[nbmask[retained[:, 0]] | nbmask[retained[:, 2]]]
        rep = cap(loc, hp.rep_nb_cap, rng) if len(loc) else retained
    rep_t = torch.as_tensor(rep, device=device); glob_t = torch.as_tensor(retained if len(retained) else rep, device=device)
    A_t, tomb_t = torch.as_tensor(A_all, device=device), torch.as_tensor(tomb, device=device)
    S_old_t, S_new_t = torch.as_tensor(step["S_old"], device=device), torch.as_tensor(step["S_new"], device=device)
    # --------------------------------------------------- trainable masks and anchors
    if flags["local"]:
        e_train = torch.as_tensor(nbmask, device=device)
        rel_ids = np.unique(np.concatenate([A_all[:, 1], tomb[:, 1]])) if len(A_all) + len(tomb) else np.zeros(0, int)
        r_train = torch.zeros(nr, dtype=torch.bool, device=device); r_train[torch.as_tensor(rel_ids, device=device)] = True
    else:
        e_train = univ.clone(); r_train = torch.ones(nr, dtype=torch.bool, device=device)
    lam_e = torch.zeros(ne, device=device); old_present = torch.as_tensor(step["present_prev"], device=device)
    lam_e[torch.as_tensor(nbmask, device=device) & old_present] = hp.lam_buf
    lam_e[torch.as_tensor(coremask, device=device) & old_present] = hp.lam_core
    lam_r = torch.where(r_train, torch.full((nr,), hp.lam_rel, device=device), torch.zeros(nr, device=device))
    E_ref, R_ref = model.ent.weight.detach().clone(), model.rel.weight.detach().clone()
    hooks = [model.ent.weight.register_hook(lambda g, m=e_train: g * m[:, None].to(g.dtype)),
             model.rel.weight.register_hook(lambda g, m=r_train: g * m[:, None].to(g.dtype))]
    kappa = calib_targets(model_old, valid_cal, step["present_prev"], hp.tomb_q, hp.tomb_shrink) if flags["tomb"] == "calib" and len(tomb) else None
    fw = torch.as_tensor(np.maximum(functionality(G_prev), hp.f_min), device=device, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr); model.train()
    smp = lambda X, n: X[torch.randint(len(X), (n,), device=device)]
    for it in range(hp.upd_steps):
        n_g = int(hp.bs_rep * hp.rep_global) if flags["replay"] == "retained_nb" else 0
        rep_b = torch.cat([smp(rep_t, hp.bs_rep - n_g)] + ([smp(glob_t, n_g)] if n_g else []))
        pos = torch.cat([smp(A_t, hp.bs_new), rep_b]) if len(A_t) else rep_b
        loss = kge_loss(model, pos, pool, hp.K, hp.alpha)
        if len(tomb_t) and flags["tomb"] != "none":
            tb = smp(tomb_t, hp.bs_tomb); s = model.score(*tb.T)
            if flags["tomb"] == "calib":    lt = F.relu(s - kappa[tb[:, 1]]).mean()
            elif flags["tomb"] == "ascent": lt = F.logsigmoid(s).mean()          # minimising this = gradient ascent on the fact's loss
            else:                           lt = -F.logsigmoid(-s).mean()        # 'neg': removed fact used as an ordinary negative
            loss = loss + hp.w_tomb * lt
        if flags["couple"] and len(S_new_t):
            j = torch.randint(len(S_new_t), (hp.bs_tomb,), device=device); so, sn = S_old_t[j], S_new_t[j]
            gap = model.score(*sn.T) - model.score(*so.T)
            loss = loss + hp.w_couple * (fw[sn[:, 1]] * F.relu(hp.margin - gap)).mean()
        if flags["distill"]:
            with torch.no_grad(): s_old = model_old.score(*rep_b.T)
            loss = loss + hp.w_dist * (model.score(*rep_b.T) - s_old).pow(2).mean()
        if flags["anchor"]:
            loss = loss + (lam_e[:, None] * (model.ent.weight - E_ref).pow(2)).sum() / max(1, int((lam_e > 0).sum())) \
                        + (lam_r[:, None] * (model.rel.weight - R_ref).pow(2)).sum() / max(1, int((lam_r > 0).sum()))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    for h in hooks: h.remove()
    sync(); model.eval(); return model, time.time() - t0


# ---------------------------------------------------------------------------
# Shared implementation section 16
# ---------------------------------------------------------------------------
def run_stream(seed, scenario, methods, tag, tune=False, oracle=True, verbose=False):
    ds, mdl = CFG.dataset, CFG.model
    set_seed(seed); rng = np.random.default_rng(seed)
    g0, steps = build_stream(KG, scenario, P.T, rng, CFG)
    flag = "add" if scenario in ("add", "mixed", "entity_deletion") else "noadd"
    bpath = f"{RES}/base/{PRESET}__{ds}__{mdl}__s{seed}__{flag}__e{CFG.base_epochs}.pt"
    base = new_model()
    if os.path.exists(bpath): base.load_state_dict(torch.load(bpath, map_location=device)); base.eval()
    else:
        set_seed(seed); train_model(base, g0["G"], np.where(g0["present"])[0], CFG.base_epochs, P.bs, P.lr, P.K, CFG.alpha, f"base {ds}/{mdl}", verbose)
        torch.save(base.state_dict(), bpath)
    test_all = KG["valid_ev"] if tune else KG["test"]
    ev0 = np.random.default_rng(999 + seed)
    t0_ = test_all[g0["present"][test_all[:, 0]] & g0["present"][test_all[:, 2]]]
    gf = Filter(np.concatenate([g0["G"], KG["valid"], KG["test"]]), KG["n_ent"], KG["n_rel"])
    base_mrr = mstats(rank_triples(base, cap(t0_, P.eval_cap, ev0), gf, torch.as_tensor(g0["present"], device=device)))["mrr"]
    models = {m: base for m in methods}; rows = []
    for si, st in enumerate(steps, 1):
        ev_rng = np.random.default_rng(1000 + seed * 10 + si)
        sets, univ_np = make_eval_sets(st, KG, ev_rng, P.eval_cap, test_triples=KG["valid_ev"] if tune else None)
        univ = torch.as_tensor(univ_np, device=device)
        filt = Filter(np.concatenate([st["G"], KG["valid"], KG["test"]]), KG["n_ent"], KG["n_rel"])
        vsub = KG["valid_ev"][univ_np[KG["valid_ev"][:, 0]] & univ_np[KG["valid_ev"][:, 2]]]
        valid_ref = torch.as_tensor(cap(vsub, 2000, ev_rng), device=device)
        out = {}
        for m, fl in methods.items():
            models[m], dt = update_model(fl, models[m], st, KG["valid_cal"], HP, seed); out[m] = (models[m], dt)
        if oracle:
            set_seed(seed * 100 + si); t1 = time.time()
            orc = train_model(new_model(), st["G"], np.where(univ_np)[0], CFG.base_epochs, P.bs, P.lr, P.K, CFG.alpha, f"oracle@{si}", verbose)
            sync(); out["Retrain (oracle)"] = (orc, time.time() - t1)
        for m, (mod, dt) in out.items():
            with torch.no_grad(): vs = mod.score(*valid_ref.T)
            rows.append(dict(tag=tag, dataset=ds, model=mdl, scenario=scenario, seed=seed, step=si, method=m, time=dt, base_test_mrr=base_mrr,
                             n_add=len(st["A"]) + len(st["S_new"]), n_del=len(st["D"]), n_sub=len(st["S_old"]),
                             **evaluate(mod, sets, filt, univ, vs)))
    return pd.DataFrame(rows)

def add_derived(df):
    """Oracle Deviation (OD), Forgetting Efficacy (FE), speed-up w.r.t. retraining; per (dataset, model, scenario, seed, step)."""
    ref = ["test_mrr", "added_mrr", "removed_mrr", "ret_nb_mrr", "ret_rand_mrr"]; out = []
    for _, g in df.groupby(["dataset", "model", "scenario", "seed", "step"]):
        g = g.set_index("method").copy(); g["OD"] = np.nan; g["FE"] = np.nan; g["speedup"] = np.nan
        if "Retrain (oracle)" in g.index:
            orc = g.loc["Retrain (oracle)"]
            g["OD"] = g[ref].astype(float).sub(orc[ref].astype(float)).abs().mean(axis=1, skipna=True)
            g["speedup"] = orc["time"] / g["time"].replace(0, np.nan)
            if "Stale" in g.index and not np.isnan(g.loc["Stale", "removed_mrr"]) and abs(g.loc["Stale", "removed_mrr"] - orc["removed_mrr"]) > 1e-6:
                g["FE"] = (g.loc["Stale", "removed_mrr"] - g["removed_mrr"]) / (g.loc["Stale", "removed_mrr"] - orc["removed_mrr"])
        out.append(g.reset_index())
    res = pd.concat(out, ignore_index=True)
    for c in ("OD", "FE", "speedup"): res[c] = pd.to_numeric(res[c], errors="coerce")
    return res

def run_experiment(tag, plan, scenarios, seeds, kind):
    """Loops over (dataset, model, scenario, seed); skips runs whose CSV already exists."""
    todo = [(d, m, sc, s) for d in plan["datasets"] for m in plan["models"] for sc in scenarios for s in seeds]
    print(f"[{tag}] {len(todo)} runs, {len(todo) * P.T} oracle trainings (+ base models, cached)")
    for d, m, sc, s in tqdm(todo, desc=tag):
        path = f"{RES}/{tag}__{d}__{m}__{sc}__s{s}.csv"
        if os.path.exists(path): continue
        configure(d, m)
        run_stream(s, sc, methods_for(kind), tag).to_csv(path, index=False)

def load_results(tag):
    fs = sorted(glob.glob(f"{RES}/{tag}__*.csv"))
    return add_derived(pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)) if fs else None


# ---------------------------------------------------------------------------
# Shared implementation section 18
# ---------------------------------------------------------------------------
def tune_all(plan):
    for d in plan["datasets"]:
        for m in plan["models"]:
            if (d, m) in TUNED: continue
            path = f"{RES}/tune__{d}__{m}__mixed__s{P.tune_seed}.csv"
            if not os.path.exists(path):
                configure(d, m); run_stream(P.tune_seed, "mixed", methods_for("tune"), "tune", tune=True).to_csv(path, index=False)
            df = add_derived(pd.read_csv(path)); last = df[(df.step == df.step.max()) & df.method.str.startswith("q=")]
            TUNED[(d, m)] = float(last.loc[last.OD.idxmin(), "method"].split("=")[1])
    json.dump({"|".join(k): v for k, v in TUNED.items()}, open(_tp, "w"), indent=1)
    print("selected contraction quantile q:", TUNED)

