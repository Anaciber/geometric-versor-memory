"""
Level 3: knowledge editing, miniature of the standard protocol.

BASE MODEL: a small transformer LM trained from scratch to memorize a
knowledge base of facts (subject, relation) -> object, each fact expressed
through 4 paraphrase templates; templates {0,1,2} are seen in training,
template {3} is HELD OUT (tests paraphrase generalization of edits).

EDITORS (sequential protocol: apply 20 edits one after another):
  FT     : naive fine-tuning on each new fact (classic baseline).
  ROME   : rank-one model editing on the layer-2 FFN down-projection,
           faithful to Meng et al.: key k* = FFN hidden at the edit prompt,
           value v* optimized to produce the new object, update
           W2 += (v* - W2 k*) (C^-1 k*)^T / (k*^T C^-1 k*).
  VERSOR : external memory module, LM FROZEN. Edit = one-shot bind of a
           random versor key with the new object's value versor, superposed
           in sparse Cl(6,0) fields (load <=3/slot, from E2). Retrieval gated
           by cosine of the frozen LM's representation against stored reps;
           deletion = subtract the bind (exact).

METRICS after k in {1,5,10,20} edits:
  efficacy   : edited facts answer the NEW object (train templates)
  paraphrase : edited facts answer the new object on the HELD-OUT template
  locality   : all unedited facts still answer correctly
And a DELETION test after all 20 edits: revert everything, measure restoration.
"""
import json
import time
import numpy as np
import jax
import jax.numpy as jnp

from level2 import init_params, layernorm, sinusoidal
from cln import geo_product, inverse, unit, random_unit_vector

D, H, NL = 32, 64, 2
N_SUBJ, N_REL, N_TPL = 60, 10, 4
FACTS_PER_SUBJ = 2


# ------------------------------------------------------------------ data ----
def build_kb(rng):
    facts = []
    for s in range(N_SUBJ):
        rels = rng.choice(N_REL, size=FACTS_PER_SUBJ, replace=False)
        for r in rels:
            facts.append((s, int(r), int(rng.integers(0, N_SUBJ))))
    return facts          # list of (s, r, o); objects share entity vocab


def encode(facts, tpl):
    # token ids: templates [0..3], subjects [4..63+4), relations [64..74)
    TPL0, SUBJ0, REL0 = 0, N_TPL, N_TPL + N_SUBJ
    X = np.array([[TPL0 + tpl, SUBJ0 + s, REL0 + r] for s, r, _ in facts])
    y = np.array([o for _, _, o in facts])
    return X, y


VOCAB = N_TPL + N_SUBJ + N_REL
PEADD = sinusoidal(D, 3)


# ----------------------------------------------------------------- model ----
def forward(params, tokens, override=None, want_hidden=False):
    """override=(layer, v[B,D]) replaces the FFN output at the LAST position.
    Returns logits and (optionally) layer FFN hidden 'a' + final rep."""
    x = params["emb"][tokens] + jnp.array(PEADD)[None, :, :]
    a_keep = None
    for li, l in enumerate(params["layers"]):
        h = layernorm(x)
        q = h @ l["Wq"].T; k = h @ l["Wk"].T; v = h @ l["Wv"].T
        att = jnp.einsum("bti,bsi->bts", q, k) / jnp.sqrt(D)
        w = jax.nn.softmax(att, -1)
        x = x + jnp.einsum("bts,bsi->bti", w, v) @ l["Wo"].T
        h2 = layernorm(x)
        a = jnp.tanh(h2 @ l["W1"].T)            # (B,T,H)
        f = a @ l["W2"].T
        if override is not None and override[0] == li:
            f = f.at[:, -1, :].set(override[1])
        x = x + f
        if li == NL - 1:
            a_keep = a[:, -1, :]                # FFN hidden at last pos
    logits = x[:, -1, :] @ params["out"].T
    if want_hidden:
        return logits, a_keep, x[:, -1, :]
    return logits


def train_base(seed=0, steps=3500):
    rngnp = np.random.default_rng(seed)
    facts = build_kb(rngnp)
    params = init_params(jax.random.PRNGKey(seed), VOCAB, D,
                         n_layers=NL, hidden=H, n_out=N_SUBJ)

    def loss_fn(p, X, y):
        lg = forward(p, X)
        return -jax.nn.log_softmax(lg)[jnp.arange(len(y)), y].mean()

    gfn = jax.jit(jax.value_and_grad(loss_fn))
    flat, tree = jax.tree_util.tree_flatten(params)
    m = [jnp.zeros_like(p) for p in flat]; v = [jnp.zeros_like(p) for p in flat]
    lr, b1, b2, eps = 3e-3, 0.9, 0.999, 1e-8

    Xs, ys = [], []
    for t in range(3):                           # train templates
        X, y = encode(facts, t)
        Xs.append(X); ys.append(y)
    Xtr = np.concatenate(Xs); ytr = np.concatenate(ys)

    for step in range(1, steps + 1):
        idx = rngnp.integers(0, len(Xtr), size=128)
        p = jax.tree_util.tree_unflatten(tree, flat)
        l, g = gfn(p, jnp.array(Xtr[idx]), jnp.array(ytr[idx]))
        gf, _ = jax.tree_util.tree_flatten(g)
        nf, nm, nv = [], [], []
        for pi, mi, vi, gi in zip(flat, m, v, gf):
            mi = b1 * mi + (1 - b1) * gi
            vi = b2 * vi + (1 - b2) * gi * gi
            nf.append(pi - lr * (mi / (1 - b1 ** step)) /
                      (jnp.sqrt(vi / (1 - b2 ** step)) + eps))
            nm.append(mi); nv.append(vi)
        flat, m, v = nf, nm, nv
    params = jax.tree_util.tree_unflatten(tree, flat)
    return params, facts


def acc(params, facts, tpl, override_facts=None):
    """Accuracy; if override_facts given (dict (s,r)->o_new), score those."""
    if override_facts:
        sel = [(s, r, override_facts[(s, r)]) for (s, r, _) in facts
               if (s, r) in override_facts]
    else:
        sel = facts
    if not sel:
        return 1.0
    X, y = encode(sel, tpl)
    lg = forward(params, jnp.array(X))
    return float((jnp.argmax(lg, -1) == jnp.array(y)).mean())


# --------------------------------------------------------------- editors ----
class FTEditor:
    def __init__(self, params):
        self.params = jax.tree_util.tree_map(lambda x: x, params)

    def edit(self, s, r, o_new):
        X, y = encode([(s, r, o_new)], 0)
        X = jnp.array(X); y = jnp.array(y)

        def loss_fn(p):
            lg = forward(p, X)
            return -jax.nn.log_softmax(lg)[0, y[0]]

        gfn = jax.jit(jax.value_and_grad(loss_fn))
        flat, tree = jax.tree_util.tree_flatten(self.params)
        m = [jnp.zeros_like(p) for p in flat]
        v = [jnp.zeros_like(p) for p in flat]
        for t in range(1, 41):
            p = jax.tree_util.tree_unflatten(tree, flat)
            l, g = gfn(p)
            if l < 0.01:
                break
            gf, _ = jax.tree_util.tree_flatten(g)
            nf, nm, nv = [], [], []
            for pi, mi, vi, gi in zip(flat, m, v, gf):
                mi = 0.9 * mi + 0.1 * gi
                vi = 0.999 * vi + 0.001 * gi * gi
                nf.append(pi - 5e-3 * (mi / (1 - 0.9 ** t)) /
                          (jnp.sqrt(vi / (1 - 0.999 ** t)) + 1e-8))
                nm.append(mi); nv.append(vi)
            flat, m, v = nf, nm, nv
        self.params = jax.tree_util.tree_unflatten(tree, flat)

    def predict_params(self):
        return self.params


class ROMEEditor:
    """Rank-one edit on layer-2 W2 with covariance whitening."""
    def __init__(self, params, facts):
        self.params = jax.tree_util.tree_map(lambda x: x, params)
        # covariance of FFN hidden a over the training corpus
        Xs = np.concatenate([encode(facts, t)[0] for t in range(3)])
        _, A, _ = forward(self.params, jnp.array(Xs), want_hidden=True)
        A = np.array(A)
        C = A.T @ A / len(A) + 1e-3 * np.eye(H)
        self.Cinv = np.linalg.inv(C)
        self.updates = []                       # for the deletion test

    def edit(self, s, r, o_new):
        X, _ = encode([(s, r, o_new)], 0)
        X = jnp.array(X)
        _, a, _ = forward(self.params, X, want_hidden=True)
        kstar = np.array(a[0])                                  # (H,)
        W2 = np.array(self.params["layers"][NL - 1]["W2"])      # (D,H)
        v = jnp.array(W2 @ kstar)

        def loss_v(v):
            lg = forward(self.params, X, override=(NL - 1, v[None, :]))
            return -jax.nn.log_softmax(lg)[0, o_new]

        gfn = jax.jit(jax.value_and_grad(loss_v))
        mv = jnp.zeros_like(v); vv = jnp.zeros_like(v)
        for t in range(1, 51):
            l, g = gfn(v)
            if l < 0.01:
                break
            mv = 0.9 * mv + 0.1 * g
            vv = 0.999 * vv + 0.001 * g * g
            v = v - 0.1 * (mv / (1 - 0.9 ** t)) / (jnp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
        vstar = np.array(v)

        w = self.Cinv @ kstar / (kstar @ self.Cinv @ kstar)     # (H,)
        delta = np.outer(vstar - W2 @ kstar, w)                 # (D,H)
        W2n = W2 + delta
        self.updates.append(delta)
        self.params["layers"][NL - 1]["W2"] = jnp.array(W2n)

    def undo_all(self):
        W2 = np.array(self.params["layers"][NL - 1]["W2"])
        for d in reversed(self.updates):
            W2 = W2 - d
        self.params["layers"][NL - 1]["W2"] = jnp.array(W2)
        self.updates = []

    def predict_params(self):
        return self.params


class VersorEditor:
    """External sparse versor memory; base LM frozen."""
    def __init__(self, params, facts, n2=6, tau=None):
        self.params = params                     # frozen
        self.n2 = n2
        self.D2 = 1 << n2
        rng = np.random.default_rng(99)
        self.valbook = np.stack([self._versor(rng) for _ in range(N_SUBJ)])
        self.entries = []                        # (rep, kversor, slot, o_new, (s,r))
        self.slots = {}                          # slot -> field
        self.rng = rng
        # calibrate gate threshold tau from the BASE model: within-fact
        # (across templates) similarity vs between-fact similarity
        reps = {}
        for t in range(3):
            X, _ = encode(facts, t)
            _, _, R = forward(params, jnp.array(X), want_hidden=True)
            R = np.array(R); R /= np.linalg.norm(R, axis=1, keepdims=True)
            for i, (s, r, _) in enumerate(facts):
                reps.setdefault((s, r), []).append(R[i])
        within = min(min(a @ b for a in v for b in v)
                     for v in reps.values() if len(v) > 1)
        keys = list(reps)
        btw = max(reps[keys[i]][0] @ reps[keys[j]][0]
                  for i in range(0, len(keys), 7)
                  for j in range(i + 1, len(keys), 7))
        self.tau = tau if tau else (within + btw) / 2
        self.calib = {"within_min": float(within), "between_max": float(btw),
                      "tau": float(self.tau)}

    def _versor(self, rng):
        V = np.zeros(self.D2); V[0] = 1.0
        for _ in range(self.n2):
            V = geo_product(V, random_unit_vector(self.n2, rng), self.n2)
        return V

    def _rep(self, s, r, tpl=0):
        X, _ = encode([(s, r, 0)], tpl)
        _, _, R = forward(self.params, jnp.array(X), want_hidden=True)
        R = np.array(R[0]); return R / np.linalg.norm(R)

    def edit(self, s, r, o_new):
        rep = self._rep(s, r)
        kv = self._versor(self.rng)
        slot = len(self.entries) // 3            # load <= 3 per slot (E2)
        if slot not in self.slots:
            self.slots[slot] = np.zeros(self.D2)
        bind = geo_product(kv, self.valbook[o_new], self.n2)
        self.slots[slot] = self.slots[slot] + bind
        self.entries.append([rep, kv, slot, o_new, (s, r), bind])

    def delete(self, s, r):
        for e in list(self.entries):
            if e[4] == (s, r):
                self.slots[e[2]] = self.slots[e[2]] - e[5]   # exact
                self.entries.remove(e)

    def predict(self, X):
        """X: (B,3) tokens. Module gate then base model."""
        _, _, R = forward(self.params, jnp.array(X), want_hidden=True)
        R = np.array(R); R /= np.linalg.norm(R, axis=1, keepdims=True)
        base = np.array(jnp.argmax(forward(self.params, jnp.array(X)), -1))
        out = base.copy()
        if self.entries:
            reps = np.stack([e[0] for e in self.entries])
            sims = R @ reps.T                       # (B, n_entries)
            best = sims.argmax(1)
            for b in range(len(X)):
                if sims[b, best[b]] >= self.tau:
                    e = self.entries[best[b]]
                    rec = geo_product(inverse(e[1], self.n2),
                                      self.slots[e[2]], self.n2)
                    sims_v = self.valbook @ rec / (
                        np.linalg.norm(self.valbook, axis=1) *
                        np.linalg.norm(rec) + 1e-12)
                    out[b] = int(np.argmax(sims_v))
        return out

    def acc_facts(self, facts, tpl, override=None):
        if override:
            sel = [(s, r, override[(s, r)]) for (s, r, _) in facts
                   if (s, r) in override]
        else:
            sel = facts
        if not sel:
            return 1.0
        X, y = encode(sel, tpl)
        return float((self.predict(X) == y).mean())


# ------------------------------------------------------------- protocol -----
def run():
    t0 = time.time()
    rng = np.random.default_rng(33)
    params, facts = train_base()
    base_train = np.mean([acc(params, facts, t) for t in range(3)])
    base_held = acc(params, facts, 3)
    print(f"base LM: train-tpl acc={base_train:.3f}  heldout-tpl acc={base_held:.3f}"
          f"  ({time.time()-t0:.0f}s)")

    # choose 20 edits: new object != old
    idx = rng.choice(len(facts), size=20, replace=False)
    edits = []
    for i in idx:
        s, r, o = facts[i]
        o_new = int((o + 1 + rng.integers(0, N_SUBJ - 1)) % N_SUBJ)
        edits.append((s, r, o_new))
    edited_keys = {(s, r) for s, r, _ in edits}
    unedited = [f for f in facts if (f[0], f[1]) not in edited_keys]

    editors = {
        "FT": FTEditor(params),
        "ROME": ROMEEditor(params, facts),
        "VERSOR": VersorEditor(params, facts),
    }
    print("versor gate calibration:", editors["VERSOR"].calib)

    checkpoints = [1, 5, 10, 20]
    results = {m: {} for m in editors}
    for m, ed in editors.items():
        tm = time.time()
        done = []
        for k, (s, r, o_new) in enumerate(edits, 1):
            ed.edit(s, r, o_new)
            done.append((s, r, o_new))
            if k in checkpoints:
                ov = {(s_, r_): o_ for s_, r_, o_ in done}
                if m == "VERSOR":
                    eff = ed.acc_facts(facts, 0, ov)
                    par = ed.acc_facts(facts, 3, ov)
                    loc = ed.acc_facts(unedited, 0)
                else:
                    p = ed.predict_params()
                    eff = acc(p, facts, 0, ov)
                    par = acc(p, facts, 3, ov)
                    loc = acc(p, unedited, 0)
                results[m][k] = {"efficacy": round(eff, 3),
                                 "paraphrase": round(par, 3),
                                 "locality": round(loc, 3)}
        print(f"{m:>7}: " + "  ".join(
            f"k={k}: eff={v['efficacy']:.2f} par={v['paraphrase']:.2f} "
            f"loc={v['locality']:.2f}" for k, v in results[m].items())
            + f"  ({time.time()-tm:.0f}s)")

    # ---------------- deletion test: revert all 20 edits -------------------
    del_res = {}
    # VERSOR: exact deletion
    ved = editors["VERSOR"]
    for s, r, _ in edits:
        ved.delete(s, r)
    del_res["VERSOR"] = {"restored_locality": round(ved.acc_facts(facts, 0), 3),
                         "mechanism": "exact bind subtraction"}
    # ROME: subtract stored rank-one updates
    red = editors["ROME"]
    red.undo_all()
    del_res["ROME"] = {"restored_locality": round(acc(red.predict_params(), facts, 0), 3),
                       "mechanism": "subtract rank-one updates"}
    # FT: no mechanism
    del_res["FT"] = {"restored_locality": round(acc(editors["FT"].predict_params(), facts, 0), 3),
                     "mechanism": "none (would require retraining)"}
    print("deletion/restoration:", json.dumps(del_res))

    out = {"base": {"train": round(float(base_train), 3),
                    "heldout": round(float(base_held), 3)},
           "calib": editors["VERSOR"].calib,
           "sequential": results, "deletion": del_res,
           "secs": round(time.time() - t0, 1)}
    json.dump(out, open("level3_results.json", "w"), indent=2)
    print(f"\nwrote level3_results.json  ({out['secs']}s total)")
    return out


if __name__ == "__main__":
    run()
