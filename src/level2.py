"""
Level 2: versor position binding inside a TRAINED transformer.

Hypothesis (falsifiable): multiplicative position binding by powers of a unit
VERSOR (x_t -> L_R^t x_t, an orthogonal map from Cl(5,0) left-multiplication)
matches or beats additive sinusoidal PE and RoPE on order-sensitive tasks,
and extrapolates to unseen lengths at least as well as RoPE.

Verified theory (test_theory): versor left-mult is orthogonal (cond=1), hence
  <P_t q, P_s k> = <q, P_{s-t} k>   -> relative-position attention, like RoPE.
RoPE is the special case where R is a product of COMMUTING 2-D rotors; a full
versor mixes all grades and its blocks do not commute.

VARIANTS (identical model, only the position mechanism differs):
  none   : no position information (control; order tasks should be unsolvable)
  sin    : additive sinusoidal PE on embeddings
  rope   : standard RoPE rotation matrices applied to q,k per position
  versor : powers of a fixed random unit versor applied to q,k per position

TASKS:
  retrieve : input [t_1..t_L, Q_p]; output = token at position p (V-way clf).
  order    : input [t_1..t_L, SEP, a, b]; output = 1 if a precedes b (binary).
  extrapolation: train 'order' on L in [6,12], evaluate on L in {16,20,24}.

Model: 2-layer, 1-head transformer, d_model=32 (= Cl(5,0)), trained with Adam.
"""
import argparse
import json
import time
import numpy as np
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, ".")
from cln import geo_product, algebra, random_unit_vector


# ------------------------------------------------------------ position maps -
def make_versor(n, k, rng):
    V = np.zeros(1 << n); V[0] = 1.0
    for _ in range(k):
        V = geo_product(V, random_unit_vector(n, rng), n)
    return V


def left_mult_matrix(a, n):
    DIM, SIGN, IDX, _, _ = algebra(n)
    L = np.zeros((DIM, DIM))
    for i in range(DIM):
        if a[i] == 0:
            continue
        for j in range(DIM):
            L[IDX[i, j], j] += SIGN[i, j] * a[i]
    return L


def position_matrices(kind, D, Lmax, rng):
    """Return (Lmax, D, D) array P with P[t] applied to q,k at position t."""
    if kind in ("none", "sin"):
        return np.broadcast_to(np.eye(D), (Lmax, D, D)).copy()
    if kind == "rope":
        # block-diagonal 2D rotations, standard frequencies
        P = np.zeros((Lmax, D, D))
        half = D // 2
        freqs = 10000.0 ** (-np.arange(half) / half)
        for t in range(Lmax):
            ang = t * freqs
            c, s = np.cos(ang), np.sin(ang)
            M = np.zeros((D, D))
            for i in range(half):
                M[2 * i, 2 * i] = c[i];     M[2 * i, 2 * i + 1] = -s[i]
                M[2 * i + 1, 2 * i] = s[i]; M[2 * i + 1, 2 * i + 1] = c[i]
            P[t] = M
        return P
    if kind == "versor_s":
        # spectrum-matched multiplicative binding: the RoPE generator
        # conjugated into RANDOM planes (rotor one-parameter group with
        # designed frequencies; non-axis-aligned).
        rngq = np.random.default_rng(13)
        Q, _ = np.linalg.qr(rngq.standard_normal((D, D)))
        B = position_matrices("rope", D, Lmax, rng)
        return np.einsum("ij,tjk,lk->til", Q, B, Q)
    if kind == "versor":
        n = int(np.log2(D))
        R = make_versor(n, n, rng)
        LR = left_mult_matrix(R, n)
        P = np.zeros((Lmax, D, D))
        M = np.eye(D)
        for t in range(Lmax):
            P[t] = M
            M = LR @ M
        return P
    raise ValueError(kind)


def sinusoidal(D, Lmax):
    pe = np.zeros((Lmax, D))
    pos = np.arange(Lmax)[:, None]
    div = 10000.0 ** (-np.arange(0, D, 2) / D)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


# ------------------------------------------------------------------- model --
def init_params(key, vocab, D, n_layers=2, hidden=64, n_out=None):
    ks = jax.random.split(key, 3 + 6 * n_layers)
    p = {"emb": jax.random.normal(ks[0], (vocab, D)) * 0.5,
         "out": jax.random.normal(ks[1], (n_out, D)) * (1 / np.sqrt(D)),
         "layers": []}
    i = 2
    for _ in range(n_layers):
        l = {
            "Wq": jax.random.normal(ks[i], (D, D)) / np.sqrt(D),
            "Wk": jax.random.normal(ks[i + 1], (D, D)) / np.sqrt(D),
            "Wv": jax.random.normal(ks[i + 2], (D, D)) / np.sqrt(D),
            "Wo": jax.random.normal(ks[i + 3], (D, D)) / np.sqrt(D),
            "W1": jax.random.normal(ks[i + 4], (hidden, D)) / np.sqrt(D),
            "W2": jax.random.normal(ks[i + 5], (D, hidden)) / np.sqrt(hidden),
        }
        p["layers"].append(l)
        i += 6
    return p


def layernorm(x):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-6)


def forward(params, tokens, P, pe_add):
    """tokens: (B, T) int; P: (T, D, D) per-position q/k maps; pe_add: (T, D)."""
    x = params["emb"][tokens] + pe_add[None, :, :]      # (B,T,D)
    for l in params["layers"]:
        h = layernorm(x)
        q = h @ l["Wq"].T
        k = h @ l["Wk"].T
        v = h @ l["Wv"].T
        # apply position maps to q and k (relative-position mechanism)
        q = jnp.einsum("tij,btj->bti", P, q)
        k = jnp.einsum("tij,btj->bti", P, k)
        att = jnp.einsum("bti,bsi->bts", q, k) / jnp.sqrt(q.shape[-1])
        w = jax.nn.softmax(att, axis=-1)
        x = x + jnp.einsum("bts,bsi->bti", w, v) @ l["Wo"].T
        h2 = layernorm(x)
        x = x + jnp.tanh(h2 @ l["W1"].T) @ l["W2"].T
    return x[:, -1, :] @ params["out"].T                # logits at last slot


# ------------------------------------------------------------------- tasks --
def gen_retrieve(rng, B, L, vocab_tok):
    """[t_1..t_L, Q_p] -> label = t_p. Query ids start at vocab_tok."""
    toks = np.stack([rng.choice(vocab_tok, size=L, replace=False)
                     for _ in range(B)])
    p = rng.integers(0, L, size=B)
    q = vocab_tok + p                                   # query token encodes p
    X = np.concatenate([toks, q[:, None]], axis=1)
    y = toks[np.arange(B), p]
    return X, y


def gen_order(rng, B, L, vocab_tok, SEP):
    """[t_1..t_L, SEP, a, b] -> 1 if a precedes b."""
    toks = np.stack([rng.choice(vocab_tok, size=L, replace=False)
                     for _ in range(B)])
    i = rng.integers(0, L, size=B)
    j = (i + 1 + rng.integers(0, L - 1, size=B)) % L
    a = toks[np.arange(B), i]; b = toks[np.arange(B), j]
    y = (i < j).astype(np.int64)
    X = np.concatenate([toks, np.full((B, 1), SEP), a[:, None], b[:, None]], 1)
    return X, y


# ---------------------------------------------------------------- training --
def train(task, pe, seed, steps, D=32, Lmax=40):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    if task == "retrieve":
        L = 10; vocab_tok = 40
        vocab = vocab_tok + L           # queries Q_0..Q_{L-1}
        n_out = vocab_tok
        T = L + 1
        gen = lambda B, LL=None: gen_retrieve(rng, B, LL or L, vocab_tok)
    else:
        vocab_tok = 40; SEP = vocab_tok
        vocab = vocab_tok + 1
        n_out = 2
        Ltr = (6, 12)
        T = Ltr[1] + 3
        def gen(B, LL=None):
            LL = LL if LL else int(rng.integers(Ltr[0], Ltr[1] + 1))
            return gen_order(rng, B, LL, vocab_tok, SEP)

    Pfull = position_matrices(pe, D, Lmax, np.random.default_rng(7))
    peadd_full = sinusoidal(D, Lmax) if pe == "sin" else np.zeros((Lmax, D))

    params = init_params(key, vocab, D, n_out=n_out)

    def loss_fn(params, X, y, P, padd):
        logits = forward(params, X, P, padd)
        ll = jax.nn.log_softmax(logits)
        return -ll[jnp.arange(len(y)), y].mean()

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Adam
    flat, tree = jax.tree_util.tree_flatten(params)
    m = [jnp.zeros_like(p) for p in flat]
    v = [jnp.zeros_like(p) for p in flat]
    lr, b1, b2, eps = 3e-3, 0.9, 0.999, 1e-8

    @jax.jit
    def adam_step(flat, m, v, grads_flat, t):
        out_f, out_m, out_v = [], [], []
        for p, mi, vi, g in zip(flat, m, v, grads_flat):
            mi = b1 * mi + (1 - b1) * g
            vi = b2 * vi + (1 - b2) * g * g
            mh = mi / (1 - b1 ** t)
            vh = vi / (1 - b2 ** t)
            out_f.append(p - lr * mh / (jnp.sqrt(vh) + eps))
            out_m.append(mi); out_v.append(vi)
        return out_f, out_m, out_v

    B = 128
    for step in range(1, steps + 1):
        X, y = gen(B)
        T_cur = X.shape[1]
        P = jnp.array(Pfull[:T_cur]); padd = jnp.array(peadd_full[:T_cur])
        params = jax.tree_util.tree_unflatten(tree, flat)
        l, grads = grad_fn(params, jnp.array(X), jnp.array(y), P, padd)
        gflat, _ = jax.tree_util.tree_flatten(grads)
        flat, m, v = adam_step(flat, m, v, gflat, step)
    params = jax.tree_util.tree_unflatten(tree, flat)

    def evaluate(LL=None, N=2000):
        X, y = gen(N, LL)
        T_cur = X.shape[1]
        P = jnp.array(Pfull[:T_cur]); padd = jnp.array(peadd_full[:T_cur])
        logits = forward(params, jnp.array(X), P, padd)
        return float((jnp.argmax(logits, -1) == jnp.array(y)).mean())

    res = {"in_dist": evaluate()}
    if task == "order":
        for LL in [16, 20, 24]:
            res[f"L{LL}"] = evaluate(LL)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["retrieve", "order"])
    ap.add_argument("--pe", required=True,
                    choices=["none", "sin", "rope", "versor", "versor_s"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    t0 = time.time()
    res = train(a.task, a.pe, a.seed, a.steps)
    res.update({"task": a.task, "pe": a.pe, "seed": a.seed,
                "steps": a.steps, "secs": round(time.time() - t0, 1)})
    print(json.dumps(res))
    with open(f"l2_{a.task}_{a.pe}_s{a.seed}.json", "w") as f:
        json.dump(res, f)
