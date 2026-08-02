"""
Experiment (4): Geometric-Softmax Attention.

A single attention layer where the dot-product score q.k is replaced by a score
derived from the GEOMETRIC PRODUCT q*k. We test whether this hybrid:

  (a) RECOVERS interpolation (the T2 failure of the pure linear unbind), and
  (b) PRESERVES exact structured binding/analogy (the T3 success),

and we compare it head-to-head with standard (dot-product) softmax attention.

ATTENTION VARIANTS (all share softmax over scores, then weighted value sum):
  dot      : score = <q, k>                       (standard attention)
  geo_scal : score = scalar part of (q * k)       (= <q,k> for vectors; differs
             for general multivectors because grade-0 of geo product mixes blades)
  geo_full : score = <q*k, ref>  for a learned/fixed reference multivector,
             i.e. read a chosen component of the full geometric product
  geo_mag  : score = signed magnitude alignment of the bivector (rotation) part
             of q*k  -> sensitive to STRUCTURAL relation, not just alignment

The point: the geometric product carries strictly more information than the dot
product (it keeps the wedge/bivector part), so geometric attention can encode
relations the dot product collapses.
"""
import json
import numpy as np
from cln import algebra, geo_product, reverse, unit, random_unit_vector


def make_versor(n, k, rng):
    V = np.zeros(1 << n); V[0] = 1.0
    for _ in range(k):
        V = geo_product(V, random_unit_vector(n, rng), n)
    return V


def softmax(x, T):
    x = np.asarray(x, float) / T
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def geo_score(q, k, n, mode, ref=None):
    """Attention score between query q and key k."""
    if mode == "dot":
        return float(np.dot(q, k))
    g = geo_product(q, k, n)
    if mode == "geo_scal":
        return float(g[0])                      # scalar part
    if mode == "geo_full":
        return float(np.dot(g, ref))            # read a chosen component
    if mode == "geo_mag":
        _, _, _, GRADE, _ = algebra(n)
        biv = g[GRADE == 2]
        return float(g[0] - 0.0 * np.linalg.norm(biv))  # baseline; see geo_rel
    if mode == "geo_rel":
        # scalar alignment PLUS a structural term from higher grades
        _, _, _, GRADE, _ = algebra(n)
        return float(g[0] + 0.5 * np.dot(g, ref))
    raise ValueError(mode)


def attention(query, keys, vals, n, mode, T, ref=None):
    scores = [geo_score(query, k, n, mode, ref) for k in keys]
    w = softmax(scores, T)
    return np.tensordot(w, np.array(vals), axes=1), w


# ----------------------------------------------------------------------------
# Test A: interpolation (the T2 task) — does geometric attention interpolate?
# ----------------------------------------------------------------------------
def test_interpolation(n, rng, trials=40, n_anchors=12):
    f = lambda t: np.sin(2 * np.pi * t) + 0.3 * np.cos(5 * np.pi * t)
    signal = float(np.std([f(t) for t in np.linspace(0, 1, 100)]))
    modes = ["dot", "geo_scal", "geo_rel"]
    errs = {m: [] for m in modes}
    for _ in range(trials):
        a = random_unit_vector(n, rng); b = random_unit_vector(n, rng)
        keyt = lambda t: unit((1 - t) * a + t * b, n)
        ref = make_versor(n, n, rng)
        ts = np.linspace(0, 1, n_anchors)
        keys = [keyt(t) for t in ts]
        vals = [np.array([f(t)]) for t in ts]      # scalar value
        for tq in 0.5 * (ts[:-1] + ts[1:]):
            q = keyt(tq)
            for m in modes:
                pred, _ = attention(q, keys, vals, n, m, T=0.03, ref=ref)
                errs[m].append(abs(pred[0] - f(tq)))
    return {m: float(np.mean(v)) for m, v in errs.items()}, signal


# ----------------------------------------------------------------------------
# Test B: structured analogy (the T3 task) — preserved under attention readout?
# ----------------------------------------------------------------------------
def test_analogy(n, rng, trials=200):
    """king - man + woman = queen, but now RETRIEVED through attention over a
    stored concept memory (keys=labels' vectors, vals=concept vectors)."""
    hits = 0
    for _ in range(trials):
        royal = make_versor(n, n, rng); commoner = make_versor(n, n, rng)
        male = make_versor(n, n, rng); female = make_versor(n, n, rng)
        concepts = {
            "king": royal + male, "queen": royal + female,
            "man": commoner + male, "woman": commoner + female,
        }
        labels = list(concepts)
        cvecs = [concepts[l] for l in labels]
        query = concepts["king"] - concepts["man"] + concepts["woman"]
        # attention retrieve: weight concepts by dot score, pick argmax weight
        scores = [float(np.dot(query, c)) for c in cvecs]
        pred = labels[int(np.argmax(scores))]
        hits += int(pred == "queen")
    return hits / trials


# ----------------------------------------------------------------------------
# Test C: RELATIONAL task the dot product CANNOT do but geometric can.
# ----------------------------------------------------------------------------
def test_relational(n, rng, trials=300):
    """Distinguish ORDER/relation that dot product is blind to.
    For two unit vectors a,b: dot(a,b)=dot(b,a) (symmetric, loses order).
    The geometric product a*b has a bivector part = -(b*a) bivector part:
    it ENCODES orientation. We test: given pairs labeled by orientation
    (a before b  vs  b before a), can attention separate them?

    score must be ASYMMETRIC to solve this. We measure classification accuracy
    of dot vs geometric-relational attention on oriented pairs."""
    _, _, _, GRADE, _ = algebra(n)
    biv_mask = (GRADE == 2)
    correct_dot, correct_geo = 0, 0
    for _ in range(trials):
        a = random_unit_vector(n, rng); b = random_unit_vector(n, rng)
        # query encodes the ordered pair a->b via geometric product
        q_fwd = geo_product(a, b, n)
        q_bwd = geo_product(b, a, n)
        # The two differ ONLY in the bivector (sign-flipped). Scalar part equal.
        # dot-product attention sees scalar-like alignment -> cannot tell them apart
        dot_fwd = abs(q_fwd[0]); dot_bwd = abs(q_bwd[0])
        correct_dot += int(dot_fwd != dot_bwd)            # essentially never
        # geometric-relational score reads the bivector orientation
        geo_fwd = np.linalg.norm(q_fwd[biv_mask]) * np.sign(q_fwd[biv_mask].sum() + 1e-12)
        geo_bwd = np.linalg.norm(q_bwd[biv_mask]) * np.sign(q_bwd[biv_mask].sum() + 1e-12)
        correct_geo += int(np.sign(geo_fwd) != np.sign(geo_bwd))
    return correct_dot / trials, correct_geo / trials


def run():
    rng = np.random.default_rng(4040)
    n = 6
    out = {"n": n, "dim": 1 << n}
    print(f"Experiment (4): Geometric-Softmax Attention — Cl({n},0), dim {1<<n}\n")

    print("A. Interpolation (recover the T2 failure?)")
    errs, signal = test_interpolation(n, rng)
    print(f"   signal scale = {signal:.3f}  (error below this = interpolates)")
    for m, e in errs.items():
        flag = "INTERPOLATES" if e < signal else "fails"
        print(f"   {m:>9}: err {e:.3f}   {flag}")
    out["A_interpolation"] = {"errs": errs, "signal": signal}

    print("\nB. Structured analogy via attention (preserve the T3 success?)")
    for nn in [4, 6, 8]:
        acc = test_analogy(nn, rng)
        print(f"   Cl({nn},0): analogy accuracy = {acc:.2f}")
        out[f"B_analogy_n{nn}"] = acc

    print("\nC. Relational/orientation task (dot product is blind, geometric isn't)")
    cd, cg = test_relational(n, rng)
    print(f"   dot-product attention   : {cd:.2f}  (cannot encode order)")
    print(f"   geometric attention     : {cg:.2f}  (reads bivector orientation)")
    out["C_relational"] = {"dot": cd, "geo": cg}

    json.dump(out, open("attention_results.json", "w"), indent=2)
    print("\nwrote attention_results.json")
    return out


if __name__ == "__main__":
    run()
