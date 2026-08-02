"""
Experiment (3): Generalization — the hard axis where LLMs are strong.

Exact recall (experiments 1-2) is the easy axis. Here we test whether the
versor memory does anything beyond looking up what was stored. Three tests,
increasingly demanding:

  T1. NOISE ROBUSTNESS (graceful degradation)
      Store (key_i -> value_i). Query with a CORRUPTED key (key_i + noise).
      Does recall still return value_i? A pure hash table fails instantly off
      the exact key; a good associative memory degrades gracefully.

  T2. INTERPOLATION ON A LEARNED MANIFOLD
      Store a smooth function: keys x_i on a curve, values y_i = f(x_i).
      Query with x* BETWEEN stored points (never seen). Is the recalled value
      close to f(x*)? This is what makes an LLM a *model*, not a lookup.

  T3. STRUCTURED ANALOGY (the VSA superpower)
      Encode role-filler bindings (king = man + royal, queen = woman + royal).
      Test the classic "king - man + woman = queen" via algebraic operations.
      This is symbolic generalization LLMs do emergently; VSAs do it exactly.

We report, for each, the versor-memory behavior vs a HASH-TABLE baseline
(which by construction cannot generalize) and, for T2, vs linear interpolation
as a sanity reference.
"""
import json
import numpy as np
from cln import geo_product, inverse, unit, random_unit_vector


def make_versor(n, k, rng):
    V = np.zeros(1 << n); V[0] = 1.0
    for _ in range(k):
        V = geo_product(V, random_unit_vector(n, rng), n)
    return V


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-12))


# ----------------------------------------------------------------------------
# T1: noise robustness
# ----------------------------------------------------------------------------
def test_noise(n, K, noise_levels, rng, trials=20):
    DIM = 1 << n
    results = {nl: [] for nl in noise_levels}
    for _ in range(trials):
        keys = [make_versor(n, n, rng) for _ in range(K)]
        vals = [make_versor(n, n, rng) for _ in range(K)]
        M = np.zeros(DIM)
        for k_, v_ in zip(keys, vals):
            M += geo_product(k_, v_, n)
        VB = np.array(vals); nb = np.linalg.norm(VB, axis=1)
        for nl in noise_levels:
            correct = 0
            for i in range(K):
                noisy = keys[i] + nl * rng.standard_normal(DIM)
                noisy = unit(noisy, n)
                # NOTE: inverse of a non-versor is not exact; use reverse as
                # approximate inverse (valid for near-versor unit multivectors)
                from cln import reverse
                approx_inv = reverse(noisy, n)
                r = geo_product(approx_inv, M, n)
                sims = VB @ r / (nb * np.linalg.norm(r) + 1e-12)
                correct += int(np.argmax(sims) == i)
            results[nl].append(correct / K)
    return {nl: float(np.mean(v)) for nl, v in results.items()}


# ----------------------------------------------------------------------------
# T2: interpolation on a learned 1-D manifold
# ----------------------------------------------------------------------------
def test_interpolation(n, n_anchors, rng, trials=30):
    """Keys = versors along a smooth path parameterized by t in [0,1].
    Values = scalar f(t) encoded as magnitude on a fixed value-versor.
    Query at unseen t*, decode predicted scalar, compare to f(t*)."""
    DIM = 1 << n
    errs_mem, errs_lin = [], []
    # smooth target function
    f = lambda t: np.sin(2 * np.pi * t) + 0.3 * np.cos(5 * np.pi * t)
    for _ in range(trials):
        # a smooth key-path: interpolate between two random unit vectors via slerp-like blend
        a = random_unit_vector(n, rng); b = random_unit_vector(n, rng)
        keyt = lambda t: unit((1 - t) * a + t * b, n)
        valbase = make_versor(n, n, rng)   # direction that carries the scalar
        ts = np.linspace(0, 1, n_anchors)
        M = np.zeros(DIM)
        stored = []
        for t in ts:
            k_ = keyt(t)
            v_ = f(t) * valbase             # value magnitude encodes f(t)
            M += geo_product(k_, v_, n)
            stored.append((t, f(t)))
        # query at midpoints (unseen)
        tq = (ts[:-1] + ts[1:]) / 2
        for t in tq:
            kq = keyt(t)
            from cln import reverse
            r = geo_product(reverse(kq, n), M, n)
            # decode scalar = projection of r onto valbase direction
            pred = np.dot(r, valbase) / (np.dot(valbase, valbase) + 1e-12)
            true = f(t)
            errs_mem.append(abs(pred - true))
            # linear interp reference
            lin = np.interp(t, ts, [f(x) for x in ts])
            errs_lin.append(abs(lin - true))
    return float(np.mean(errs_mem)), float(np.mean(errs_lin)), float(np.std([f(t) for t in np.linspace(0,1,50)]))


# ----------------------------------------------------------------------------
# T3: structured analogy (king - man + woman = queen)
# ----------------------------------------------------------------------------
def test_analogy(n, rng, trials=200):
    """Concepts as superpositions of role-versors. Test vector analogy.
    king  = royal + male
    queen = royal + female
    man   = commoner + male
    woman = commoner + female
    Then king - man + woman should be closest to queen among the 4."""
    hits = 0
    for _ in range(trials):
        royal    = make_versor(n, n, rng)
        commoner = make_versor(n, n, rng)
        male     = make_versor(n, n, rng)
        female   = make_versor(n, n, rng)
        king  = royal + male
        queen = royal + female
        man   = commoner + male
        woman = commoner + female
        query = king - man + woman           # = royal + female = queen (exactly)
        cands = {"king": king, "queen": queen, "man": man, "woman": woman}
        best = max(cands, key=lambda c: cos(query, cands[c]))
        hits += int(best == "queen")
    return hits / trials


def run():
    rng = np.random.default_rng(314)
    n = 6
    out = {"n": n, "dim": 1 << n}

    print(f"Experiment (3): generalization — Cl({n},0), dim {1<<n}\n")

    # T1
    print("T1. Noise robustness (graceful degradation off exact key)")
    nls = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    r1 = test_noise(n, 3, nls, rng)
    print(f"   {'key noise':>10}{'accuracy':>10}{'  vs hash-table':>16}")
    for nl in nls:
        ht = "1.00" if nl == 0.0 else "0.00"
        print(f"   {nl:>10.2f}{r1[nl]:>10.2f}{ht:>16}")
    out["T1_noise"] = r1

    # T2
    print("\nT2. Interpolation on a learned manifold (query unseen midpoints)")
    mem_err, lin_err, signal = test_interpolation(n, 9, rng)
    print(f"   mean abs error (versor memory): {mem_err:.3f}")
    print(f"   mean abs error (linear interp): {lin_err:.3f}")
    print(f"   signal std (for scale):         {signal:.3f}")
    print(f"   -> memory {'interpolates' if mem_err < signal else 'does NOT interpolate'} "
          f"(error {'<' if mem_err<signal else '>='} signal)")
    out["T2_interp"] = {"mem_err": mem_err, "lin_err": lin_err, "signal": signal}

    # T3
    print("\nT3. Structured analogy: king - man + woman = queen ?")
    for nn in [4, 6, 8]:
        acc = test_analogy(nn, rng)
        print(f"   Cl({nn},0) dim {1<<nn:>3}: analogy accuracy = {acc:.2f}")
        out[f"T3_analogy_n{nn}"] = acc

    json.dump(out, open("generalization_results.json", "w"), indent=2)
    print("\nwrote generalization_results.json")
    return out


if __name__ == "__main__":
    run()
