"""
Versor Associative Memory (VAM) on Cl(n,0), and a capacity experiment.

MODEL
-----
We store K associations (key_i -> value_i). Keys and values are random unit
VERSORS (products of unit vectors), so that binding is invertible exactly.

Two binding schemes are compared:

  A) ADDITIVE SUPERPOSITION (VSA-style):
        M = sum_i  bind(key_i, value_i)        (bind = geometric product)
     recall(q) = unbind(key_q, M) = key_q^{-1} * M
        = value_q  +  sum_{i!=q} key_q^{-1} * key_i * value_i   (crosstalk)
     The crosstalk term is the noise that limits capacity.

  B) MULTIPLICATIVE FIELD (thesis-style versor chain):
        M = V_K * ... * V_2 * V_1   where V_i = value_i * key_i^{-1} ...
     (kept for reference; additive is the one with a clean capacity story.)

RECALL FIDELITY
---------------
We measure, for each stored pair, the cosine similarity between the noisy
recalled multivector and the true stored value, AND whether a nearest-neighbour
"clean-up" against the stored value codebook returns the correct value
(this clean-up step is the non-linearity, exactly as in VSA).

OUTPUT: capacity curves -> capacity_results.json
"""
import json
import numpy as np
from cln import algebra, geo_product, inverse, unit, random_unit_vector


def make_versor(n, k, rng):
    """Genuine unit versor = geometric product of k random unit vectors."""
    V = np.zeros(1 << n)
    V[0] = 1.0
    for _ in range(k):
        V = geo_product(V, random_unit_vector(n, rng), n)
    return V


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_memory(n, K, rng, versor_grade=2):
    """Create K key/value versor pairs and the superposed memory M."""
    DIM = 1 << n
    keys = [make_versor(n, versor_grade, rng) for _ in range(K)]
    vals = [make_versor(n, versor_grade, rng) for _ in range(K)]
    M = np.zeros(DIM)
    for k_, v_ in zip(keys, vals):
        M = M + geo_product(k_, v_, n)   # bind then superpose
    return keys, vals, M


def evaluate(n, K, rng, versor_grade=2, trials=1):
    """Return mean cosine of noisy recall and clean-up accuracy over trials."""
    cos_acc = []
    cleanup_acc = []
    for _ in range(trials):
        keys, vals, M = build_memory(n, K, rng, versor_grade)
        val_codebook = np.array(vals)          # K x DIM
        correct = 0
        cosines = []
        for i in range(K):
            ki_inv = inverse(keys[i], n)
            recalled = geo_product(ki_inv, M, n)   # noisy value_i + crosstalk
            cosines.append(cosine(recalled, vals[i]))
            # clean-up: nearest stored value by cosine
            sims = val_codebook @ recalled / (
                np.linalg.norm(val_codebook, axis=1) * np.linalg.norm(recalled) + 1e-12)
            if int(np.argmax(sims)) == i:
                correct += 1
        cos_acc.append(np.mean(cosines))
        cleanup_acc.append(correct / K)
    return float(np.mean(cos_acc)), float(np.mean(cleanup_acc))


def run():
    rng = np.random.default_rng(7)
    dims = {2: 4, 3: 8, 4: 16, 5: 32, 6: 64, 7: 128, 8: 256}
    Ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    results = {}
    print(f"{'n':>3} {'dim':>5} | capacity curve (K -> cleanup accuracy)")
    print("-" * 70)
    for n, dim in dims.items():
        row = {}
        line = f"{n:>3} {dim:>5} | "
        for K in Ks:
            if K > 4 * dim:        # don't bother far past plausible capacity
                continue
            cos_m, acc = evaluate(n, K, rng, versor_grade=2,
                                  trials=5 if K <= 64 else 2)
            row[K] = {"cosine": round(cos_m, 3), "cleanup_acc": round(acc, 3)}
            line += f"{K}:{acc:.2f} "
        results[n] = {"dim": dim, "curve": row}
        print(line)
    # Find empirical capacity: max K with cleanup_acc >= 0.95
    print("\nEmpirical capacity (max K with >=95% clean-up accuracy):")
    cap = {}
    for n, r in results.items():
        good = [K for K, v in r["curve"].items() if v["cleanup_acc"] >= 0.95]
        cap[n] = max(good) if good else 0
        print(f"  Cl({n},0) dim={r['dim']:>3}:  capacity ~ {cap[n]} associations"
              f"   ({cap[n]/r['dim']:.2f} x dim)")
    with open("capacity_results.json", "w") as f:
        json.dump({"results": results, "capacity": cap}, f, indent=2)
    print("\nwrote capacity_results.json")
    return results, cap


if __name__ == "__main__":
    run()
