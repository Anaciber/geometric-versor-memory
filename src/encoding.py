"""
Experiment (1): does the encoding scheme fix the capacity problem?

We compare key/value code families at equal dimension and identical protocol
(bind = geometric product, superpose = add, recall = unbind by inverse,
clean-up = nearest stored value). We measure clean-up accuracy vs K.

CODE FAMILIES
  v2  : versor = product of 2 unit vectors      (the thesis model, grade-2 heavy)
  v_n : versor = product of n unit vectors       (mixed grade, exact inverse)
  rot : even-grade rotor = exp of a bivector blade(approx via product of vectors,
        kept even)                                (spinor-like)
  full: random full-spectrum unit multivector + EXACT matrix inverse
        (not a versor; upper bound a la HRR-on-Clifford)

KEY QUESTION
  If 'full' scales ~linearly with dim but the versor families don't, the
  capacity limit is a property of the VERSOR CONSTRAINT, not of the algebra.
  If even 'full' scales poorly at these dims, the limit is the small dimension
  itself (superposition noise), and the fix is many small fields, not bigger codes.
"""
import json
import numpy as np
from cln import algebra, geo_product, inverse, unit, random_unit_vector


def make_versor(n, k, rng):
    V = np.zeros(1 << n); V[0] = 1.0
    for _ in range(k):
        V = geo_product(V, random_unit_vector(n, rng), n)
    return V


def gp_matrix(a, n):
    DIM, SIGN, IDX, _, _ = algebra(n)
    L = np.zeros((DIM, DIM))
    nz = np.nonzero(a)[0]
    for i in nz:
        for j in range(DIM):
            L[IDX[i, j], j] += SIGN[i, j] * a[i]
    return L


def full_inverse(a, n):
    DIM = 1 << n
    ident = np.zeros(DIM); ident[0] = 1
    try:
        return np.linalg.solve(gp_matrix(a, n), ident)
    except np.linalg.LinAlgError:
        return None


def gen_code(family, n, rng):
    DIM = 1 << n
    if family == "v2":
        return make_versor(n, 2, rng), "versor"
    if family == "v_n":
        return make_versor(n, n, rng), "versor"
    if family == "rot":          # even versor: product of an even # of vectors
        return make_versor(n, max(2, n - (n % 2)), rng), "versor"
    if family == "full":
        return unit(rng.standard_normal(DIM), n), "full"
    raise ValueError(family)


def accuracy(family, n, K, rng, trials):
    DIM = 1 << n
    accs = []
    for _ in range(trials):
        keys, kinv, vals = [], [], []
        for _ in range(K):
            k_, kind = gen_code(family, n, rng)
            keys.append(k_)
            kinv.append(inverse(k_, n) if kind == "versor" else full_inverse(k_, n))
            v_, _ = gen_code(family, n, rng)
            vals.append(v_)
        M = np.zeros(DIM)
        for k_, v_ in zip(keys, vals):
            M = M + geo_product(k_, v_, n)
        VB = np.array(vals); nb = np.linalg.norm(VB, axis=1)
        correct = 0
        for i in range(K):
            if kinv[i] is None:
                continue
            r = geo_product(kinv[i], M, n)
            sims = VB @ r / (nb * np.linalg.norm(r) + 1e-12)
            correct += int(np.argmax(sims) == i)
        accs.append(correct / K)
    return float(np.mean(accs))


def capacity_at(family, n, rng, thresh=0.95, trials=8):
    dim = 1 << n
    cap = 0
    for K in range(1, dim + 1):
        if accuracy(family, n, K, rng, trials) >= thresh:
            cap = K
        else:
            break
    return cap


def run():
    rng = np.random.default_rng(2024)
    families = ["v2", "v_n", "rot", "full"]
    out = {}
    print("Capacity @95% clean-up accuracy, by encoding family and dimension")
    print(f"{'n':>3}{'dim':>6} | " + "".join(f"{f:>8}" for f in families))
    print("-" * 50)
    for n in [4, 5, 6, 7]:
        dim = 1 << n
        row = {}
        line = f"{n:>3}{dim:>6} | "
        for f in families:
            # 'full' inverse is O(dim^3) per key; keep trials modest at high dim
            tr = 8 if (f != "full" or n <= 6) else 4
            cap = capacity_at(f, n, rng, trials=tr)
            row[f] = {"cap": cap, "ratio": round(cap / dim, 3)}
            line += f"{cap:>8}"
        out[n] = {"dim": dim, "families": row}
        print(line)
    print("\nCapacity / dimension ratio (higher = better scaling):")
    print(f"{'n':>3}{'dim':>6} | " + "".join(f"{f:>8}" for f in families))
    for n, r in out.items():
        line = f"{n:>3}{r['dim']:>6} | "
        for f in families:
            line += f"{r['families'][f]['ratio']:>8.3f}"
        print(line)
    json.dump(out, open("encoding_results.json", "w"), indent=2)
    print("\nwrote encoding_results.json")
    return out


if __name__ == "__main__":
    run()
