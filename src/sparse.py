"""
Experiment (2): Sparse / addressable versor memory.

Instead of ONE field saturated by all K associations, use M small fields
("slots"), each holding only a few associations (below its crosstalk ceiling),
plus an addressing function that routes each key to a slot.

  store(key, value):  s = address(key); field[s] += bind(key, value)
  recall(key):        s = address(key); return cleanup( unbind(key, field[s]) )

The building block is the MIXED-GRADE versor validated in experiment (1)
(product of n unit vectors), whose unbind is norm-preserving (cond = 1).

ADDRESSING SCHEMES COMPARED
  hash   : deterministic hash of the key -> slot.  O(1), no extra state.
           Risk: collisions overload some slots (balls-in-bins).
  anchor : M random anchor versors; key -> argmax cosine(key, anchor_s).
           Content-addressable, but anchors must be searched (O(M)).

KEY QUESTION
  Does total capacity scale ~linearly with the number of slots M?
  If yes, sparsity beats the single-field ceiling. We also report the
  per-slot load and the addressing overhead, to be honest about cost.
"""
import json
import numpy as np
from cln import geo_product, inverse, unit, random_unit_vector


def make_versor(n, k, rng):
    """Mixed-grade unit versor = product of k unit vectors (k=n by default)."""
    V = np.zeros(1 << n); V[0] = 1.0
    for _ in range(k):
        V = geo_product(V, random_unit_vector(n, rng), n)
    return V


class SparseVersorMemory:
    def __init__(self, n, M, scheme="hash", rng=None):
        self.n = n
        self.DIM = 1 << n
        self.M = M
        self.scheme = scheme
        self.rng = rng or np.random.default_rng(0)
        self.fields = [np.zeros(self.DIM) for _ in range(M)]
        # per-slot codebooks for clean-up (store the (key,value) we put in)
        self.slot_keys = [[] for _ in range(M)]
        self.slot_kinv = [[] for _ in range(M)]
        self.slot_vals = [[] for _ in range(M)]
        if scheme == "anchor":
            self.anchors = np.array([make_versor(n, n, self.rng) for _ in range(M)])
            self.anchor_norm = np.linalg.norm(self.anchors, axis=1)

    def address(self, key):
        if self.scheme == "hash":
            # deterministic, content-derived slot from the key's bytes
            h = hash(key.tobytes())
            return h % self.M
        else:  # anchor: content-addressable nearest anchor
            sims = self.anchors @ key / (self.anchor_norm * np.linalg.norm(key) + 1e-12)
            return int(np.argmax(sims))

    def store(self, key, value):
        s = self.address(key)
        self.fields[s] = self.fields[s] + geo_product(key, value, self.n)
        self.slot_keys[s].append(key)
        self.slot_kinv[s].append(inverse(key, self.n))
        self.slot_vals[s].append(value)

    def recall(self, key, key_index_in_slot=None):
        s = self.address(key)
        VB = self.slot_vals[s]
        if not VB:
            return None
        # find this key's inverse (we stored it); in practice unbind by key
        ki = inverse(key, self.n)
        r = geo_product(ki, self.fields[s], self.n)
        VBa = np.array(VB)
        nb = np.linalg.norm(VBa, axis=1)
        sims = VBa @ r / (nb * np.linalg.norm(r) + 1e-12)
        return int(np.argmax(sims)), s


def evaluate(n, M, scheme, load_per_slot, rng, trials=5):
    """Store M*load associations; measure global clean-up accuracy."""
    accs = []
    loads = []
    for _ in range(trials):
        mem = SparseVersorMemory(n, M, scheme, np.random.default_rng(rng.integers(1 << 30)))
        K = M * load_per_slot
        keys = [make_versor(n, n, mem.rng) for _ in range(K)]
        vals = [make_versor(n, n, mem.rng) for _ in range(K)]
        for k_, v_ in zip(keys, vals):
            mem.store(k_, v_)
        # accuracy: does recall(key_i) return value_i?
        correct = 0
        for i in range(K):
            s = mem.address(keys[i])
            # index of this exact value within its slot
            true_idx = None
            for j, v in enumerate(mem.slot_vals[s]):
                if np.array_equal(v, vals[i]):
                    true_idx = j
                    break
            pred, _ = mem.recall(keys[i])
            correct += int(pred == true_idx)
        accs.append(correct / K)
        loads.append(np.mean([len(x) for x in mem.slot_vals]))
    return float(np.mean(accs)), float(np.mean(loads))


def run():
    rng = np.random.default_rng(2025)
    n = 6                      # dim 64 building block (validated grade-mixed)
    print(f"Sparse versor memory — building block Cl({n},0), dim {1<<n}")
    print("Total capacity @>=90% accuracy vs number of slots M\n")

    out = {"n": n, "dim": 1 << n, "schemes": {}}
    for scheme in ["hash", "anchor"]:
        print(f"--- addressing: {scheme} ---")
        print(f"{'M slots':>8}{'load/slot':>10}{'K total':>9}{'accuracy':>10}{'cap@90%':>9}")
        rows = []
        # pick a per-slot load just under the single-field ceiling (~3-4 for dim64)
        for M in [1, 2, 4, 8, 16, 32, 64]:
            # find max total K with accuracy>=0.90 by scanning load_per_slot
            best_cap = 0
            best = None
            for load in [1, 2, 3, 4, 6, 8]:
                acc, real_load = evaluate(n, M, scheme, load, rng, trials=4)
                if acc >= 0.90:
                    best_cap = max(best_cap, M * load)
                    best = (load, real_load, acc)
            if best:
                load, real_load, acc = best
                print(f"{M:>8}{real_load:>10.1f}{M*load:>9}{acc:>10.2f}{best_cap:>9}")
                rows.append({"M": M, "load": load, "K": M * load,
                             "acc": round(acc, 3), "cap": best_cap})
            else:
                print(f"{M:>8}{'-':>10}{'-':>9}{'<90%':>10}{0:>9}")
                rows.append({"M": M, "cap": 0})
        out["schemes"][scheme] = rows
        print()
    json.dump(out, open("sparse_results.json", "w"), indent=2)
    print("wrote sparse_results.json")
    return out


if __name__ == "__main__":
    run()
