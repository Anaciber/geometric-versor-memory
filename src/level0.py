"""
Level 0: trained / explicit-storage baselines vs the versor memory,
on the IDENTICAL task and protocol of E1-E3:

  store K random (key -> value) pairs in dimension d, query with the key,
  clean-up = nearest stored value by cosine, accuracy over all K.

SYSTEMS (with their storage budgets, stated honestly):
  versor   : superposed Cl(n,0) field. Storage = d floats (ONE multivector).
             One-shot write (1 geometric product). Exact delete (subtract bind).
  hebbian  : linear associator W = sum_i v_i k_i^T (Kohonen/Anderson).
             Storage = d^2 floats. One-shot write (outer product).
             Exact delete (subtract outer product).
  hopfield : modern Hopfield / attention memory with EXPLICIT pattern storage.
             retrieval = softmax(beta * K q) V  (Ramsauer et al. 2020).
             Storage = 2*K*d floats (grows with content). One-shot append.
             Exact delete (remove row).
  mlp      : 2-layer MLP trained by Adam to map keys->values under a FIXED
             step budget (gradient memorization, the LLM regime).
             Storage = #params floats. Write = epochs of gradient descent.
             No exact delete (retraining).

METRICS:
  capacity@90  : max K with >= 90% clean-up accuracy
  density      : capacity / storage floats at that capacity
  write cost   : one-shot vs gradient steps
  noise@0.2    : accuracy at capacity/2 load when the query key has 0.2 noise
"""
import json
import time
import numpy as np
from cln import geo_product, inverse, unit, random_unit_vector


# ---------------------------------------------------------------- versor ----
def make_versor(n, k, rng):
    V = np.zeros(1 << n); V[0] = 1.0
    for _ in range(k):
        V = geo_product(V, random_unit_vector(n, rng), n)
    return V


class VersorMemory:
    def __init__(self, n, rng):
        self.n = n
        self.d = 1 << n
        self.rng = rng
        self.M = np.zeros(self.d)
        self.keys, self.vals = [], []

    def write(self, k, v):
        self.M = self.M + geo_product(k, v, self.n)
        self.keys.append(k); self.vals.append(v)

    def query(self, q, noisy=False):
        ki = inverse(q, self.n) if not noisy else None
        if noisy:
            from cln import reverse
            ki = reverse(q, self.n)      # approximate inverse for non-versor
        return geo_product(ki, self.M, self.n)

    @property
    def storage(self):
        return self.d


# --------------------------------------------------------------- hebbian ----
class HebbianMemory:
    def __init__(self, d, rng):
        self.d = d
        self.W = np.zeros((d, d))
        self.keys, self.vals = [], []

    def write(self, k, v):
        self.W += np.outer(v, k)
        self.keys.append(k); self.vals.append(v)

    def query(self, q, noisy=False):
        return self.W @ q

    @property
    def storage(self):
        return self.d * self.d


# -------------------------------------------------------------- hopfield ----
class HopfieldMemory:
    """Modern Hopfield retrieval = attention over explicitly stored patterns."""
    def __init__(self, d, rng, beta=50.0):
        self.d = d; self.beta = beta
        self.K = []; self.V = []

    def write(self, k, v):
        self.K.append(k); self.V.append(v)

    def query(self, q, noisy=False):
        Karr = np.array(self.K); Varr = np.array(self.V)
        s = self.beta * (Karr @ q)
        s -= s.max()
        w = np.exp(s); w /= w.sum()
        return w @ Varr

    @property
    def storage(self):
        return 2 * len(self.K) * self.d


# ------------------------------------------------------------------- mlp ----
class MLPMemory:
    """2-layer tanh MLP trained with Adam under a fixed step budget."""
    def __init__(self, d, rng, hidden=64, steps=3000, lr=1e-2):
        self.d, self.h, self.steps, self.lr = d, hidden, steps, lr
        self.rng = rng
        s1 = 1.0 / np.sqrt(d); s2 = 1.0 / np.sqrt(hidden)
        self.W1 = rng.standard_normal((hidden, d)) * s1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((d, hidden)) * s2
        self.b2 = np.zeros(d)
        self.keys, self.vals = [], []
        self._trained = False

    def write(self, k, v):
        self.keys.append(k); self.vals.append(v)
        self._trained = False

    def _train(self):
        X = np.array(self.keys); Y = np.array(self.vals)
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v_ = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.steps + 1):
            H = np.tanh(X @ self.W1.T + self.b1)      # (K,h)
            P = H @ self.W2.T + self.b2               # (K,d)
            E = P - Y                                 # (K,d)
            gW2 = E.T @ H / len(X)
            gb2 = E.mean(0)
            dH = (E @ self.W2) * (1 - H * H)
            gW1 = dH.T @ X / len(X)
            gb1 = dH.mean(0)
            grads = [gW1, gb1, gW2, gb2]
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = b1 * m[i] + (1 - b1) * g
                v_[i] = b2 * v_[i] + (1 - b2) * g * g
                mh = m[i] / (1 - b1 ** t)
                vh = v_[i] / (1 - b2 ** t)
                p -= self.lr * mh / (np.sqrt(vh) + eps)
        self._trained = True

    def query(self, q, noisy=False):
        if not self._trained:
            self._train()
        h = np.tanh(self.W1 @ q + self.b1)
        return self.W2 @ h + self.b2

    @property
    def storage(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


# ------------------------------------------------------------ evaluation ----
def gen_pairs(system, n, d, K, rng):
    if system == "versor":
        keys = [make_versor(n, n, rng) for _ in range(K)]
        vals = [make_versor(n, n, rng) for _ in range(K)]
    else:
        keys = [unit_vec(d, rng) for _ in range(K)]
        vals = [unit_vec(d, rng) for _ in range(K)]
    return keys, vals


def unit_vec(d, rng):
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def build(system, n, d, rng):
    if system == "versor":
        return VersorMemory(n, rng)
    if system == "hebbian":
        return HebbianMemory(d, rng)
    if system == "hopfield":
        return HopfieldMemory(d, rng)
    if system == "mlp":
        return MLPMemory(d, rng)
    raise ValueError(system)


def accuracy(system, n, d, K, rng, noise=0.0, trials=4):
    accs = []
    for _ in range(trials):
        mem = build(system, n, d, rng)
        keys, vals = gen_pairs(system, n, d, K, rng)
        for k, v in zip(keys, vals):
            mem.write(k, v)
        VB = np.array(vals); nb = np.linalg.norm(VB, axis=1)
        correct = 0
        for i in range(K):
            q = keys[i]
            if noise > 0:
                q = q + noise * rng.standard_normal(len(q))
                if system == "versor":
                    q = unit(q, n)
                else:
                    q = q / np.linalg.norm(q)
            r = mem.query(q, noisy=(noise > 0))
            sims = VB @ r / (nb * np.linalg.norm(r) + 1e-12)
            correct += int(np.argmax(sims) == i)
        accs.append(correct / K)
    return float(np.mean(accs))


def capacity(system, n, d, rng, thresh=0.90, Ks=None):
    if Ks is None:
        Ks = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 26, 32, 44, 64, 90, 128, 180, 256]
    cap = 0
    curve = {}
    for K in Ks:
        tr = 4 if system != "mlp" else 3
        a = accuracy(system, n, d, K, rng, trials=tr)
        curve[K] = round(a, 3)
        if a >= thresh:
            cap = K
        # stop scanning when clearly past the cliff (2 consecutive fails)
        fails = [k for k in curve if curve[k] < thresh and k > cap]
        if len(fails) >= 2 and K > 2 * max(cap, 1):
            break
    return cap, curve


def run():
    rng = np.random.default_rng(606)
    n = 6; d = 1 << n
    systems = ["versor", "hebbian", "hopfield", "mlp"]
    out = {"n": n, "d": d, "systems": {}}

    print(f"LEVEL 0 — trained/explicit baselines vs versor memory  (d = {d})\n")
    print(f"{'system':>9} {'cap@90':>7} {'storage':>9} {'density':>10} "
          f"{'write':>10} {'delete':>7} {'noise@0.2':>10}")
    print("-" * 70)

    meta = {
        "versor":   {"write": "1-shot",  "delete": "exact"},
        "hebbian":  {"write": "1-shot",  "delete": "exact"},
        "hopfield": {"write": "append",  "delete": "exact"},
        "mlp":      {"write": "3k steps","delete": "retrain"},
    }

    for s in systems:
        t0 = time.time()
        cap, curve = capacity(s, n, d, rng)
        # storage at capacity
        mem = build(s, n, d, rng)
        if s == "hopfield":
            storage = 2 * max(cap, 1) * d
        else:
            storage = mem.storage
        dens = cap / storage if storage else 0.0
        # noise robustness at half capacity load
        Kn = max(2, cap // 2) if cap > 0 else 2
        noi = accuracy(s, n, d, Kn, rng, noise=0.2, trials=4)
        dt = time.time() - t0
        print(f"{s:>9} {cap:>7} {storage:>9,} {dens:>10.4f} "
              f"{meta[s]['write']:>10} {meta[s]['delete']:>7} {noi:>10.2f}"
              f"   ({dt:.0f}s)")
        out["systems"][s] = {
            "capacity90": cap, "storage": storage,
            "density": round(dens, 5), "curve": curve,
            "noise02_acc": round(noi, 3), **meta[s],
        }

    json.dump(out, open("level0_results.json", "w"), indent=2)
    print("\nwrote level0_results.json")
    return out


if __name__ == "__main__":
    run()
