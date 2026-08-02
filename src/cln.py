"""
Cl(n,0) geometric algebra — generalized & vectorized for memory experiments.

We need to vary n (and thus dimension 2^n) to study how associative-memory
capacity scales. This builds the sign/index tables once per n and implements
the geometric product as a vectorized scatter-add so we can run thousands of
products quickly.

A versor here is a UNIT multivector. Versors form a group under the geometric
product: the inverse is reverse(X) / (X * reverse(X)). Storing an association
is multiplication by a versor; recall is multiplication by its inverse.
"""
import numpy as np
from functools import lru_cache


def popcount(x):
    return bin(x).count("1")


@lru_cache(maxsize=None)
def algebra(n):
    """Return (DIM, SIGN, IDX, GRADE, REV_SIGN) for Cl(n,0)."""
    DIM = 1 << n

    def blade_mul(a, b):
        sign = 1
        result = a
        for i in range(n):
            if not (b & (1 << i)):
                continue
            higher = 0
            for j in range(i + 1, n):
                if result & (1 << j):
                    higher += 1
            if higher & 1:
                sign = -sign
            if result & (1 << i):
                result &= ~(1 << i)      # e_i^2 = +1
            else:
                result |= (1 << i)
        return sign, result

    SIGN = np.zeros((DIM, DIM), dtype=np.float64)
    IDX = np.zeros((DIM, DIM), dtype=np.int64)
    for i in range(DIM):
        for j in range(DIM):
            s, r = blade_mul(i, j)
            SIGN[i, j] = s
            IDX[i, j] = r
    GRADE = np.array([popcount(m) for m in range(DIM)])
    REV_SIGN = np.array([-1.0 if ((g * (g - 1) // 2) & 1) else 1.0 for g in GRADE])
    # Precompute flat scatter indices for fast product
    return DIM, SIGN, IDX, GRADE, REV_SIGN


def geo_product(a, b, n):
    """Geometric product of two multivectors (1-D arrays length 2^n)."""
    DIM, SIGN, IDX, _, _ = algebra(n)
    out = np.zeros(DIM)
    # contributions[i,j] = SIGN[i,j]*a[i]*b[j] scattered to IDX[i,j]
    contrib = (a[:, None] * b[None, :]) * SIGN
    np.add.at(out, IDX.ravel(), contrib.ravel())
    return out


def reverse(a, n):
    _, _, _, _, REV = algebra(n)
    return a * REV


def norm(a, n):
    return np.sqrt(abs(geo_product(a, reverse(a, n), n)[0]))


def unit(a, n):
    nv = norm(a, n)
    if nv == 0:
        raise ValueError("zero")
    return a / nv


def inverse(a, n):
    r = reverse(a, n)
    denom = geo_product(a, r, n)[0]
    return r / denom


def random_versor(n, rng):
    """A random unit multivector (full-grade)."""
    return unit(rng.standard_normal(1 << n), n)


def random_unit_vector(n, rng):
    """A random unit grade-1 multivector (only e_i components)."""
    DIM = 1 << n
    v = np.zeros(DIM)
    for i in range(n):
        v[1 << i] = rng.standard_normal()
    return unit(v, n)
