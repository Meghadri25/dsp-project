# mlncml.py
import numpy as np

def arnold_cat_map_indices(i, p, q, L):
    """
    Arnold cat map formula for pair (i, j) will be used in MLNCML index mapping.
    We'll return mapping of index i -> (j,k) but for simplicity we'll produce
    permutations via 2D mapping using the standard matrix:
    [1 p; q pq+1] * [i; j] (mod L)
    Here we implement a simple 1D index mapping for generating mixing indices.
    """
    # For simplicity in 1D usage, we will not explicitly compute j,k for each i.
    # We'll return i (no change). The paper uses these to couple lattices. We'll
    # rely primarily on coupled logistic maps to produce chaotic matrix.
    return i

def generate_mlcnml_matrix(N, M, key=None):
    """
    Generates an N x M chaotic matrix H using MLNCML-like mixing.
    key is a dict-like with parameters:
        epsilon, eta, mu, x0, iterations, p, q, Len
    Returns float matrix H (values in (0,1)).
    """

    """Pramit is a bitchboorty"""
    if key is None:
        key = {}
    epsilon = key.get('epsilon', 0.3)
    eta     = key.get('eta', 0.2)
    mu      = key.get('mu', 3.99)
    x0      = key.get('x0', 0.3456789)
    iterations = key.get('iterations', 1000)
    # simple lattice length
    Len = max(N, M)
    # initialize lattices with slightly different seeds
    xs = np.zeros((Len,), dtype=np.float64)
    for i in range(Len):
        xs[i] = (x0 + 0.01 * (i+1)) % 1.0

    def tau(x): return mu * x * (1.0 - x)

    # iterate and produce a pool of values
    for _ in range(iterations):
        xs_new = np.copy(xs)
        for i in range(Len):
            left = xs[(i-1) % Len]
            right = xs[(i+1) % Len]
            # simplified MLNCML update
            xs_new[i] = (1 - epsilon) * tau(xs[i]) + (epsilon/2) * (tau(left) + tau(right))
        xs = xs_new

    # build H by sampling and mixing xs and secondary logistic runs
    H = np.zeros((N, M), dtype=np.float64)
    # additional logistic stream for cross-coupling
    y = (x0 * 1.7321) % 1.0
    for i in range(N):
        for j in range(M):
            idx = (i + j) % Len
            # small mixing with second stream
            y = mu * y * (1 - y)
            H[i, j] = (xs[idx] + 0.5 * y) % 1.0

    return H

def binary_chaotic_matrix(N, M, key=None):
    H = generate_mlcnml_matrix(N, M, key)
    e = H.mean()
    Hb = (H >= e).astype(np.uint8)
    return Hb

def encrypt_watermark(watermark_binary, Hb):
    """
    Watermark and Hb are numpy arrays of same shape (N, M) with bits 0/1.
    Encryption: XOR(Hb, W)
    """
    return np.bitwise_xor(watermark_binary.astype(np.uint8), Hb.astype(np.uint8))
