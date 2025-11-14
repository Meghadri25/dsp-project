# mlncml.py
import numpy as np


def arnold_map(i, p, q, L):
    """
    Implements the Arnold-cat map indices per paper Eq.(9):
        [ j ]   [1  p ] [ i ]   (mod L)
        [ k ] = [q pq+1] [ i ]
    For index i (0..L-1) returns j,k in 0..L-1 (integers).
    (The paper presents these relations for selecting lattice links.)
    """
    i_int = int(i) % L
    j = (1 * i_int + p * i_int) % L
    k = (q * i_int + (p * q + 1) * i_int) % L
    return int(j), int(k)


def generate_mlcnml_matrix(N, M, key=None):
    """
    Generate N x M chaotic matrix H using MLNCML per the paper.
    key : dict-like with keys
        epsilon, eta, mu, x0, iterations, p, q, Len
    Returns H with values in (0,1).
    """
    if key is None:
        key = {}

    epsilon = float(key.get("epsilon", 0.3))
    eta = float(key.get("eta", 0.2))
    mu = float(key.get("mu", 3.99))
    x0 = float(key.get("x0", 0.3456789))
    iterations = int(key.get("iterations", 1000))
    p = int(key.get("p", 1))
    q = int(key.get("q", 1))

    Len = int(key.get("Len", max(N, M)))

    indices = np.arange(Len)
    xs = (x0 + 0.01 * (indices + 1)) % 1.0

    j_indices = np.zeros(Len, dtype=int)
    k_indices = np.zeros(Len, dtype=int)
    for i in range(Len):
        j_indices[i], k_indices[i] = arnold_map(i, p, q, Len)

    def tau(x):
        return mu * x * (1.0 - x)

    for _ in range(iterations):
        tau_xs = tau(xs)
        tau_left = np.roll(tau_xs, 1)
        tau_right = np.roll(tau_xs, -1)
        tau_j = tau_xs[j_indices]
        tau_k = tau_xs[k_indices]

        term_self = (1.0 - epsilon) * tau_xs
        term_local = ((1.0 - eta) * epsilon / 2.0) * (tau_right + tau_left)
        term_arnold = (eta * epsilon / 2.0) * (tau_j + tau_k)
        xs_new = term_self + term_local + term_arnold

        xs_new = np.clip(xs_new, 1e-15, 1.0 - 1e-15)
        xs = xs_new

    total_elements = N * M
    y_values = np.zeros(total_elements)
    y = (x0 * 1.7321) % 1.0
    for k in range(total_elements):
        y_values[k] = y
        y = mu * y * (1.0 - y)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(M), indexing="ij")
    idx = (i_grid + j_grid) % Len
    H = (xs[idx] + 0.5 * y_values.reshape(N, M)) % 1.0

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
