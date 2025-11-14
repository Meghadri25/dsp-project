# dtmt.py
import numpy as np
import math


def tchebichef_polynomials(N, K):
    """
    Computes T[n,x] = t_n(x;N) for n=0..K-1 and x=0..N-1
    using the exact three-term recurrence from the paper.
    """
    x = np.arange(N, dtype=np.float64)
    T = np.zeros((K, N), dtype=np.float64)

    # initial conditions
    T[0, :] = 1.0
    if K > 1:
        T[1, :] = 2 * x - N + 1

    # recurrence (paper eq. 5)
    for n in range(1, K - 1):
        a = (2 * n + 1) * (2 * x - N + 1) / (n + 1)
        b = n * (N**2 - n**2) / (n + 1)
        T[n + 1, :] = a * T[n, :] - b * T[n - 1, :]

    return T


def tchebichef_norm(n, N):
    return math.factorial(2 * n) * math.comb(N + n, 2 * n + 1)


def tchebichef_norms(K, N):
    return np.array([tchebichef_norm(n, N) for n in range(K)], dtype=np.float64)


def dtmt(signal, K=None):
    """
    Compute Discrete Tchebichef Moment Transform for a 1D signal.

    Parameters:
    - signal: 1D numpy array of length N
    - K: number of orders. If None, uses N

    Returns:
    - M: moment vector of length K
    - Tn: normalized Tchebichef matrix of shape (K, N)
    """
    x = np.asarray(signal, dtype=np.float64)
    N = len(x)
    if N == 0:
        return np.array([]), np.zeros((0, 0))

    K = N if K is None else min(K, N)
    K = max(1, K)  # Ensure at least 1

    T = tchebichef_polynomials(N, K)
    rho = tchebichef_norms(K, N)
    Tn = T / np.sqrt(rho[:, None])  # normalized basis

    M = Tn @ x  # moments
    return M, Tn


def idtmt(M, Tn):
    """
    Inverse DTMT: reconstruct signal from moments M and normalized matrix Tn.
    """
    M = np.asarray(M, dtype=np.float64)
    Tn = np.asarray(Tn, dtype=np.float64)
    if Tn.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return Tn.T @ M
