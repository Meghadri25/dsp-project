# dtmt.py
import numpy as np

def tchebichef_polynomials(K, N):
    """
    Compute Tchebichef polynomials up to order K-1 evaluated at x=0..N-1.
    Returns T_raw of shape (K, N) where row n is t_n(x; N).
    Uses the three-term recurrence; numerically normalized later.
    """
    x = np.arange(N, dtype=np.float64)
    T_raw = np.zeros((K, N), dtype=np.float64)
    if K <= 0 or N <= 0:
        return T_raw
    T_raw[0, :] = 1.0
    if K == 1:
        return T_raw

    # Second order
    T_raw[1, :] = 2 * x - N + 1

    for n in range(1, K-1):
        a = (2 * n + 1) * (2 * x - N + 1)
        b = n * (N**2 - n**2)
        # (n+1) * t_{n+1} = a * t_n - b * t_{n-1}
        T_raw[n+1, :] = (a * T_raw[n, :] - b * T_raw[n-1, :]) / (n + 1)

    return T_raw

def tchebichef_matrix(K, N):
    """
    Return normalized T matrix of shape (K, N) with orthonormal rows (approx).
    K = number of moment orders used (number of rows).
    N = length of the signal (number of columns).
    """
    T_raw = tchebichef_polynomials(K, N)
    norms = np.linalg.norm(T_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    T = T_raw / norms
    return T

def dtmt(signal_block, K=None, maxK=64):
    """
    Compute DTMT (truncated) for a 1D signal_block of length N.

    Parameters:
      - signal_block : 1D numpy array of length N
      - K : number of Tchebichef orders to compute. If None, choose min(maxK, N).
      - maxK : default upper bound for K (useful safety control)

    Returns:
      - M : moment vector of length K
      - T : Tchebichef matrix of shape (K, N)
    """
    x = np.asarray(signal_block, dtype=np.float64)
    N = len(x)
    if N == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)

    if K is None:
        K = min(maxK, N)
    else:
        K = int(min(K, N, maxK))

    if K <= 0:
        K = min(maxK, N)

    T = tchebichef_matrix(K, N)  # shape (K, N)
    # moments: M_n = sum_x T[n,x] * signal[x]
    M = T.dot(x)  # shape (K,)
    return M, T

def idtmt(M, T):
    """
    Inverse (approximate) DTMT: reconstruct signal from moments M and matrix T.
    Since K <= N generally, we solve the linear least-squares problem:
        minimize || T @ signal - M ||_2
    Returns reconstructed 1D signal of length N.
    """
    M = np.asarray(M, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if T.size == 0:
        return np.zeros((0,), dtype=np.float64)
    # Solve T @ signal = M for 'signal' with least-squares -> returns length N
    sol, *_ = np.linalg.lstsq(T, M, rcond=None)
    return sol
