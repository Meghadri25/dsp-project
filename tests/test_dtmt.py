import numpy as np
from src.dtmt import tchebichef_polynomials, tchebichef_norms, dtmt, idtmt


def test_tchebichef_polynomials_shape():
    K, N = 5, 10
    T = tchebichef_polynomials(N, K)
    assert T.shape == (K, N)


def test_tchebichef_polynomials_orthonormality():
    K, N = 4, 8
    T = tchebichef_polynomials(N, K)
    rho = tchebichef_norms(K, N)
    Tn = T / np.sqrt(rho[:, None])
    # Check orthonormality: Tn @ Tn.T ≈ I
    gram = Tn @ Tn.T
    identity = np.eye(K)
    np.testing.assert_allclose(gram, identity, atol=1e-10)


def test_dtmt_basic():
    signal = np.random.randn(10)
    K = 5
    M, Tn = dtmt(signal, K=K)
    assert M.shape == (K,)
    assert Tn.shape == (K, 10)


def test_dtmt_idtmt_reconstruction():
    # Test perfect reconstruction for low-order polynomials
    signal = np.ones(8)  # Constant signal, should be reconstructed perfectly
    K = 1  # Only DC component
    M, Tn = dtmt(signal, K=K)
    reconstructed = idtmt(M, Tn)
    np.testing.assert_allclose(reconstructed, signal, atol=1e-12)


def test_dtmt_idtmt_approximate_reconstruction():
    signal = np.sin(np.linspace(0, 2 * np.pi, 16))
    K = 8
    M, Tn = dtmt(signal, K=K)
    reconstructed = idtmt(M, Tn)
    # Should be approximate, not exact
    mse = np.mean((signal - reconstructed) ** 2)
    assert mse < 0.1  # Reasonable approximation


def test_dtmt_edge_cases():
    # Empty signal
    M, Tn = dtmt(np.array([]))
    assert M.size == 0
    assert Tn.size == 0

    # K > N
    signal = np.ones(5)
    M, Tn = dtmt(signal, K=10)
    assert M.shape == (5,)  # K capped to N
    assert Tn.shape == (5, 5)


def test_idtmt_edge_cases():
    # Empty
    reconstructed = idtmt(np.array([]), np.zeros((0, 0)))
    assert reconstructed.size == 0
