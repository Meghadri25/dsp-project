import numpy as np
from src.dwt import haar_dwt, haar_idwt, dwt3_level, idwt3_level


def test_haar_dwt_basic():
    signal = np.array([1, 2, 3, 4])
    cA, cD = haar_dwt(signal)
    assert len(cA) == 2
    assert len(cD) == 2


def test_haar_dwt_odd_length():
    signal = np.array([1, 2, 3])
    cA, cD = haar_dwt(signal)
    assert len(cA) == 2
    assert len(cD) == 2


def test_haar_idwt_basic():
    cA = np.array([2.12132034, 4.94974747])
    cD = np.array([-0.70710678, -0.70710678])
    rec = haar_idwt(cA, cD)
    expected = np.array([1, 2, 3, 4])
    np.testing.assert_allclose(rec, expected, atol=1e-6)


def test_haar_dwt_idwt_reconstruction():
    signal = np.random.randn(8)
    cA, cD = haar_dwt(signal)
    rec = haar_idwt(cA, cD)
    np.testing.assert_allclose(rec[: len(signal)], signal, atol=1e-10)


def test_haar_idwt_padding():
    cA = np.array([1, 2])
    cD = np.array([0.5])
    rec = haar_idwt(cA, cD)
    assert len(rec) == 4


def test_haar_idwt_empty():
    rec = haar_idwt(np.array([]), np.array([]))
    assert len(rec) == 0


def test_dwt3_level():
    signal = np.random.randn(16)
    A3, D3, D2, D1 = dwt3_level(signal)
    assert len(A3) == 2
    assert len(D3) == 2
    assert len(D2) == 4
    assert len(D1) == 8


def test_idwt3_level_reconstruction():
    signal = np.random.randn(16)
    coeffs = dwt3_level(signal)
    rec = idwt3_level(*coeffs)
    np.testing.assert_allclose(rec[: len(signal)], signal, atol=1e-10)
