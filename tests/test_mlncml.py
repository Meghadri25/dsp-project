import numpy as np
from src.mlncml import (
    arnold_map,
    generate_mlcnml_matrix,
    binary_chaotic_matrix,
    encrypt_watermark,
)


def test_arnold_map_basic():
    j, k = arnold_map(0, 1, 1, 10)
    assert 0 <= j < 10
    assert 0 <= k < 10
    assert isinstance(j, int)
    assert isinstance(k, int)


def test_arnold_map_deterministic():
    j1, k1 = arnold_map(5, 2, 3, 20)
    j2, k2 = arnold_map(5, 2, 3, 20)
    assert j1 == j2
    assert k1 == k2


def test_arnold_map_modulo():
    L = 7
    for i in range(L):
        j, k = arnold_map(i, 1, 1, L)
        assert 0 <= j < L
        assert 0 <= k < L


def test_generate_mlcnml_matrix_shape():
    N, M = 10, 8
    H = generate_mlcnml_matrix(N, M)
    assert H.shape == (N, M)


def test_generate_mlcnml_matrix_values():
    H = generate_mlcnml_matrix(5, 5)
    assert np.all(H > 0)
    assert np.all(H < 1)


def test_generate_mlcnml_matrix_deterministic():
    key = {"mu": 3.99, "x0": 0.5, "iterations": 100}
    H1 = generate_mlcnml_matrix(4, 4, key=key)
    H2 = generate_mlcnml_matrix(4, 4, key=key)
    np.testing.assert_array_equal(H1, H2)


def test_binary_chaotic_matrix_shape():
    N, M = 6, 7
    Hb = binary_chaotic_matrix(N, M)
    assert Hb.shape == (N, M)
    assert Hb.dtype == np.uint8


def test_binary_chaotic_matrix_binary():
    Hb = binary_chaotic_matrix(5, 5)
    unique = np.unique(Hb)
    assert set(unique) == {0, 1}


def test_binary_chaotic_matrix_threshold():
    H = generate_mlcnml_matrix(4, 4)
    e = H.mean()
    Hb_manual = (H >= e).astype(np.uint8)
    Hb = binary_chaotic_matrix(4, 4)
    np.testing.assert_array_equal(Hb, Hb_manual)


def test_encrypt_watermark_basic():
    W = np.random.randint(0, 2, (5, 5), dtype=np.uint8)
    Hb = binary_chaotic_matrix(5, 5)
    encrypted = encrypt_watermark(W, Hb)
    assert encrypted.shape == W.shape
    assert encrypted.dtype == np.uint8
    assert np.array_equal(np.unique(encrypted), [0, 1])


def test_encrypt_watermark_xor_property():
    W = np.random.randint(0, 2, (4, 4), dtype=np.uint8)
    Hb = binary_chaotic_matrix(4, 4)
    encrypted = encrypt_watermark(W, Hb)
    # XOR twice should recover original
    recovered = encrypt_watermark(encrypted, Hb)
    np.testing.assert_array_equal(recovered, W)
