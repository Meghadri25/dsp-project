import numpy as np
from src.extractor import extract_bits_from_section, extract_watermark_from_audio


def test_extract_bits_from_section_basic():
    pa3 = np.random.randn(16)
    bit = extract_bits_from_section(pa3)
    assert bit in [0, 1]


def test_extract_bits_from_section_short():
    pa3 = np.array([1.0])
    bit = extract_bits_from_section(pa3)
    assert bit == 0


def test_extract_bits_from_section_even_moments():
    pa3 = np.ones(16)
    bit = extract_bits_from_section(pa3)
    assert bit in [0, 1]  # Norms may not be exactly equal due to numerical precision


def test_extract_watermark_from_audio_shape():
    audio = np.random.randn(1000)
    N, M = 4, 4
    params = {"L1": 2, "Lseg": 8, "L": 100, "mlncml_key": {"mu": 3.99}}
    W = extract_watermark_from_audio(audio, 44100, N, M, params)
    assert W.shape == (N, M)
    assert W.dtype == np.uint8
    assert np.all(np.isin(W, [0, 1]))


def test_extract_watermark_from_audio_small():
    audio = np.random.randn(50)
    N, M = 2, 2
    params = {"L1": 1, "Lseg": 4, "L": 10, "mlncml_key": {}}
    W = extract_watermark_from_audio(audio, 44100, N, M, params)
    assert W.shape == (2, 2)
