# embed.py
import numpy as np
from dtmt import dtmt, idtmt
from dwt import dwt3_level, idwt3_level
from mlncml import binary_chaotic_matrix, encrypt_watermark


def prepare_watermark_for_embedding(
    W_bin: np.ndarray, N: int, M: int, mlncml_key: dict
) -> np.ndarray:
    """Encrypt and flatten the watermark."""
    Hb = binary_chaotic_matrix(N, M, key=mlncml_key)
    W1 = encrypt_watermark(W_bin, Hb)
    W2 = W1.flatten()
    return W2


def segment_and_embed(
    audio: np.ndarray,
    W2: np.ndarray,
    L1: int,
    delta: float,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Perform segmentation and embedding on the audio."""
    Lw = len(W2)
    Lseg = int(np.ceil(Lw / L1))
    L = int(np.floor(len(audio) / Lseg))
    if L <= 0:
        raise ValueError("Audio too short for requested segmentation.")

    total_slots = Lseg * L1
    if len(W2) < total_slots:
        W2 = np.pad(W2, (0, total_slots - len(W2)), mode="constant")

    out_audio = np.copy(audio)
    idx_bit = 0
    for n in range(Lseg):
        seg_start = n * L
        seg = audio[seg_start : seg_start + L]
        if len(seg) < L:
            seg = np.pad(seg, (0, L - len(seg)), mode=pad_mode)

        A3, D3, D2, D1 = dwt3_level(seg)
        A3 = np.array(A3, dtype=np.float64)

        L2 = int(np.floor(len(A3) / L1))
        if L2 <= 0:
            L2 = 1

        A3_mod = A3.copy()
        for m in range(L1):
            start = m * L2
            pa3 = A3[start : start + L2].astype(np.float64)
            if len(pa3) < L2:
                pa3 = np.pad(pa3, (0, L2 - len(pa3)), mode=pad_mode)
            K = min(64, max(8, len(pa3) // 2))
            Mvec, T = dtmt(pa3, K=K)

            M1 = Mvec[0::2].copy()
            M2 = Mvec[1::2].copy()
            eps = 1e-12
            sigma1 = max(np.linalg.norm(M1), eps)
            sigma2 = max(np.linalg.norm(M2), eps)
            bit = int(W2[idx_bit]) if idx_bit < len(W2) else 0
            idx_bit += 1

            avg = 0.5 * (sigma1 + sigma2)
            if bit == 1:
                new_sigma1, new_sigma2 = avg + delta, avg - delta
            else:
                new_sigma1, new_sigma2 = avg - delta, avg + delta

            scale1 = (new_sigma1 / sigma1) if sigma1 != 0 else 1.0
            scale2 = (new_sigma2 / sigma2) if sigma2 != 0 else 1.0

            M1_w = M1 * scale1
            M2_w = M2 * scale2
            M_w = np.zeros_like(Mvec)
            M_w[0::2] = M1_w
            M_w[1::2] = M2_w
            pa3_w = idtmt(M_w, T)
            A3_mod[start : start + L2] = pa3_w[:L2]

        seg_w = idwt3_level(A3_mod, D3, D2, D1)
        seg_len = min(len(seg_w), L)
        out_audio[seg_start : seg_start + seg_len] = seg_w[:seg_len]

    return out_audio


def embed_watermark(
    audio_signal: np.ndarray,
    Fs: int,
    W_bin: np.ndarray,
    N: int,
    M: int,
    L1: int = 4,
    delta: float = 0.05,
    mlncml_key: dict = None,
) -> tuple[dict, tuple[int, int], np.ndarray, int]:
    """High-level function to embed watermark into audio."""
    W2 = prepare_watermark_for_embedding(W_bin, N, M, mlncml_key)

    Lw = N * M
    Lseg = int(np.ceil(Lw / L1))
    L = int(np.floor(len(audio_signal) / Lseg))

    params = {"L1": L1, "Lseg": Lseg, "L": L, "delta": delta, "mlncml_key": mlncml_key}

    out_audio = segment_and_embed(audio_signal, W2, L1, delta)

    return params, (N, M), out_audio, Fs
