# extractor.py
import numpy as np
from dtmt import dtmt
from mlncml import binary_chaotic_matrix
from dwt import dwt3_level

def extract_bits_from_section(pa3_section):
    """
    For a single P_A3 section (1D array), compute DTMT, subsample moments,
    compute norms sigma1, sigma2 and return bit (1 if sigma1>sigma2 else 0).
    """
    # Match K heuristic with embedding
    K = min(64, max(8, len(pa3_section) // 2))
    M, T = dtmt(pa3_section, K=K)

    if len(M) < 2:  # too short to split into M1, M2
        return 0

    M1 = M[0::2]
    M2 = M[1::2]
    sigma1 = np.linalg.norm(M1)
    sigma2 = np.linalg.norm(M2)
    return 1 if sigma1 > sigma2 else 0


def extract_watermark_from_audio(audio_signal, Fs, N, M, params):
    """
    audio_signal: 1D numpy array
    N, M: watermark dimensions (rows, cols)
    params: dict containing segmentation params used during embedding:
            L1, Lseg, L, delta, mlncml_key
    Returns extracted binary watermark of shape (N, M)
    """
    L1 = int(params.get('L1', 4))
    Lseg = int(params.get('Lseg', max(1, int(np.ceil((N * M) / L1)))))
    L = int(params.get('L', max(1, int(np.floor(len(audio_signal) / Lseg)))))

    bit_seq = []
    for n in range(Lseg):
        seg = audio_signal[n * L : n * L + L]
        if len(seg) < L:
            seg = np.pad(seg, (0, L - len(seg)))

        A3, D3, D2, D1 = dwt3_level(seg)
        L2 = int(np.floor(len(A3) / L1))
        if L2 <= 0:
            L2 = 1

        for m in range(L1):
            start = m * L2
            pa3 = A3[start:start + L2]
            if len(pa3) < L2:
                pa3 = np.pad(pa3, (0, L2 - len(pa3)))

            bit = extract_bits_from_section(pa3)
            bit_seq.append(bit)

    # convert bit sequence to 2D watermark
    total_bits = N * M
    W2 = np.array(bit_seq[:total_bits], dtype=np.uint8)
    if W2.size < total_bits:
        W2 = np.pad(W2, (0, total_bits - W2.size), constant_values=0)
    W1 = W2.reshape((N, M))

    # decrypt using MLNCML chaotic key
    Hb = binary_chaotic_matrix(N, M, key=params.get('mlncml_key', None))
    W_extracted = np.bitwise_xor(W1, Hb).astype(np.uint8)

    print("\nDEBUG extractor output:")
    print(f"  Extracted bit count: {len(bit_seq)}")
    print(f"  Watermark dims: ({N}, {M})")
    print(f"  Final extracted shape: {W_extracted.shape}")
    print(f"  Unique values in watermark: {np.unique(W_extracted)}\n")

    return W_extracted
