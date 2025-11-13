# dwt.py
import numpy as np

def _ensure_array(x):
    return np.asarray(x, dtype=float)

def haar_dwt(signal):
    """
    Single-level Haar DWT (returns approximation and detail coefficients).
    Pads the input if its length is odd by repeating the last sample.
    """
    signal = _ensure_array(signal)
    if len(signal) % 2 != 0:
        # pad by repeating last sample so pairs are complete
        signal = np.append(signal, signal[-1])
    # Haar filters: (1/sqrt(2))[1,1] and (1/sqrt(2))[1,-1]
    cA = (signal[0::2] + signal[1::2]) / np.sqrt(2)
    cD = (signal[0::2] - signal[1::2]) / np.sqrt(2)
    return cA, cD

def haar_idwt(cA, cD):
    """
    Single-level inverse Haar DWT.
    If cA and cD have different lengths (can happen after multilevel dwt),
    pad the shorter one by repeating its last element so they match.
    """
    cA = _ensure_array(cA)
    cD = _ensure_array(cD)
    # make lengths equal by padding the shorter sequence with its last value
    la = len(cA)
    ld = len(cD)
    if la == 0 and ld == 0:
        return np.array([], dtype=float)
    if la < ld:
        if la == 0:
            cA = np.repeat(0.0, ld)
        else:
            cA = np.pad(cA, (0, ld - la), mode='edge')
    elif ld < la:
        if ld == 0:
            cD = np.repeat(0.0, la)
        else:
            cD = np.pad(cD, (0, la - ld), mode='edge')

    n = max(len(cA), len(cD))
    rec = np.empty(2 * n, dtype=float)
    rec[0::2] = (cA + cD) / np.sqrt(2)
    rec[1::2] = (cA - cD) / np.sqrt(2)
    return rec

def dwt3_level(signal):
    """
    Performs 3-level Haar DWT decomposition manually.
    Returns (A3, D3, D2, D1)
    """
    A1, D1 = haar_dwt(signal)
    A2, D2 = haar_dwt(A1)
    A3, D3 = haar_dwt(A2)
    return A3, D3, D2, D1

def idwt3_level(A3, D3, D2, D1):
    """
    Reconstructs signal from 3-level Haar coefficients.
    Uses robust inverse that tolerates different coefficient lengths.
    """
    A2 = haar_idwt(A3, D3)
    A1 = haar_idwt(A2, D2)
    rec = haar_idwt(A1, D1)
    return rec
