# dwt.py
import numpy as np


def haar_dwt(signal):
    signal = np.asarray(signal, dtype=float)
    if len(signal) % 2 != 0:
        signal = np.append(signal, signal[-1])
    cA = (signal[0::2] + signal[1::2]) / np.sqrt(2)
    cD = (signal[0::2] - signal[1::2]) / np.sqrt(2)
    return cA, cD


def haar_idwt(cA, cD):
    cA = np.asarray(cA, dtype=float)
    cD = np.asarray(cD, dtype=float)
    la = len(cA)
    ld = len(cD)
    if la == 0 and ld == 0:
        return np.array([], dtype=float)
    if la < ld:
        if la == 0:
            cA = np.repeat(0.0, ld)
        else:
            cA = np.pad(cA, (0, ld - la), mode="edge")
    elif ld < la:
        if ld == 0:
            cD = np.repeat(0.0, la)
        else:
            cD = np.pad(cD, (0, la - ld), mode="edge")

    n = max(len(cA), len(cD))
    rec = np.empty(2 * n, dtype=float)
    rec[0::2] = (cA + cD) / np.sqrt(2)
    rec[1::2] = (cA - cD) / np.sqrt(2)
    return rec


def dwt3_level(signal):
    A1, D1 = haar_dwt(signal)
    A2, D2 = haar_dwt(A1)
    A3, D3 = haar_dwt(A2)
    return A3, D3, D2, D1


def idwt3_level(A3, D3, D2, D1):
    A2 = haar_idwt(A3, D3)
    A1 = haar_idwt(A2, D2)
    rec = haar_idwt(A1, D1)
    return rec
