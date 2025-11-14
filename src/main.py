# main.py
import numpy as np
import imageio
import soundfile as sf
from dtmt import dtmt, idtmt
from dwt import dwt3_level, idwt3_level
from mlncml import binary_chaotic_matrix, encrypt_watermark
from extractor import extract_watermark_from_audio

import struct
import zlib
from typing import Tuple

# ---------------- PNG low-level writer (no PIL) ----------------
def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk = chunk_type + data
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk) & 0xffffffff)
    return length + chunk + crc

def save_png_native(path: str, arr: np.ndarray) -> None:
    a = np.asarray(arr)
    # Normalize to uint8
    if np.issubdtype(a.dtype, np.floating):
        if a.size and a.max() <= 1.0:
            a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            a = np.clip(a, 0.0, 255.0).round().astype(np.uint8)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)

    # Collapse last channel if (H,W,1)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]

    if a.ndim == 2:
        H, W = a.shape
        color_type = 0   # grayscale
        bit_depth = 8
        bytes_per_pixel = 1
    elif a.ndim == 3 and a.shape[2] == 3:
        H, W, C = a.shape
        if C != 3:
            raise ValueError("Only 3-channel RGB supported for 3D arrays.")
        color_type = 2   # truecolor RGB
        bit_depth = 8
        bytes_per_pixel = 3
    else:
        raise ValueError(f"Unsupported array shape for PNG: {a.shape}")

    a = np.ascontiguousarray(a)
    raw_rows = []
    for y in range(H):
        row = a[y].tobytes()
        raw_rows.append(b'\x00' + row)

    raw = b"".join(raw_rows)
    comp = zlib.compress(raw)

    png_sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack(">IIBBBBB",
                       W, H, bit_depth, color_type, 0, 0, 0)
    ihdr_chunk = _png_chunk(b'IHDR', ihdr)
    idat_chunk = _png_chunk(b'IDAT', comp)
    iend_chunk = _png_chunk(b'IEND', b'')

    with open(path, "wb") as f:
        f.write(png_sig)
        f.write(ihdr_chunk)
        f.write(idat_chunk)
        f.write(iend_chunk)

# --------------- BMP fallback writer (no PIL) -----------------
def save_bmp_native(path: str, arr: np.ndarray) -> None:
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        if a.size and a.max() <= 1.0:
            a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            a = np.clip(a, 0.0, 255.0).round().astype(np.uint8)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    if a.ndim == 2:
        H, W = a.shape
        rgb = np.stack([a, a, a], axis=2)
    elif a.ndim == 3 and a.shape[2] == 3:
        H, W, _ = a.shape
        rgb = a
    else:
        rgb = a[..., :3]
        H, W, _ = rgb.shape

    row_bytes = W * 3
    padding = (4 - (row_bytes % 4)) % 4
    bmp_data = bytearray()
    for y in range(H-1, -1, -1):
        row = rgb[y].tobytes()
        row_bgr = bytearray()
        for i in range(0, len(row), 3):
            r = row[i]; g = row[i+1]; b = row[i+2]
            row_bgr.extend([b, g, r])
        bmp_data.extend(row_bgr)
        if padding:
            bmp_data.extend(b'\x00' * padding)

    file_size = 14 + 40 + len(bmp_data)
    bfType = b'BM'
    bfSize = struct.pack('<I', file_size)
    bfReserved = struct.pack('<HH', 0, 0)
    bfOffBits = struct.pack('<I', 14 + 40)
    biSize = struct.pack('<I', 40)
    biWidth = struct.pack('<i', W)
    biHeight = struct.pack('<i', H)
    biPlanes = struct.pack('<H', 1)
    biBitCount = struct.pack('<H', 24)
    biCompression = struct.pack('<I', 0)
    biSizeImage = struct.pack('<I', len(bmp_data))
    biXPelsPerMeter = struct.pack('<i', 0)
    biYPelsPerMeter = struct.pack('<i', 0)
    biClrUsed = struct.pack('<I', 0)
    biClrImportant = struct.pack('<I', 0)

    with open(path, 'wb') as f:
        f.write(bfType)
        f.write(bfSize)
        f.write(bfReserved)
        f.write(bfOffBits)
        f.write(biSize)
        f.write(biWidth)
        f.write(biHeight)
        f.write(biPlanes)
        f.write(biBitCount)
        f.write(biCompression)
        f.write(biSizeImage)
        f.write(biXPelsPerMeter)
        f.write(biYPelsPerMeter)
        f.write(biClrUsed)
        f.write(biClrImportant)
        f.write(bmp_data)

# ---------------- Diagnostics & finalize save -----------------
def debug_array_info(name: str, a: np.ndarray):
    a = np.asarray(a)
    print(f"DEBUG {name}: shape={a.shape}, dtype={a.dtype}, "
          f"min={np.min(a) if a.size else 'NA'}, max={np.max(a) if a.size else 'NA'}, "
          f"unique_count={len(np.unique(a)) if a.size else 0}")

def save_variants_and_debug(a, basename="extracted_watermark", N_expected=None, M_expected=None):
    """
    a: numpy array (extracted watermark)
    Saves: .npy, .png, .bmp, and transposed variants. Prints diagnostics.
    """
    arr = np.asarray(a)
    debug_array_info("raw_extracted", arr)

    # reshape if flattened and expected dims given
    if arr.ndim == 1:
        if N_expected and M_expected and arr.size == (N_expected * M_expected):
            arr = arr.reshape((N_expected, M_expected))
            print("Reshaped 1D array to expected (N,M).")
        else:
            s = int(np.round(np.sqrt(arr.size)))
            if s*s == arr.size:
                arr = arr.reshape((s, s))
                print("Reshaped 1D array into square (s,s).")
            else:
                arr = arr.reshape((1, arr.size))
                print("Reshaped 1D array into (1, length).")

    # scale booleans or 0/1 -> 0..255
    if np.issubdtype(arr.dtype, np.bool_) or (arr.size and np.max(arr) <= 1 and np.min(arr) >= 0):
        a_out = (arr.astype(np.uint8) * 255)
    else:
        a_out = np.clip(arr, 0, 255).astype(np.uint8)

    debug_array_info("prepared_for_save", a_out)

    np.save(basename + ".npy", a_out)
    print("Saved raw .npy for inspection:", basename + ".npy")

    # Save PNG (native)
    try:
        save_png_native(basename + ".png", a_out)
        print("Saved PNG:", basename + ".png")
    except Exception as e:
        print("PNG save failed:", e)

    # Save BMP fallback
    try:
        save_bmp_native(basename + ".bmp", a_out)
        print("Saved BMP:", basename + ".bmp")
    except Exception as e:
        print("BMP save failed:", e)

# ---------------- Embedding function (unchanged except K heuristic) ----------------
def embed_watermark(audio_path, watermark_path, out_path,
                    L1=4, delta=0.05, mlncml_key=None):
    audio, Fs = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    W_img = imageio.imread(watermark_path)
    if W_img.ndim == 3:
        W_img = np.mean(W_img, axis=2)
    W_bin = (W_img > W_img.mean()).astype(np.uint8)
    N, M = W_bin.shape
    Lw = N * M

    Hb = binary_chaotic_matrix(N, M, key=mlncml_key)
    W1 = encrypt_watermark(W_bin, Hb)
    W2 = W1.flatten()

    Lseg = int(np.ceil(Lw / L1))
    L = int(np.floor(len(audio) / Lseg))
    if L <= 0:
        raise ValueError("Audio too short for requested segmentation.")

    params = {
        'L1': L1,
        'Lseg': Lseg,
        'L': L,
        'delta': delta,
        'mlncml_key': mlncml_key
    }

    out_audio = np.copy(audio)
    idx_bit = 0
    for n in range(Lseg):
        seg_start = n * L
        seg = audio[seg_start:seg_start + L]
        if len(seg) < L:
            seg = np.pad(seg, (0, L - len(seg)))
        A3, D3, D2, D1 = dwt3_level(seg)
        A3 = np.array(A3, dtype=np.float64)

        L2 = int(np.floor(len(A3) / L1))
        if L2 <= 0:
            L2 = 1
        A3_mod = A3.copy()
        for m in range(L1):
            start = m * L2
            pa3 = A3[start:start + L2].astype(np.float64)
            if len(pa3) < L2:
                pa3 = np.pad(pa3, (0, L2 - len(pa3)))
            K = min(64, max(8, len(pa3) // 2))   # heuristic: <=64 orders
            Mvec, T = dtmt(pa3, K=K)

            M1 = Mvec[0::2].copy()
            M2 = Mvec[1::2].copy()
            sigma1 = np.linalg.norm(M1)
            sigma2 = np.linalg.norm(M2)
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
            A3_mod[start:start + L2] = pa3_w[:L2]

        seg_w = idwt3_level(A3_mod, D3, D2, D1)
        out_audio[seg_start:seg_start + min(len(seg_w), L)] = seg_w[:min(len(seg_w), L)]

    sf.write(out_path, out_audio, Fs)
    print(f"✅ Watermarked audio written to: {out_path}")
    return params, (N, M)

# ==========================================================
# ===============  MAIN: EMBED OR EXTRACT  =================
# ==========================================================
if __name__ == "__main__":
    mode = "embed"   # set "embed" or "extract"

    key = {'epsilon': 0.3, 'eta': 0.2, 'mu': 0.1, 'x0': 0.3456789, 'iterations': 500}

    if mode == "embed":
        audio_in = "The_Color_Violet.wav"
        watermark_img = "watermark.png"
        audio_out = "watermarked.wav"

        params, dims = embed_watermark(audio_in, watermark_img, audio_out,
                                       L1=4, delta=0.0001, mlncml_key=key)

        print("✅ Embedding complete.")
        print("params:", params)
        print("dims:", dims)

        np.savez("embed_params.npz", **params, N=dims[0], M=dims[1])
        print("🔒 Saved embedding parameters to embed_params.npz")

    elif mode == "extract":
        data = np.load("embed_params.npz", allow_pickle=True)

        def safe_scalar(val):
            try:
                arr = np.array(val)
                if arr.size == 0:
                    return 0
                return int(arr.flatten()[0])
            except Exception:
                return int(val) if np.isscalar(val) else 0

        # Safe load for N, M
        N = safe_scalar(data.get("N", 0))
        M = safe_scalar(data.get("M", 0))

        print(f"Loaded watermark dims from file: N={N}, M={M}")

        # If M accidentally loaded as 0, check for fallback
        if M == 0 or N == 0:
            print("⚠️ Warning: Invalid watermark dimensions loaded.")
            print("Check that embedding actually completed and saved dims correctly.")
            # Try to load the watermark image to infer correct dims
            try:
                W_img = imageio.imread("watermark.png")
                if W_img.ndim == 3:
                    W_img = np.mean(W_img, axis=2)
                N, M = W_img.shape
                print(f"✅ Recovered dims from watermark.png: N={N}, M={M}")
            except Exception as e:
                print("❌ Could not recover dims from watermark:", e)
        params = {k: data[k].item() if data[k].shape == () else data[k]
                  for k in data.files if k not in ("N", "M")}

        watermarked_audio = "watermarked.wav"

        audio_signal, Fs = sf.read(watermarked_audio)
        if audio_signal.ndim > 1:
            audio_signal = audio_signal.mean(axis=1)

        params["Lseg"] = int(np.ceil((N * M) / max(1, params["L1"])))
        if params["Lseg"] <= 0:
            params["Lseg"] = 1
        if params["Lseg"] > 0:
            params["L"] = int(np.floor(len(audio_signal) / params["Lseg"]))
        else:
            params["L"] = len(audio_signal)

        print("🔍 Extracting watermark...")
        W_extracted = extract_watermark_from_audio(audio_signal, Fs, N, M, params)

        # Save diagnostics + multiple variants (PNG, BMP, .npy, transposed)
        save_variants_and_debug(W_extracted, basename="extracted_watermark", N_expected=N, M_expected=M)
        print("✅ Extraction + diagnostics complete.")
