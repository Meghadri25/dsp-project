#!/usr/bin/env python3
"""
test.py

Runs DSP attacks on a watermarked audio file, computes SNR (vs original),
extracts the watermark and computes BER and Normalized Correlation (NC).
Saves one non-transposed PNG of the extracted watermark and the attacked audio
for each attack.

Usage:
    python test.py --original original.wav --watermarked watermarked.wav \
        --watermark watermark.png --params embed_params.npz --outdir results

Attacks available: lpf, hpf, crop, noise
Default: runs all attacks
"""
import argparse
import os
import numpy as np
import imageio
import soundfile as sf

from extractor import extract_watermark_from_audio

EPS = 1e-12

# ---------------- DSP attack implementations (pure numpy) -----------------
def fft_lowpass(signal, Fs, cutoff_hz):
    n = len(signal)
    S = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / Fs)
    mask = freqs <= cutoff_hz
    S_filtered = S * mask
    out = np.fft.irfft(S_filtered, n=n)
    return out

def fft_highpass(signal, Fs, cutoff_hz):
    n = len(signal)
    S = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / Fs)
    mask = freqs >= cutoff_hz
    S_filtered = S * mask
    out = np.fft.irfft(S_filtered, n=n)
    return out

def cropping_random_zero(signal, fraction=0.2, seed=None):
    rng = np.random.default_rng(seed)
    n = len(signal)
    k = int(np.round(fraction * n))
    if k <= 0:
        return signal.copy()
    idx = rng.choice(n, size=k, replace=False)
    out = signal.copy()
    out[idx] = 0.0
    return out

def add_zero_mean_gaussian_noise(signal, target_snr_db=20.0, seed=None):
    """
    Add zero-mean Gaussian noise to achieve approximately the given SNR (dB)
    SNR = 10*log10(P_signal / P_noise)
    """
    rng = np.random.default_rng(seed)
    sig = np.asarray(signal, dtype=float)
    # signal power (use mean power)
    p_signal = np.mean(sig ** 2)
    if p_signal <= 0:
        # fallback small noise
        sigma = 1e-6
    else:
        target_linear = 10.0 ** (target_snr_db / 10.0)
        p_noise = p_signal / max(target_linear, 1e-12)
        sigma = np.sqrt(p_noise)
    noise = rng.normal(loc=0.0, scale=sigma, size=sig.shape)
    return sig + noise

# ---------------- metrics -----------------
def compute_snr(clean, test):
    # ensure same length
    n = min(len(clean), len(test))
    clean = np.asarray(clean[:n], dtype=float)
    test = np.asarray(test[:n], dtype=float)
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((clean - test) ** 2)
    if noise_power <= 0:
        return float('inf')
    snr_db = 10.0 * np.log10((signal_power + EPS) / (noise_power + EPS))
    return snr_db

def prepare_watermark_image(path, N_expected=None, M_expected=None):
    W_img = imageio.imread(path)
    # convert to grayscale if needed
    if W_img.ndim == 3:
        W_img = np.mean(W_img, axis=2)
    # binarize by thresholding at mean
    thr = W_img.mean() if W_img.size else 0.5
    W_bin = (W_img > thr).astype(np.uint8)
    # if expected dims provided, try to reshape or crop/pad
    if N_expected and M_expected:
        H, W = W_bin.shape
        if (H, W) != (N_expected, M_expected):
            out = np.zeros((N_expected, M_expected), dtype=np.uint8)
            h = min(H, N_expected)
            w = min(W, M_expected)
            out[:h, :w] = W_bin[:h, :w]
            W_bin = out
    return W_bin

def bicmp_to_bipolar(W):
    # map {0,1} -> {-1, +1}
    return (W.astype(np.int8) * 2) - 1

def normalized_correlation(W_ref, W_extr):
    # expects same shape
    Wb = bicmp_to_bipolar(W_ref).astype(float)
    Wh = bicmp_to_bipolar(W_extr).astype(float)
    return float(np.sum(Wb * Wh) / (Wb.size + EPS))

def bit_error_rate(W_ref, W_extr):
    # align shapes by cropping/padding if needed
    Nr, Mr = W_ref.shape
    Ne, Me = W_extr.shape
    if (Nr, Mr) != (Ne, Me):
        W = np.zeros((Nr, Mr), dtype=np.uint8)
        h = min(Nr, Ne)
        w = min(Mr, Me)
        W[:h, :w] = W_extr[:h, :w]
        W_extr = W
    return float(np.mean(W_ref.flatten() != W_extr.flatten()))

# ---------------- helpers for safe param loading -----------------
def safe_scalar(val):
    try:
        arr = np.array(val)
        if arr.size == 0:
            return 0
        return int(arr.flatten()[0])
    except Exception:
        try:
            return int(val)
        except Exception:
            return 0

def load_embed_params(params_path):
    data = np.load(params_path, allow_pickle=True)
    N = safe_scalar(data.get("N", 0))
    M = safe_scalar(data.get("M", 0))
    params = {}
    for k in data.files:
        if k in ("N", "M"):
            continue
        v = data[k]
        try:
            if np.asarray(v).shape == ():
                params[k] = v.item()
            else:
                params[k] = v
        except Exception:
            params[k] = v
    return N, M, params

# ---------------- main test runner -----------------
def run_attack_and_eval(name, attack_fn, audio_orig, audio_wm, Fs, N, M, params, watermark_bin, outdir, save_audio=True):
    print(f"\n--- Attack: {name} ---")
    attacked = attack_fn(audio_wm, Fs)
    # ensure same length as original (trim/ pad)
    if len(attacked) < len(audio_orig):
        attacked = np.pad(attacked, (0, len(audio_orig) - len(attacked)))
    elif len(attacked) > len(audio_orig):
        attacked = attacked[:len(audio_orig)]

    # SNR wrt original (clean) audio
    snr_db = compute_snr(audio_orig, attacked)
    print(f"SNR (original vs attacked): {snr_db:.3f} dB")

    # recompute params for extraction depending on attacked length (like main.py)
    params_for_extr = dict(params)  # shallow copy
    if "L1" in params_for_extr:
        try:
            params_for_extr["Lseg"] = int(np.ceil((N * M) / max(1, int(params_for_extr["L1"]))))
        except Exception:
            params_for_extr["Lseg"] = int(params_for_extr.get("Lseg", 1))
    else:
        params_for_extr["Lseg"] = int(params_for_extr.get("Lseg", 1))
    if params_for_extr["Lseg"] <= 0:
        params_for_extr["Lseg"] = 1
    params_for_extr["L"] = int(np.floor(len(attacked) / params_for_extr["Lseg"])) if params_for_extr["Lseg"] > 0 else len(attacked)

    # Extract watermark (expects extractor.extract_watermark_from_audio signature)
    W_extracted = extract_watermark_from_audio(attacked, Fs, N, M, params_for_extr)
    # ensure binary 0/1
    W_extr_bin = (np.asarray(W_extracted) > 0).astype(np.uint8)

    # if flattened/incorrect shape but same total size, reshape
    if W_extr_bin.size == watermark_bin.size and W_extr_bin.shape != watermark_bin.shape:
        W_extr_bin = W_extr_bin.reshape(watermark_bin.shape)
    else:
        # crop/pad if needed
        if W_extr_bin.shape != watermark_bin.shape:
            H, W = watermark_bin.shape
            out = np.zeros((H, W), dtype=np.uint8)
            h = min(H, W_extr_bin.shape[0])
            w = min(W, W_extr_bin.shape[1])
            out[:h, :w] = W_extr_bin[:h, :w]
            W_extr_bin = out

    # BER and NC
    ber = bit_error_rate(watermark_bin, W_extr_bin)
    nc = normalized_correlation(watermark_bin, W_extr_bin)
    print(f"BER (watermark): {ber:.6f}")
    print(f"Normalized Correlation (NC): {nc:.6f}")

    # save one non-transposed PNG (0/1 -> 0/255)
    png_path = os.path.join(outdir, f"{name}_extracted.png")
    try:
        img_to_save = (W_extr_bin * 255).astype(np.uint8)
        imageio.imwrite(png_path, img_to_save)
        print("Saved extracted watermark PNG to", png_path)
    except Exception as e:
        print("Failed to save PNG:", e)

    # save attacked audio
    if save_audio:
        wav_path = os.path.join(outdir, f"{name}_attacked.wav")
        try:
            sf.write(wav_path, attacked.astype(np.float32), Fs)
            print("Saved attacked audio to", wav_path)
        except Exception as e:
            print("Failed to save attacked audio:", e)

    return {"attack": name, "snr_db": snr_db, "ber": ber, "nc": nc}

def main():
    parser = argparse.ArgumentParser(description="Test DSP attacks on watermarked audio and evaluate SNR, BER, NC")
    parser.add_argument("--original",default='The_Color_Violet.wav' , help="Path to original (clean) audio WAV")
    parser.add_argument("--watermarked", default= 'watermarked.wav', help="Path to watermarked audio WAV")
    parser.add_argument("--watermark", default='watermark.png', help="Path to original watermark image (PNG/JPG)")
    parser.add_argument("--params", default="embed_params.npz", help="Path to embed_params.npz (saved during embedding)")
    parser.add_argument("--outdir", default="results", help="Directory to save outputs")
    parser.add_argument("--attacks", nargs="*", choices=["lpf","hpf","crop","noise"], default=None,
                        help="Which attacks to run. If omitted --all-attacks runs all.")
    parser.add_argument("--lpf-cutoff", type=float, default=4000.0, help="LPF cutoff frequency (Hz)")
    parser.add_argument("--hpf-cutoff", type=float, default=300.0, help="HPF cutoff frequency (Hz)")
    parser.add_argument("--crop-fraction", type=float, default=0.2, help="Fraction of samples to zero for cropping attack")
    parser.add_argument("--noise-snr", type=float, default=20.0, help="Target SNR (dB) for zero-mean Gaussian noise attack")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for cropping/noise")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading audios...")
    audio_orig, Fs1 = sf.read(args.original)
    audio_wm, Fs2 = sf.read(args.watermarked)
    if Fs1 != Fs2:
        print("Warning: sampling rates differ between original and watermarked.")
    Fs = Fs2

    # make mono if needed
    if audio_orig.ndim > 1:
        audio_orig = audio_orig.mean(axis=1)
    if audio_wm.ndim > 1:
        audio_wm = audio_wm.mean(axis=1)

    N, M, params = load_embed_params(args.params)
    if N == 0 or M == 0:
        print("Warning: failed to load N,M from params. Trying to infer from watermark image.")
        try:
            W_tmp = imageio.imread(args.watermark)
            if W_tmp.ndim == 3:
                W_tmp = np.mean(W_tmp, axis=2)
            N, M = W_tmp.shape
            print(f"Inferred N,M from watermark image: {N} x {M}")
        except Exception:
            raise RuntimeError("Cannot infer watermark dimensions. Ensure embed_params.npz or watermark image exists.")

    watermark_bin = prepare_watermark_image(args.watermark, N_expected=N, M_expected=M)

    # choose attacks
    if args.attacks is None:
        attacks_to_run = ["lpf", "hpf", "crop", "noise"]
    else:
        attacks_to_run = args.attacks

    results = []
    for atk in attacks_to_run:
        if atk == "lpf":
            name = f"LPF_{int(args.lpf_cutoff)}Hz"
            fn = lambda sig, Fs_local, c=args.lpf_cutoff: fft_lowpass(sig, Fs_local, c)
        elif atk == "hpf":
            name = f"HPF_{int(args.hpf_cutoff)}Hz"
            fn = lambda sig, Fs_local, c=args.hpf_cutoff: fft_highpass(sig, Fs_local, c)
        elif atk == "crop":
            name = f"CROP_{int(args.crop_fraction*100)}pct"
            fn = lambda sig, Fs_local, f=args.crop_fraction, s=args.seed: cropping_random_zero(sig, f, seed=s)
        elif atk == "noise":
            name = f"GAUSS_noise_{int(args.noise_snr)}dB"
            fn = lambda sig, Fs_local, s=args.seed, snr=args.noise_snr: add_zero_mean_gaussian_noise(sig, target_snr_db=snr, seed=s)
        else:
            continue
        res = run_attack_and_eval(name, fn, audio_orig, audio_wm, Fs, N, M, params, watermark_bin, args.outdir)
        results.append(res)

    # print summary
    print("\n=== Summary ===")
    print(f"{'Attack':30s} {'SNR(dB)':>10s} {'BER':>10s} {'NC':>10s}")
    for r in results:
        print(f"{r['attack']:30s} {r['snr_db']:10.3f} {r['ber']:10.6f} {r['nc']:10.6f}")

    # save CSV
    try:
        import csv
        csv_path = os.path.join(args.outdir, "results_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["attack","snr_db","ber","nc"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print("Saved summary CSV to", csv_path)
    except Exception:
        pass

if __name__ == "__main__":
    main()
