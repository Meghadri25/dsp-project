import os
import numpy as np
import csv
import soundfile as sf
import tempfile
from pydub import AudioSegment
import librosa
from scipy.ndimage import laplace
from skimage.metrics import structural_similarity as ssim

from extractor import extract_watermark_from_audio
from utils import load_audio, load_image, load_parameters, save_image

EPS = 1e-12


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
    idx = rng.choice(n, size=k, replace=False)
    out = signal.copy()
    out[idx] = 0.0
    return out


def add_zero_mean_gaussian_noise(signal, target_snr_db=20.0, seed=None):
    rng = np.random.default_rng(seed)
    sig = np.asarray(signal, dtype=float)
    p_signal = np.mean(sig**2)
    target_linear = 10.0 ** (target_snr_db / 10.0)
    p_noise = p_signal / max(target_linear, 1e-12)
    sigma = np.sqrt(p_noise)
    noise = rng.normal(loc=0.0, scale=sigma, size=sig.shape)
    return sig + noise


def compute_snr(clean, test):
    n = min(len(clean), len(test))
    clean = np.asarray(clean[:n], dtype=float)
    test = np.asarray(test[:n], dtype=float)
    signal_power = np.sum(clean**2)
    noise_power = np.sum((clean - test) ** 2)
    if noise_power <= 0:
        return float("inf")
    snr_db = 10.0 * np.log10((signal_power + EPS) / (noise_power + EPS))
    return snr_db


def si_sdr(clean, test):

    reference = clean.astype(np.float64)
    estimated = test.astype(np.float64)

    min_len = min(len(reference), len(estimated))
    reference = reference[:min_len]
    estimated = estimated[:min_len]

    alpha = np.dot(estimated, reference) / np.dot(reference, reference)
    projected = alpha * reference

    noise = estimated - projected

    ratio = np.sum(projected ** 2) / np.sum(noise ** 2)
    si_sdr_value = 10 * np.log10(ratio)

    return si_sdr_value


def bicmp_to_bipolar(W):
    return (W.astype(np.int8) * 2) - 1


def normalized_correlation(W_ref, W_extr):
    Wb = bicmp_to_bipolar(W_ref).astype(float)
    Wh = bicmp_to_bipolar(W_extr).astype(float)
    return float(np.sum(Wb * Wh) / (Wb.size + EPS))

def laplacian_mse(W_ref, W_extr):
    Wb = W_ref.astype(float)
    Wh = W_extr.astype(float)

    lap_ref = laplace(Wb)
    lap_extr = laplace(Wh)

    mse = np.mean((lap_ref - lap_extr) ** 2)
    return float(mse)

def structural_similarity(W_ref, W_extr):
    Wb = W_ref.astype(float)
    Wh = W_extr.astype(float)

    ssim_value= ssim(Wb, Wh, data_range = Wh.max() - Wh.min(), full=False)
    return float(ssim_value)

def bit_error_rate(W_ref, W_extr):
    Nr, Mr = W_ref.shape
    Ne, Me = W_extr.shape
    if (Nr, Mr) != (Ne, Me):
        W = np.zeros((Nr, Mr), dtype=np.uint8)
        h = min(Nr, Ne)
        w = min(Mr, Me)
        W[:h, :w] = W_extr[:h, :w]
        W_extr = W
    return float(np.mean(W_ref.flatten() != W_extr.flatten()))


def mp3_compression_decompression(signal, Fs, bitrate=128):
    try:
        # Convert to 16-bit int
        signal_int = (signal * 32767).astype(np.int16)
        # Create AudioSegment
        audio = AudioSegment(
            signal_int.tobytes(), frame_rate=int(Fs), sample_width=2, channels=1
        )
        # Temp files
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3_path = tmp_mp3.name
        try:
            # Export to MP3
            audio.export(tmp_mp3_path, format="mp3", bitrate=f"{bitrate}k")
            # Import back
            audio_back = AudioSegment.from_mp3(tmp_mp3_path)
            # To numpy
            samples = np.array(audio_back.get_array_of_samples())
            if audio_back.channels > 1:
                samples = samples.reshape((-1, audio_back.channels)).mean(axis=1)
            # Normalize back to float
            out = samples.astype(np.float64) / 32768.0
        finally:
            os.unlink(tmp_mp3_path)
        return out
    except (FileNotFoundError, PermissionError):
        print("FFmpeg not found, skipping MP3 compression. Returning original signal.")
        return signal


def generate_simple_rir(length=2048, decay=0.95):
    rir = np.zeros(length)
    rir[0] = 1.0
    for i in range(1, length):
        rir[i] = rir[i - 1] * decay
    return rir


def re_recording_attack(signal, Fs):
    # Generate RIR
    rir = generate_simple_rir(length=2048, decay=0.95)
    # Convolve
    convolved = np.convolve(signal, rir, mode="full")[: len(signal)]
    # Add reverb: add delayed copies
    delay_samples = int(0.1 * Fs)  # 100ms
    reverb_signal = (
        convolved
        + 0.5 * np.roll(convolved, delay_samples)
        + 0.3 * np.roll(convolved, 2 * delay_samples)
    )
    # Normalize
    reverb_signal = (
        reverb_signal / np.max(np.abs(reverb_signal)) * np.max(np.abs(signal))
    )
    # Low-pass at 5kHz
    filtered = fft_lowpass(reverb_signal, Fs, 5000)
    return filtered


def randomized_time_scaling(signal, Fs, seed=None):
    rng = np.random.default_rng(seed)
    # Divide into segments, say 10 segments
    n_segments = 10
    segment_len = len(signal) // n_segments
    scaled_segments = []
    for i in range(n_segments):
        start = i * segment_len
        end = (i + 1) * segment_len if i < n_segments - 1 else len(signal)
        seg = signal[start:end]
        # Random speed factor 0.8 to 1.2
        speed = rng.uniform(0.8, 1.2)
        # Time stretch
        stretched = librosa.effects.time_stretch(seg, rate=speed)
        scaled_segments.append(stretched)
    # Concatenate
    out = np.concatenate(scaled_segments)
    # Resize to original length
    out = np.resize(out, len(signal))
    return out


# ---------------- main test runner -----------------
def main():
    # Hardcoded parameters
    original = "The_Color_Violet.wav"
    watermarked = "results/embed_results/watermarked.wav"
    watermark = "watermark.png"
    outdir = "results/attack_results"
    lpf_cutoff = 4000.0
    hpf_cutoff = 100.0
    crop_fraction = 0.2
    noise_snr = 20.0
    seed = 1234

    os.makedirs(outdir, exist_ok=True)

    audio_orig, Fs = load_audio(original)
    audio_wm, Fs = load_audio(watermarked)

    N, M, params = load_parameters()

    watermark_bin, N, M = load_image(watermark)

    results = []
    attack_configs = [
        (
            f"LPF_{int(lpf_cutoff)}Hz",
            lambda sig, Fs_local, c=lpf_cutoff: fft_lowpass(sig, Fs_local, c),
        ),
        (
            f"HPF_{int(hpf_cutoff)}Hz",
            lambda sig, Fs_local, c=hpf_cutoff: fft_highpass(sig, Fs_local, c),
        ),
        (
            f"CROP_{int(crop_fraction*100)}pct",
            lambda sig, Fs_local, f=crop_fraction, s=seed: cropping_random_zero(
                sig, f, seed=s
            ),
        ),
        (
            f"GAUSS_noise_{int(noise_snr)}dB",
            lambda sig, Fs_local, s=seed, snr=noise_snr: add_zero_mean_gaussian_noise(
                sig, target_snr_db=snr, seed=s
            ),
        ),
        (
            "MP3_128kbps",
            lambda sig, Fs_local: mp3_compression_decompression(sig, Fs_local, 128),
        ),
        (
            "MP3_64kbps",
            lambda sig, Fs_local: mp3_compression_decompression(sig, Fs_local, 64),
        ),
        (
            "MP3_32kbps",
            lambda sig, Fs_local: mp3_compression_decompression(sig, Fs_local, 32),
        ),
        (
            "Re_recording",
            lambda sig, Fs_local: re_recording_attack(sig, Fs_local),
        ),
        (
            "Random_time_scaling",
            lambda sig, Fs_local, s=seed: randomized_time_scaling(
                sig, Fs_local, seed=s
            ),
        ),
    ]

    csv_path = os.path.join(outdir, "results_summary.csv")
    existing_results = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_results = list(reader)

    baseline_exists = any(r.get("attack") == "No_Attack_Baseline" for r in existing_results)
    if not baseline_exists:
        print("\n--- Baseline: No Attack ---")
        baseline_snr = compute_snr(audio_orig, audio_wm)
        baseline_si_sdr = si_sdr(audio_orig, audio_wm)

        W_baseline = extract_watermark_from_audio(audio_wm, Fs, N, M, params)
        W_baseline_bin = (np.asarray(W_baseline) > 0).astype(np.uint8)

        baseline_dir = os.path.join(outdir, "No_Attack_Baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        save_image(W_baseline, N, M, baseline_dir)

        baseline_ber = bit_error_rate(watermark_bin, W_baseline_bin)
        baseline_nc = normalized_correlation(watermark_bin, W_baseline_bin)
        baseline_lap_mse = laplacian_mse(watermark_bin, W_baseline_bin)
        baseline_ssim = structural_similarity(watermark_bin, W_baseline_bin)
                
        baseline_result = {
            "attack": "No_Attack_Baseline",
            "snr_db": baseline_snr,
            "si_sdr": baseline_si_sdr,
            "ber": baseline_ber,
            "nc": baseline_nc,
            "lap_mse": baseline_lap_mse,
            "ssim": baseline_ssim
        }
        results.insert(0, baseline_result)

    for name, fn in attack_configs:
        attack_dir = os.path.join(outdir, name)
        if os.path.exists(attack_dir):
            print(f"Skipping {name}, directory already exists.")
            continue

        os.makedirs(attack_dir, exist_ok=True)

        attacked = fn(audio_wm, Fs)
        attacked = np.resize(attacked, len(audio_orig))

        audio_path = os.path.join(attack_dir, "attacked.wav")
        sf.write(audio_path, attacked, Fs)
        print(f"Attacked audio saved to: {audio_path}")

        snr_db = compute_snr(audio_orig, attacked)
        si_sdr_value = si_sdr(audio_orig, attacked)

        W_extracted = extract_watermark_from_audio(attacked, Fs, N, M, params)
        W_extr_bin = (np.asarray(W_extracted) > 0).astype(np.uint8)

        save_image(W_extracted, N, M, attack_dir)

        ber = bit_error_rate(watermark_bin, W_extr_bin)
        nc = normalized_correlation(watermark_bin, W_extr_bin)
        lap_mse = laplacian_mse(watermark_bin, W_extr_bin)
        ssim = structural_similarity(watermark_bin, W_extr_bin)

        results.append({"attack": name, "snr_db": snr_db, "si_sdr": si_sdr_value, "ber": ber, "nc": nc, "lap_mse": lap_mse, "ssim": ssim})

    # Combine existing and new results
    all_results = existing_results + results

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "snr_db", "si_sdr", "ber", "nc", "lap_mse", "ssim"])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)


if __name__ == "__main__":
    main()
