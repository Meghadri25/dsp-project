# main.py
import numpy as np
import os
import soundfile as sf
import argparse
import json
import imageio
from embed import embed_watermark
from extractor import extract_watermark_from_audio
from utils import (
    load_audio,
    load_image,
    save_image,
    save_parameters,
    load_parameters,
)


def embed():
    key = {"epsilon": 0.3, "eta": 0.2, "mu": 3.99, "x0": 0.3456789, "iterations": 1000}

    output_dir = "results/embed_results"
    os.makedirs(output_dir, exist_ok=True)

    audio_in = "The_Color_Violet.wav"
    watermark_img = "watermark.png"
    audio_out = os.path.join(output_dir, "watermarked.wav")

    audio_signal, Fs = load_audio(audio_in)
    W_bin, N, M = load_image(watermark_img)

    params, dims, out_audio, Fs_out = embed_watermark(
        audio_signal, Fs, W_bin, N, M, L1=4, delta=0.05, mlncml_key=key
    )

    sf.write(audio_out, out_audio, Fs_out)
    print("Watermarked audio written to:", audio_out)

    signal_power = np.mean(audio_signal**2)
    noise = out_audio - audio_signal
    noise_power = np.mean(noise**2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    print(f"SNR: {snr_db:.2f} dB")

    print("Embedding complete.")
    print("params:", params)
    print("dims:", dims)

    save_parameters(params, dims, output_dir)


def extract():
    extract_output_dir = "results/extract_results"
    os.makedirs(extract_output_dir, exist_ok=True)

    N, M, params = load_parameters()
    audio_signal, Fs = load_audio("results/embed_results/watermarked.wav")

    W_extracted = extract_watermark_from_audio(audio_signal, Fs, N, M, params)

    save_image(W_extracted, N, M, extract_output_dir)

    print("Extraction complete.")
    print(f"Results saved to: {extract_output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Watermark embedding and extraction tool"
    )
    parser.add_argument(
        "--mode",
        default="embed",
        choices=["embed", "extract"],
        help="Mode: embed or extract (default: embed)",
    )
    args = parser.parse_args()
    mode = args.mode

    if mode == "embed":
        embed()
    elif mode == "extract":
        extract()
