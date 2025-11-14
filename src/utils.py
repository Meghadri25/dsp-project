# utils.py
import numpy as np
import struct
import zlib
import imageio
import soundfile as sf
from typing import Tuple
import os
import json
from scipy.ndimage import zoom


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Load and preprocess audio signal."""
    audio, Fs = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    return audio, Fs


def load_image(watermark_path: str) -> tuple[np.ndarray, int, int]:
    """Load and preprocess watermark image."""
    W_img = imageio.imread(watermark_path)
    if W_img.ndim == 3:
        W_img = np.mean(W_img, axis=2)

    height, width = W_img.shape

    new_width = 200
    new_height = int(height * new_width / width)
    W_img = zoom(W_img, (new_height / height, new_width / width))

    W_bin = (W_img > W_img.mean()).astype(np.uint8)
    N, M = W_bin.shape
    return W_bin, N, M


def save_image(W_extracted, N, M, output_dir):
    """Reshape and scale extracted watermark for PNG and save it."""
    arr = np.asarray(W_extracted)
    if arr.ndim == 1:
        arr = arr.reshape((N, M))

    if np.issubdtype(arr.dtype, np.bool_) or (
        arr.size and np.max(arr) <= 1 and np.min(arr) >= 0
    ):
        arr = arr.astype(np.uint8) * 255
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    png_path = os.path.join(output_dir, "extracted_watermark.png")
    imageio.imwrite(png_path, arr)
    print(f"Saved extracted watermark to {png_path}")


def save_parameters(params, dims, output_dir):
    """Save embedding parameters to JSON file."""
    params_file = os.path.join(output_dir, "embed_params.json")
    params_dict = {k: v.item() if hasattr(v, "item") else v for k, v in params.items()}
    params_dict["dims"] = dims
    with open(params_file, "w") as f:
        json.dump(params_dict, f)
    print(f"Saved embedding parameters to {params_file}")


def load_parameters() -> tuple[int, int, dict]:
    """Load embedding parameters from JSON file."""
    params_path = "results/embed_results/embed_params.json"
    with open(params_path, "r") as f:
        data = json.load(f)
    dims = data["dims"]
    N, M = dims
    params = {k: data[k] for k in data if k != "dims"}
    return N, M, params
