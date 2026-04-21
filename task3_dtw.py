"""
Task 3.2 - Prosody Warping via Dynamic Time Warping (DTW)
Extracts F0 (pitch) and Energy from the professor's lecture, then
applies DTW to map these contours onto synthesized speech.

Run:
    python task3_dtw.py --source audio/lecture_clean.wav \
                        --target audio/synth_flat.wav \
                        --output audio/synth_prosody.wav
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import torch
from scipy.signal import medfilt
from scipy.interpolate import interp1d


# ── F0 Extraction ─────────────────────────────────────────────────────────────
def extract_f0(y, sr, hop_length=256, fmin=50, fmax=600):
    """
    Extract fundamental frequency (F0/pitch) contour.
    Uses pyworld (WORLD vocoder) if available, else librosa pyin.
    """
    try:
        import pyworld as pw
        y_64 = y.astype(np.float64)
        f0, sp, ap = pw.dio(y_64, sr, f0_floor=fmin, f0_ceil=fmax,
                             frame_period=hop_length/sr*1000)
        f0 = pw.stonemask(y_64, f0, sp, ap, sr)
        return f0.astype(np.float32)
    except ImportError:
        pass

    # Fallback: librosa pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32)


def extract_energy(y, sr, hop_length=256, win_length=1024):
    """Extract RMS energy contour from audio."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=win_length)[0]
    return rms.astype(np.float32)


def smooth_contour(contour, kernel_size=5):
    """Apply median filter to smooth F0/energy contour."""
    if len(contour) < kernel_size:
        return contour
    return medfilt(contour, kernel_size=kernel_size)


def interpolate_contour(contour, target_len):
    """Stretch/squeeze a contour to a different length."""
    src_len = len(contour)
    if src_len == target_len:
        return contour
    src_x = np.linspace(0, 1, src_len)
    tgt_x = np.linspace(0, 1, target_len)
    interp = interp1d(src_x, contour, kind="linear", fill_value="extrapolate")
    return interp(tgt_x).astype(np.float32)


# ── DTW Alignment ─────────────────────────────────────────────────────────────
def dtw_align(source, target):
    """
    Apply DTW to find the optimal alignment path between source and target contours.
    Returns the warped source aligned to target length.
    """
    try:
        from dtw import dtw as dtw_func
        alignment = dtw_func(source.reshape(-1, 1),
                             target.reshape(-1, 1),
                             keep_internals=True)
        # Map source frames to target frames via the warping path
        path_src = alignment.index1
        path_tgt = alignment.index2

        warped = np.zeros(len(target), dtype=np.float32)
        counts = np.zeros(len(target), dtype=np.float32)
        for s_idx, t_idx in zip(path_src, path_tgt):
            if t_idx < len(warped):
                warped[t_idx] += source[s_idx]
                counts[t_idx] += 1
        counts[counts == 0] = 1
        return warped / counts

    except ImportError:
        print("[DTW] dtw-python not installed. Using linear interpolation.")
        return interpolate_contour(source, len(target))


# ── WORLD Vocoder Resynthesis ─────────────────────────────────────────────────
def resynthesize_with_prosody(y_target, sr, f0_new, energy_new, hop_length=256):
    """
    Replace the F0 and energy of target audio with new contours using WORLD vocoder.
    Falls back to simple pitch shifting if WORLD not available.
    """
    try:
        import pyworld as pw

        y_64 = y_target.astype(np.float64)
        f0_orig, sp, ap = pw.dio(y_64, sr, frame_period=hop_length/sr*1000)
        f0_orig = pw.stonemask(y_64, f0_orig, sp, ap, sr)

        # Align f0_new length to sp length
        n_frames = len(f0_orig)
        f0_warped = interpolate_contour(f0_new, n_frames).astype(np.float64)

        # Apply energy scaling
        energy_orig = np.sqrt(np.mean(sp, axis=1)) + 1e-8
        energy_warped = interpolate_contour(energy_new, n_frames).astype(np.float64)
        energy_scale  = (energy_warped + 1e-8) / (energy_orig + 1e-8)
        energy_scale  = np.clip(energy_scale, 0.3, 3.0)

        sp_scaled = sp * energy_scale[:, np.newaxis]

        # Synthesize
        y_synth = pw.synthesize(f0_warped, sp_scaled, ap, sr,
                                frame_period=hop_length/sr*1000)
        return y_synth.astype(np.float32)

    except ImportError:
        print("[DTW] pyworld not available. Applying simple pitch shift only.")
        # Simple fallback: pitch shift by mean F0 ratio
        f0_mean_src = np.mean(f0_new[f0_new > 0]) if np.any(f0_new > 0) else 100.0
        f0_mean_tgt = np.mean(extract_f0(y_target, sr))
        f0_mean_tgt = f0_mean_tgt if f0_mean_tgt > 0 else 100.0
        n_steps     = 12 * np.log2(f0_mean_src / f0_mean_tgt)
        n_steps     = np.clip(n_steps, -6, 6)
        return librosa.effects.pitch_shift(y_target, sr=sr, n_steps=float(n_steps))


# ── Main Prosody Transfer Function ────────────────────────────────────────────
def apply_prosody_warping(source_path, target_path, output_path, hop_length=256):
    """
    Transfer prosody (F0 + energy) from source (professor's lecture)
    to target (synthesized flat speech).
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Load both audio files
    print(f"[DTW] Loading source (professor): {source_path}")
    y_src, sr_src = librosa.load(source_path, sr=22050, mono=True)

    print(f"[DTW] Loading target (synthesized): {target_path}")
    y_tgt, sr_tgt = librosa.load(target_path, sr=22050, mono=True)

    sr = 22050

    # Extract F0 and energy from source (professor)
    print("[DTW] Extracting F0 and energy from source...")
    f0_src     = extract_f0(y_src, sr, hop_length=hop_length)
    energy_src = extract_energy(y_src, sr, hop_length=hop_length)

    f0_src     = smooth_contour(f0_src)
    energy_src = smooth_contour(energy_src)

    # Extract F0 and energy from target (synthesized)
    print("[DTW] Extracting F0 and energy from target...")
    f0_tgt     = extract_f0(y_tgt, sr, hop_length=hop_length)
    energy_tgt = extract_energy(y_tgt, sr, hop_length=hop_length)

    # Save extracted contours
    os.makedirs("data", exist_ok=True)
    np.save("data/prof_f0.npy",     f0_src)
    np.save("data/prof_energy.npy", energy_src)
    np.save("data/synth_f0.npy",    f0_tgt)
    print("[DTW] Saved F0 and energy contours to data/")

    # DTW alignment: warp source prosody to match target duration
    print("[DTW] Applying DTW alignment...")

    # Only align over voiced segments
    voiced_src = f0_src[f0_src > 0]
    voiced_tgt = f0_tgt[f0_tgt > 0]

    if len(voiced_src) > 10 and len(voiced_tgt) > 10:
        f0_warped  = dtw_align(f0_src, f0_tgt)
    else:
        print("[DTW] Insufficient voiced frames for DTW, using interpolation.")
        f0_warped  = interpolate_contour(f0_src, len(f0_tgt))

    energy_warped = interpolate_contour(energy_src, len(energy_tgt))

    # Re-synthesize target with warped prosody
    print("[DTW] Re-synthesizing with warped prosody...")
    y_out = resynthesize_with_prosody(y_tgt, sr, f0_warped, energy_warped, hop_length)

    # Save output
    sf.write(output_path, y_out, sr)
    print(f"[DTW] Prosody-warped audio saved -> {output_path}")

    # Quick stats
    f0_src_mean = np.mean(f0_src[f0_src > 0]) if np.any(f0_src > 0) else 0
    f0_tgt_mean = np.mean(f0_tgt[f0_tgt > 0]) if np.any(f0_tgt > 0) else 0
    print(f"[DTW] Source F0 mean: {f0_src_mean:.1f} Hz | Target F0 mean: {f0_tgt_mean:.1f} Hz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply DTW prosody warping")
    parser.add_argument("--source",      required=True, help="Source audio (professor lecture)")
    parser.add_argument("--target",      required=True, help="Target audio (synthesized flat speech)")
    parser.add_argument("--output",      required=True, help="Output path for prosody-warped audio")
    parser.add_argument("--hop_length",  type=int, default=256)
    args = parser.parse_args()

    apply_prosody_warping(args.source, args.target, args.output, args.hop_length)