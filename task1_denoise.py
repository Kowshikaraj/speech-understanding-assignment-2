"""
Task 1.3 - Denoising & Normalization
Uses DeepFilterNet to remove classroom background noise and reverb.
Fallback: Spectral Subtraction if DeepFilterNet unavailable.

Run:
    python task1_denoise.py --input audio/lecture.wav --output audio/lecture_clean.wav
"""

import argparse
import numpy as np
import soundfile as sf
import librosa
import os


def spectral_subtraction(y, sr, noise_frames=20, alpha=2.0):
    """Fallback: classic spectral subtraction denoising."""
    print("[Fallback] Using Spectral Subtraction...")
    S = librosa.stft(y)
    mag = np.abs(S)
    phase = np.angle(S)

    # Estimate noise from first N frames
    noise_est = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)

    # Subtract scaled noise estimate
    mag_denoised = np.maximum(mag - alpha * noise_est, 0.0)

    # Reconstruct signal
    S_denoised = mag_denoised * np.exp(1j * phase)
    y_denoised = librosa.istft(S_denoised, length=len(y))
    return y_denoised


def denoise_deepfilternet(input_path, output_path):
    """Primary: DeepFilterNet-based denoising."""
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        from df import config

        print("[DeepFilterNet] Loading model...")
        model, df_state, _ = init_df()

        print(f"[DeepFilterNet] Loading audio: {input_path}")
        audio, _ = load_audio(input_path, sr=df_state.sr())

        print("[DeepFilterNet] Enhancing (denoising)...")
        enhanced = enhance(model, df_state, audio)

        print(f"[DeepFilterNet] Saving to: {output_path}")
        save_audio(output_path, enhanced, df_state.sr())
        print("[DeepFilterNet] Done!")
        return True

    except ImportError:
        print("[Warning] DeepFilterNet not installed. Falling back to Spectral Subtraction.")
        return False
    except Exception as e:
        print(f"[Warning] DeepFilterNet failed: {e}. Falling back.")
        return False


def normalize_audio(y):
    """Peak normalize audio to -1dB."""
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.891  # -1dBFS
    return y


def denoise(input_path, output_path):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Try DeepFilterNet first
    success = denoise_deepfilternet(input_path, output_path)

    if not success:
        # Fallback: spectral subtraction
        print(f"[Denoise] Loading: {input_path}")
        y, sr = librosa.load(input_path, sr=None, mono=True)
        print(f"[Denoise] Sample rate: {sr} Hz, Duration: {len(y)/sr:.1f}s")

        y_denoised = spectral_subtraction(y, sr)
        y_normalized = normalize_audio(y_denoised)

        print(f"[Denoise] Saving clean audio to: {output_path}")
        sf.write(output_path, y_normalized, sr)
        print("[Denoise] Done!")

    # Verify output
    y_out, sr_out = librosa.load(output_path, sr=None)
    print(f"[Denoise] Output: {output_path} | SR={sr_out}Hz | Duration={len(y_out)/sr_out:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise lecture audio")
    parser.add_argument("--input",  required=True, help="Path to noisy input WAV")
    parser.add_argument("--output", required=True, help="Path to save clean WAV")
    args = parser.parse_args()

    denoise(args.input, args.output)