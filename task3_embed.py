"""
Task 3.1 - Voice Embedding Extraction
Extracts a high-dimensional speaker d-vector/x-vector from your 60s voice recording
using SpeechBrain's ECAPA-TDNN pre-trained model.

Run:
    python task3_embed.py --voice audio/student_voice_ref.wav \
                          --output models/speaker_embedding.pt
"""

import os
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf


TARGET_SR  = 16000
MIN_DUR    = 5.0     # minimum seconds of audio required
TARGET_DUR = 60.0    # expected reference duration


def resample_audio(input_path, output_path, target_sr=TARGET_SR):
    """Resample audio to target sample rate and save."""
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    sf.write(output_path, y, target_sr)
    return y, target_sr


def extract_embedding_speechbrain(audio_path, output_path):
    """Primary: SpeechBrain ECAPA-TDNN speaker embedding."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        print("[Embed] Loading SpeechBrain ECAPA-TDNN...")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/speechbrain_ecapa"
        )

        print(f"[Embed] Extracting embedding from: {audio_path}")
        signal, fs = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        signal_t   = torch.from_numpy(signal).unsqueeze(0)

        with torch.no_grad():
            embedding = classifier.encode_batch(signal_t)

        # embedding shape: (1, 1, 192) for ECAPA
        emb = embedding.squeeze().cpu()
        print(f"[Embed] Embedding shape: {emb.shape}")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        torch.save({
            "embedding":   emb,
            "source":      audio_path,
            "model":       "ECAPA-TDNN",
            "dim":         emb.shape[-1],
            "sample_rate": TARGET_SR
        }, output_path)
        print(f"[Embed] Saved embedding -> {output_path}")
        return emb

    except ImportError:
        print("[Warning] SpeechBrain not installed. Trying HuggingFace wav2vec2 fallback...")
        return None
    except Exception as e:
        print(f"[Warning] SpeechBrain failed: {e}")
        return None


def extract_embedding_wav2vec(audio_path, output_path):
    """Fallback: wav2vec2-based speaker embedding via mean pooling."""
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        print("[Embed] Loading Wav2Vec2 for speaker embedding (fallback)...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        model.eval()

        y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)

        # Process in 10s chunks and average
        chunk_size = TARGET_SR * 10
        embeddings = []
        for start in range(0, len(y), chunk_size):
            chunk = y[start:start + chunk_size]
            if len(chunk) < TARGET_SR:  # skip too-short last chunk
                continue
            inputs = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs).last_hidden_state  # (1, T, 768)
            embeddings.append(out.mean(dim=1).squeeze())  # (768,)

        emb = torch.stack(embeddings).mean(dim=0)  # (768,)
        print(f"[Embed] Wav2Vec2 embedding shape: {emb.shape}")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        torch.save({
            "embedding":   emb.cpu(),
            "source":      audio_path,
            "model":       "wav2vec2-base-960h",
            "dim":         emb.shape[-1],
            "sample_rate": TARGET_SR
        }, output_path)
        print(f"[Embed] Saved embedding -> {output_path}")
        return emb

    except Exception as e:
        print(f"[Error] Wav2Vec2 also failed: {e}")
        return None


def extract_embedding_mfcc(audio_path, output_path):
    """Last resort: d-vector from MFCC statistics (simple but works)."""
    print("[Embed] Using MFCC statistics as fallback embedding...")
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stats: mean, std, min, max per coefficient
    feats = np.concatenate([mfcc, delta, delta2], axis=0)  # (120, T)
    emb_np = np.concatenate([
        feats.mean(axis=1),
        feats.std(axis=1),
        feats.min(axis=1),
        feats.max(axis=1)
    ])  # (480,)

    emb = torch.from_numpy(emb_np.astype(np.float32))
    print(f"[Embed] MFCC embedding shape: {emb.shape}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save({
        "embedding":   emb,
        "source":      audio_path,
        "model":       "mfcc-stats",
        "dim":         emb.shape[-1],
        "sample_rate": TARGET_SR
    }, output_path)
    print(f"[Embed] Saved embedding -> {output_path}")
    return emb


def validate_audio(audio_path):
    """Verify audio meets requirements."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr
    print(f"[Embed] Audio duration: {duration:.1f}s | Sample rate: {sr}Hz")

    if duration < MIN_DUR:
        raise ValueError(f"Audio too short ({duration:.1f}s). Need at least {MIN_DUR}s.")
    if duration < TARGET_DUR * 0.8:
        print(f"[Warning] Audio is {duration:.1f}s (target: {TARGET_DUR}s). Shorter = worse embedding.")

    return y, sr


def extract_embedding(voice_path, output_path):
    validate_audio(voice_path)

    # Try methods in order of quality
    emb = extract_embedding_speechbrain(voice_path, output_path)
    if emb is None:
        emb = extract_embedding_wav2vec(voice_path, output_path)
    if emb is None:
        emb = extract_embedding_mfcc(voice_path, output_path)

    if emb is not None:
        saved = torch.load(output_path)
        print(f"\n[Embed] Speaker embedding extracted!")
        print(f"  Model:     {saved['model']}")
        print(f"  Dimension: {saved['dim']}")
        print(f"  File:      {output_path}")
    else:
        raise RuntimeError("All embedding methods failed.")

    return emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract speaker embedding from voice recording")
    parser.add_argument("--voice",  required=True, help="Path to 60s reference WAV (16kHz mono)")
    parser.add_argument("--output", default="models/speaker_embedding.pt")
    args = parser.parse_args()

    extract_embedding(args.voice, args.output)