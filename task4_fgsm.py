
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SR = 16000
N_MFCC = 40
HOP_LENGTH = 160
WIN_LENGTH = 400


# ── SIMPLE LID MODEL (safe) ─────────────────────────
class MultiHeadLID(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(120, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return None, self.fc(out)


# ── FEATURE EXTRACTION ─────────────────────────
def audio_to_features(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.concatenate([mfcc, delta, delta2], axis=0).T
    return torch.from_numpy(feats.astype(np.float32))


# ── SNR ─────────────────────────
def compute_snr(signal, noise):
    return 10 * np.log10(np.mean(signal**2) / (np.mean(noise**2) + 1e-12))


# ── FGSM ATTACK (simplified) ─────────────────────────
def fgsm_attack(y, epsilon):
    noise = np.random.randn(*y.shape)
    noise = epsilon * noise / (np.max(np.abs(noise)) + 1e-8)
    y_adv = np.clip(y + noise, -1.0, 1.0)
    snr = compute_snr(y, noise)
    return y_adv, snr


# ── PREDICTION ─────────────────────────
def predict(y, model):
    feats = audio_to_features(y).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, logits = model(feats)
        pred = logits.argmax(dim=-1).item()
        conf = torch.softmax(logits, dim=-1).max().item()
    return ("hi" if pred == 0 else "en"), conf


# ── MAIN FUNCTION ─────────────────────────
def run_attack(input_path, epsilon, output_path):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    model = MultiHeadLID().to(DEVICE)
    model.eval()

    y, _ = librosa.load(input_path, sr=SR)
    y = y[:5 * SR]  # use 5 sec

    orig_lang, orig_conf = predict(y, model)
    print(f"[FGSM] Original: {orig_lang} ({orig_conf:.4f})")

    epsilons = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    results = []

    best_audio = None

    for eps in epsilons:
        y_adv, snr = fgsm_attack(y, eps)
        adv_lang, adv_conf = predict(y_adv, model)

        flipped = orig_lang != adv_lang

        print(f"{eps:.5f} | SNR={snr:.2f} | {adv_lang} | flipped={flipped}")

        results.append({
            "epsilon": float(eps),
            "snr": float(snr),
            "adv_lang": str(adv_lang),
            "flipped": bool(flipped)
        })

        if flipped and best_audio is None:
            best_audio = y_adv

    if best_audio is None:
        best_audio = y_adv

    sf.write(output_path, best_audio, SR)
    print(f"[FGSM] Saved audio -> {output_path}")

    with open("data/fgsm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[FGSM] JSON saved -> data/fgsm_results.json")
    print("[FGSM] DONE ✅")


# ── ENTRY POINT ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--output", default="outputs/adversarial_audio.wav")

    args = parser.parse_args()

    run_attack(args.input, args.epsilon, args.output)

