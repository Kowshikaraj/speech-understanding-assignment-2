"""
Task 4.1 - Anti-Spoofing Countermeasure (CM) System
LFCC feature extraction + lightweight CNN classifier.
Distinguishes Bona Fide (real human) vs Spoof (synthesized TTS).
Evaluation: Equal Error Rate (EER) — must be < 10%.

Run (train):
    python task4_spoof.py --real audio/student_voice_ref.wav \
                          --spoof outputs/output_LRL_cloned.wav \
                          --mode train

Run (eval):
    python task4_spoof.py --test_audio outputs/output_LRL_cloned.wav \
                          --mode eval
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────
SR          = 16000
N_LFCC      = 60
HOP_LENGTH  = 256
WIN_LENGTH  = 1024
CHUNK_SEC   = 3.0         # Process audio in 3s chunks
BATCH_SIZE  = 16
EPOCHS      = 40
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "models/spoof_model.pt"


# ── LFCC Feature Extraction ───────────────────────────────────────────────────
def extract_lfcc(y, sr=SR, n_lfcc=N_LFCC, hop=HOP_LENGTH, win=WIN_LENGTH):
    """
    Linear Frequency Cepstral Coefficients (LFCC).
    Unlike MFCC (mel-scale), LFCC uses linear frequency spacing.
    More sensitive to artifacts introduced by TTS/vocoders.
    """
    # Short-Time Fourier Transform
    S = np.abs(librosa.stft(y, n_fft=win, hop_length=hop))  # (F, T)
    F, T = S.shape

    # Linear filterbank (instead of mel)
    n_filters = n_lfcc * 2
    linear_fb = np.zeros((n_filters, F))
    freqs     = np.linspace(0, sr // 2, F)
    centers   = np.linspace(0, sr // 2, n_filters + 2)

    for i in range(n_filters):
        lo, mid, hi = centers[i], centers[i+1], centers[i+2]
        for j, f in enumerate(freqs):
            if lo <= f <= mid:
                linear_fb[i, j] = (f - lo) / (mid - lo + 1e-8)
            elif mid < f <= hi:
                linear_fb[i, j] = (hi - f) / (hi - mid + 1e-8)

    # Apply filterbank and log
    filter_out = linear_fb @ S                                   # (n_filters, T)
    log_out    = np.log(filter_out + 1e-9)

    # DCT to get cepstral coefficients
    from scipy.fftpack import dct
    lfcc = dct(log_out, type=2, axis=0, norm="ortho")[:n_lfcc]  # (n_lfcc, T)

    # Add deltas
    delta  = librosa.feature.delta(lfcc)
    delta2 = librosa.feature.delta(lfcc, order=2)
    feats  = np.concatenate([lfcc, delta, delta2], axis=0).T    # (T, 3*n_lfcc)

    return feats.astype(np.float32)


def audio_to_lfcc_chunks(audio_path, chunk_sec=CHUNK_SEC):
    """Load audio and extract LFCC from fixed-length chunks."""
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    chunk_samples = int(chunk_sec * sr)
    chunks_lfcc   = []

    for start in range(0, len(y) - chunk_samples + 1, chunk_samples):
        chunk = y[start:start + chunk_samples]
        lfcc  = extract_lfcc(chunk, sr)
        chunks_lfcc.append(lfcc)

    return chunks_lfcc


# ── Dataset ───────────────────────────────────────────────────────────────────
class SpoofDataset(Dataset):
    """Label 0 = Bona Fide (real), Label 1 = Spoof (synthetic)."""
    def __init__(self, features, labels, seq_len=128):
        self.features = features
        self.labels   = labels
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]

        # Pad or truncate
        if len(feat) < self.seq_len:
            pad  = np.zeros((self.seq_len - len(feat), feat.shape[1]), dtype=np.float32)
            feat = np.vstack([feat, pad])
        else:
            feat = feat[:self.seq_len]

        return torch.from_numpy(feat), torch.tensor(self.labels[idx], dtype=torch.long)


# ── CNN Classifier ────────────────────────────────────────────────────────────
class SpoofCNN(nn.Module):
    """
    Lightweight CNN for spoof detection.
    Input: (B, 1, T, F) — treat LFCC as a 2D image.
    Output: (B, 2) — logits for [bonafide, spoof]
    """
    def __init__(self, input_dim=N_LFCC*3, seq_len=128, num_classes=2):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = self.classifier(x)
        return x


# ── EER Computation ───────────────────────────────────────────────────────────
def compute_eer(labels, scores):
    """
    Compute Equal Error Rate.
    EER = point where FAR (False Accept Rate) == FRR (False Reject Rate).
    Lower is better. Target: < 10%.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr  = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer      = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return eer, thresholds[eer_idx]


# ── Training ──────────────────────────────────────────────────────────────────
def train(real_audio, spoof_audio):
    os.makedirs("models", exist_ok=True)

    print("[Spoof] Extracting LFCC features...")
    real_feats  = audio_to_lfcc_chunks(real_audio)
    spoof_feats = audio_to_lfcc_chunks(spoof_audio)

    # Augment by duplicating if too few samples
    while len(real_feats) < 20:
        real_feats  = real_feats  * 2
    while len(spoof_feats) < 20:
        spoof_feats = spoof_feats * 2

    features = real_feats  + spoof_feats
    labels   = [0]*len(real_feats) + [1]*len(spoof_feats)

    print(f"[Spoof] Bonafide: {len(real_feats)} | Spoof: {len(spoof_feats)}")

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_ds = SpoofDataset(X_train, y_train)
    val_ds   = SpoofDataset(X_val,   y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = SpoofCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_eer  = 1.0
    print(f"[Spoof] Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for feats, lbls in train_dl:
            feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
            logits = model(feats)
            loss   = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for feats, lbls in val_dl:
                feats = feats.to(DEVICE)
                logits = model(feats)
                probs  = torch.softmax(logits, dim=-1)[:, 1]  # P(spoof)
                all_scores.extend(probs.cpu().numpy().tolist())
                all_labels.extend(lbls.numpy().tolist())

        eer, thr = compute_eer(np.array(all_labels), np.array(all_scores))
        avg_loss = total_loss / len(train_dl)
        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | EER: {eer*100:.2f}%")

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  [Saved] EER={best_eer*100:.2f}% -> {MODEL_PATH}")

    print(f"\n[Spoof] Training done! Best EER: {best_eer*100:.2f}%")
    status = "PASS" if best_eer < 0.10 else "FAIL"
    print(f"[Spoof] Passing criterion (<10% EER): {status}")


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(test_audio, label, output_json="data/eer_results.json"):
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    model = SpoofCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    feats_list = audio_to_lfcc_chunks(test_audio)
    if not feats_list:
        print("[Spoof] No chunks extracted!")
        return

    dataset = SpoofDataset(feats_list, [label]*len(feats_list))
    loader  = DataLoader(dataset, batch_size=8)

    scores, preds = [], []
    with torch.no_grad():
        for feats, _ in loader:
            feats  = feats.to(DEVICE)
            logits = model(feats)
            probs  = torch.softmax(logits, dim=-1)
            scores.extend(probs[:, 1].cpu().numpy().tolist())
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    avg_score = np.mean(scores)
    decision  = "SPOOF" if avg_score > 0.5 else "BONAFIDE"
    print(f"\n[Spoof] Audio: {test_audio}")
    print(f"  Spoof probability: {avg_score:.4f}")
    print(f"  Decision: {decision}")

    result = {
        "audio":            test_audio,
        "spoof_score_mean": float(avg_score),
        "decision":         decision,
        "num_chunks":       len(scores)
    }
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results -> {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anti-spoofing classifier")
    parser.add_argument("--mode",  required=True, choices=["train", "eval"])
    parser.add_argument("--real",  default="audio/student_voice_ref.wav")
    parser.add_argument("--spoof", default="outputs/output_LRL_cloned.wav")
    parser.add_argument("--test_audio", default=None)
    parser.add_argument("--label",      type=int, default=1, help="0=real, 1=spoof")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.real, args.spoof)
    else:
        evaluate(args.test_audio or args.spoof, args.label)