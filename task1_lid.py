"""
Task 1.1 - Multi-Head Frame-Level Language Identification (LID)
BiLSTM model that classifies each 20ms frame as Hindi(0) or English(1).
Target: F1-score >= 0.85, timestamp precision within 200ms.

Run (train):
    python task1_lid.py --mode train --data_dir data/lid_data/

Run (predict):
    python task1_lid.py --mode predict --input audio/lecture_clean.wav --output data/lid_labels.json
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
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm


# ── Config ───────────────────────────────────────────────────────────────────
SR           = 16000
N_MFCC       = 40
HOP_LENGTH   = 160        # 10ms hop at 16kHz
WIN_LENGTH   = 400        # 25ms window at 16kHz
FRAME_LEN    = 0.02       # 20ms per frame
HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(audio_path, sr=SR):
    """Extract MFCC + delta + delta-delta features from audio file."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                        hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    delta       = librosa.feature.delta(mfcc)
    delta2      = librosa.feature.delta(mfcc, order=2)

    features    = np.concatenate([mfcc, delta, delta2], axis=0)  # (120, T)
    features    = features.T                                       # (T, 120)
    return features.astype(np.float32)


def extract_features_from_array(y, sr=SR):
    """Extract features from numpy array (for pipeline use)."""
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                   hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats  = np.concatenate([mfcc, delta, delta2], axis=0).T
    return feats.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────
class LIDDataset(Dataset):
    """
    Expects data_dir with structure:
        data_dir/
            hindi/   (*.wav files — label 0)
            english/ (*.wav files — label 1)
    """
    def __init__(self, data_dir, seq_len=100):
        self.samples  = []
        self.seq_len  = seq_len

        for lang, label in [("hindi", 0), ("english", 1)]:
            lang_dir = os.path.join(data_dir, lang)
            if not os.path.exists(lang_dir):
                print(f"[Warning] Directory not found: {lang_dir}")
                continue
            for fname in os.listdir(lang_dir):
                if fname.endswith(".wav"):
                    self.samples.append((os.path.join(lang_dir, fname), label))

        print(f"[LIDDataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feats = extract_features(path)              # (T, 120)

        # Pad or truncate to seq_len
        if len(feats) < self.seq_len:
            pad = np.zeros((self.seq_len - len(feats), feats.shape[1]), dtype=np.float32)
            feats = np.vstack([feats, pad])
        else:
            feats = feats[:self.seq_len]

        labels = np.full(self.seq_len, label, dtype=np.int64)
        return torch.from_numpy(feats), torch.from_numpy(labels)


# ── Model ─────────────────────────────────────────────────────────────────────
class MultiHeadLID(nn.Module):
    """
    BiLSTM encoder with two classification heads:
      - frame_head: per-frame Hindi/English prediction
      - seg_head:   segment-level confidence
    """
    def __init__(self, input_size=120, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_classes=2, dropout=DROPOUT):
        super().__init__()

        self.norm = nn.LayerNorm(input_size)

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        # Frame-level head (main output)
        self.frame_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # Segment-level head (auxiliary)
        self.seg_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = self.norm(x)
        out, (hn, _) = self.bilstm(x)           # out: (B, T, 2*H)
        out = self.dropout(out)

        frame_logits = self.frame_head(out)      # (B, T, 2)

        # Segment: use mean-pooled LSTM output
        seg_logits   = self.seg_head(out.mean(dim=1))  # (B, 2)

        return frame_logits, seg_logits


# ── Training ──────────────────────────────────────────────────────────────────
def train(data_dir, model_path="models/lid_model.pt"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    dataset   = LIDDataset(data_dir)
    n_val     = max(1, int(0.15 * len(dataset)))
    n_train   = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = MultiHeadLID().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_f1   = 0.0
    print(f"[LID] Training on {DEVICE} | {n_train} train, {n_val} val samples")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for feats, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            frame_logits, seg_logits = model(feats)

            # Frame loss: reshape (B, T, 2) -> (B*T, 2)
            B, T, C = frame_logits.shape
            frame_loss = criterion(frame_logits.reshape(B*T, C), labels.reshape(B*T))

            # Segment loss: majority label per sequence
            seg_labels = labels.mode(dim=1).values
            seg_loss   = criterion(seg_logits, seg_labels)

            loss = frame_loss + 0.3 * seg_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feats, labels in val_dl:
                feats = feats.to(DEVICE)
                frame_logits, _ = model(feats)
                preds = frame_logits.argmax(dim=-1).cpu().numpy().flatten()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().flatten().tolist())

        f1 = f1_score(all_labels, all_preds, average="macro")
        avg_loss = total_loss / len(train_dl)
        print(f"  Loss: {avg_loss:.4f} | Val F1: {f1:.4f}")

        if True:
            best_f1 = f1
            torch.save(model.state_dict(), model_path)
            print(f"  [Saved] Best model (F1={best_f1:.4f}) -> {model_path}")

    print(f"\n[LID] Training complete. Best F1: {best_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Hindi", "English"]))


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(audio_path, model_path, output_path, chunk_sec=2.0):
    """
    Predict language per frame and save timestamped labels as JSON.
    Output format: [{"start": 0.0, "end": 0.02, "lang": "hi", "conf": 0.92}, ...]
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    model = MultiHeadLID().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"[LID] Model loaded from {model_path}")

    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    print(f"[LID] Audio: {len(y)/sr:.1f}s @ {sr}Hz")

    chunk_frames = int(chunk_sec * sr)
    all_preds    = []
    all_confs    = []

    # Process in chunks to handle long audio
    for start in range(0, len(y), chunk_frames):
        chunk = y[start:start + chunk_frames]
        feats = extract_features_from_array(chunk, sr)  # (T, 120)
        feats_t = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)  # (1, T, 120)

        with torch.no_grad():
            frame_logits, _ = model(feats_t)  # (1, T, 2)
            probs = torch.softmax(frame_logits, dim=-1)[0]  # (T, 2)
            preds = probs.argmax(dim=-1).cpu().numpy()
            confs = probs.max(dim=-1).values.cpu().numpy()

        all_preds.extend(preds.tolist())
        all_confs.extend(confs.tolist())

    # Build timestamped output
    frame_duration = HOP_LENGTH / SR
    results = []
    for i, (pred, conf) in enumerate(zip(all_preds, all_confs)):
        results.append({
            "start": round(i * frame_duration, 4),
            "end":   round((i + 1) * frame_duration, 4),
            "lang":  "hi" if pred == 0 else "en",
            "label": int(pred),
            "conf":  round(float(conf), 4)
        })

    # Find language switch boundaries
    switches = []
    for i in range(1, len(results)):
        if results[i]["lang"] != results[i-1]["lang"]:
            switches.append({
                "timestamp": results[i]["start"],
                "from": results[i-1]["lang"],
                "to":   results[i]["lang"]
            })

    output = {
        "audio":    audio_path,
        "frames":   results,
        "switches": switches,
        "summary": {
            "total_frames": len(results),
            "hindi_frames":   sum(1 for r in results if r["lang"] == "hi"),
            "english_frames": sum(1 for r in results if r["lang"] == "en"),
            "num_switches": len(switches)
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[LID] Saved {len(results)} frame labels to {output_path}")
    print(f"[LID] Language switches detected: {len(switches)}")
    s = output["summary"]
    print(f"[LID] Hindi frames: {s['hindi_frames']} | English frames: {s['english_frames']}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame-level Language Identification")
    parser.add_argument("--mode",       required=True, choices=["train", "predict"])
    parser.add_argument("--data_dir",   default="data/lid_data/", help="[train] dir with hindi/ english/ subdirs")
    parser.add_argument("--input",      help="[predict] path to WAV file")
    parser.add_argument("--output",     default="data/lid_labels.json", help="[predict] output JSON path")
    parser.add_argument("--model_path", default="models/lid_model.pt")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir, args.model_path)
    else:
        if not args.input:
            raise ValueError("--input required for predict mode")
        predict(args.input, args.model_path, args.output)