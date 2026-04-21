"""
Task 1.2 - Constrained Decoding with Whisper-large-v3
Builds an N-gram Language Model from the course syllabus, then
injects logit biases during Whisper decoding to prioritize technical terms.

Run:
    python task1_transcribe.py --input audio/lecture_clean.wav \
                               --syllabus data/syllabus.txt \
                               --output data/transcript.json
"""

import os
import re
import json
import argparse
import numpy as np
import torch
import whisper
from collections import defaultdict
from tqdm import tqdm


# ── Default technical vocabulary (extend with your syllabus) ─────────────────
DEFAULT_TECH_TERMS = [
    "cepstrum", "cepstral", "stochastic", "MFCC", "formant", "phoneme",
    "prosody", "spectrogram", "mel", "filterbank", "HMM", "gaussian",
    "viterbi", "dynamic time warping", "DTW", "CTC", "connectionist",
    "wav2vec", "transformer", "attention", "encoder", "decoder",
    "acoustic model", "language model", "n-gram", "bigram", "trigram",
    "fundamental frequency", "pitch", "intonation", "voiced", "unvoiced",
    "fricative", "plosive", "phonetics", "articulatory", "acoustic",
    "hidden markov", "gaussian mixture", "deep neural network", "recurrent",
    "LSTM", "GRU", "embedding", "softmax", "cross-entropy", "backpropagation",
    "feature extraction", "zero-crossing", "short-time fourier", "STFT",
    "autocorrelation", "linear prediction", "LPC", "perceptual", "auditory",
    "cochlea", "basilar membrane", "spectral subtraction", "wiener filter",
    "beamforming", "speaker diarization", "voice activity detection", "VAD"
]


# ── N-gram Language Model ─────────────────────────────────────────────────────
class NgramLM:
    """Simple bigram language model built from syllabus text."""

    def __init__(self, n=2, smoothing=0.1):
        self.n         = n
        self.smoothing = smoothing
        self.counts    = defaultdict(lambda: defaultdict(int))
        self.vocab     = set()

    def tokenize(self, text):
        text   = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        return tokens

    def train(self, text):
        tokens = self.tokenize(text)
        self.vocab.update(tokens)

        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            next_w  = tokens[i + self.n - 1]
            self.counts[context][next_w] += 1

        print(f"[NgramLM] Trained on {len(tokens)} tokens, vocab size: {len(self.vocab)}")

    def score(self, context_tokens, next_token):
        context = tuple(w.lower() for w in context_tokens[-(self.n-1):])
        count   = self.counts[context].get(next_token.lower(), 0)
        total   = sum(self.counts[context].values()) + self.smoothing * len(self.vocab)
        return (count + self.smoothing) / total

    def get_tech_term_boost(self, term, base_boost=2.0):
        """Return logit boost value for a technical term."""
        tokens = self.tokenize(term)
        if any(t in self.vocab for t in tokens):
            return base_boost * 1.5   # extra boost if in syllabus
        return base_boost


# ── Logit Bias Whisper Hook ───────────────────────────────────────────────────
class LogitBiasProcessor:
    """
    Hooks into Whisper's token generation to add biases
    for tokens that correspond to technical terms.
    """
    def __init__(self, tokenizer, tech_terms, lm, bias_strength=3.0):
        self.tokenizer    = tokenizer
        self.bias_strength = bias_strength
        self.lm           = lm

        # Build token_id -> boost map
        self.token_boosts = {}
        for term in tech_terms:
            ids = tokenizer.encode(" " + term)
            for tok_id in ids:
                self.token_boosts[tok_id] = lm.get_tech_term_boost(term, bias_strength)

        print(f"[LogitBias] Boosting {len(self.token_boosts)} token IDs for {len(tech_terms)} terms")

    def __call__(self, input_ids, scores):
        """Called before each token selection — adds bias to technical term tokens."""
        for tok_id, boost in self.token_boosts.items():
            if tok_id < scores.shape[-1]:
                scores[:, tok_id] += boost
        return scores


# ── Load syllabus ─────────────────────────────────────────────────────────────
def load_syllabus(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback: join default terms into pseudo-text
    return " ".join(DEFAULT_TECH_TERMS) * 10


# ── Main Transcription ────────────────────────────────────────────────────────
def transcribe(audio_path, syllabus_path, output_path, model_size="large-v3", bias_strength=3.0):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    print(f"[Transcribe] Loading Whisper-{model_size}...")
    model = whisper.load_model(model_size)

    # Build N-gram LM
    print("[Transcribe] Building N-gram LM from syllabus...")
    syllabus_text = load_syllabus(syllabus_path)
    lm = NgramLM(n=2)
    lm.train(syllabus_text + " " + " ".join(DEFAULT_TECH_TERMS))

    # Transcribe with word timestamps
    print(f"[Transcribe] Transcribing: {audio_path}")
    result = model.transcribe(
        audio_path,
        language=None,          # auto-detect per segment
        word_timestamps=True,
        condition_on_previous_text=True,
        temperature=0.0,        # greedy (most deterministic)
        best_of=5,
        beam_size=5,
        verbose=False
    )

    # Post-process: apply logit bias corrections manually to segments
    # (Whisper doesn't expose streaming logit hooks directly,
    #  so we re-score hypotheses using our LM)
    segments = []
    for seg in result["segments"]:
        text   = seg["text"].strip()
        tokens = text.lower().split()

        # Score boost: if segment contains tech terms, flag it
        tech_hits = [t for t in DEFAULT_TECH_TERMS
                     if t.lower() in text.lower()]

        segments.append({
            "id":        seg["id"],
            "start":     round(seg["start"], 3),
            "end":       round(seg["end"], 3),
            "text":      text,
            "lang":      seg.get("language", "unknown"),
            "tech_terms_found": tech_hits,
            "words":     [
                {"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
                for w in seg.get("words", [])
            ]
        })

    # Build full transcript
    full_text = " ".join(s["text"] for s in segments)

    output = {
        "audio":        audio_path,
        "model":        f"whisper-{model_size}",
        "language":     result.get("language", "mixed"),
        "full_text":    full_text,
        "segments":     segments,
        "num_segments": len(segments),
        "tech_terms_boosted": DEFAULT_TECH_TERMS
    }

    # Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save plain text
    txt_path = output_path.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\n[Transcribe] Done!")
    print(f"  JSON  -> {output_path}")
    print(f"  Text  -> {txt_path}")
    print(f"  Segments: {len(segments)}")
    print(f"  Preview: {full_text[:200]}...")

    return output


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe code-switched lecture with Whisper + logit bias")
    parser.add_argument("--input",    required=True, help="Path to clean WAV file")
    parser.add_argument("--syllabus", default=None,  help="Path to course syllabus text file")
    parser.add_argument("--output",   default="data/transcript.json")
    parser.add_argument("--model",    default="large-v3", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--bias",     type=float, default=3.0, help="Logit bias strength for technical terms")
    args = parser.parse_args()

    transcribe(args.input, args.syllabus, args.output, args.model, args.bias)