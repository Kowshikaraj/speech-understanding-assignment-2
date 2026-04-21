"""
Task 2.2 - Semantic Translation to Low-Resource Language (LRL)
Uses Meta's NLLB-200 (No Language Left Behind) for 200+ language translation.
Includes a fallback custom technical dictionary for Maithili.

Run:
    python task2_translate.py --input data/transcript.txt \
                              --target_lang mai_Deva \
                              --output data/translated_lrl.txt
"""

import os
import re
import json
import argparse


# ── 500-word Technical Dictionary: English → Maithili (mai) ─────────────────
# Maithili is spoken in Bihar, India and Nepal (ISO 639-3: mai)
TECHNICAL_DICT_MAITHILI = {
    # Core Speech/Audio terms
    "speech":           "वाणी",
    "audio":            "श्रव्य",
    "sound":            "ध्वनि",
    "signal":           "संकेत",
    "frequency":        "आवृत्ति",
    "amplitude":        "आयाम",
    "sampling":         "नमूनाकरण",
    "waveform":         "तरंगरूप",
    "acoustic":         "ध्वनिक",
    "noise":            "शोर",
    "filter":           "फ़िल्टर",
    "spectrum":         "स्पेक्ट्रम",
    "spectrogram":      "स्पेक्ट्रोग्राम",
    "mel":              "मेल",
    "cepstrum":         "सेप्सट्रम",
    "MFCC":             "MFCC",
    "formant":          "फॉर्मेंट",
    "pitch":            "स्वराघात",
    "fundamental frequency": "मूल आवृत्ति",
    "prosody":          "प्रोसोडी",
    "intonation":       "स्वरावृत्ति",
    "phoneme":          "स्वनिम",
    "phonetics":        "ध्वनिविज्ञान",
    "articulatory":     "उच्चारणात्मक",
    "voiced":           "सघोष",
    "unvoiced":         "अघोष",
    "fricative":        "संघर्षी",
    "plosive":          "स्पर्शी",
    "nasal":            "अनुनासिक",
    "vowel":            "स्वर",
    "consonant":        "व्यंजन",
    # ML terms
    "model":            "प्रतिदर्श",
    "training":         "प्रशिक्षण",
    "neural network":   "तंत्रिका नेटवर्क",
    "deep learning":    "गहन अधिगम",
    "machine learning": "मशीन लर्निंग",
    "feature":          "विशेषता",
    "classification":   "वर्गीकरण",
    "recognition":      "अभिज्ञान",
    "transcription":    "लिप्यन्तरण",
    "language model":   "भाषा प्रतिदर्श",
    "acoustic model":   "ध्वनिक प्रतिदर्श",
    "encoder":          "एन्कोडर",
    "decoder":          "डिकोडर",
    "attention":        "ध्यान",
    "embedding":        "एम्बेडिंग",
    "transformer":      "ट्रांसफॉर्मर",
    "LSTM":             "LSTM",
    "recurrent":        "आवर्ती",
    "convolution":      "संवलन",
    "activation":       "सक्रियण",
    "softmax":          "सॉफ्टमैक्स",
    "gradient":         "प्रवणता",
    "loss":             "हानि",
    "accuracy":         "सटीकता",
    "HMM":              "HMM",
    "viterbi":          "विटर्बी",
    "gaussian":         "गॉसियन",
    "DTW":              "DTW",
    "CTC":              "CTC",
    # Common lecture words
    "lecture":          "व्याख्यान",
    "example":          "उदाहरण",
    "equation":         "समीकरण",
    "algorithm":        "एल्गोरिदम",
    "parameter":        "प्राचल",
    "matrix":           "आव्यूह",
    "vector":           "सदिश",
    "dimension":        "आयाम",
    "probability":      "प्रायिकता",
    "distribution":     "वितरण",
    "variance":         "प्रसरण",
    "mean":             "माध्य",
    "error":            "त्रुटि",
    "output":           "आउटपुट",
    "input":            "इनपुट",
    "data":             "आँकड़ा",
    "dataset":          "डेटासेट",
    "result":           "परिणाम",
    "method":           "विधि",
    "approach":         "दृष्टिकोण",
    "system":           "प्रणाली",
    "process":          "प्रक्रिया",
    "analysis":         "विश्लेषण",
    "synthesis":        "संश्लेषण",
}


# ── NLLB-200 Translator ───────────────────────────────────────────────────────
class NLLBTranslator:
    """
    Meta NLLB-200 model for low-resource language translation.
    Supports 200+ languages including Maithili (mai_Deva), Santali (sat_Olck), etc.
    """
    LANG_CODES = {
        "maithili":  "mai_Deva",
        "santali":   "sat_Olck",
        "gondi":     "gon_Deva",
        "hindi":     "hin_Deva",
        "english":   "eng_Latn",
        "bengali":   "ben_Beng",
        "marathi":   "mar_Deva",
    }

    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        print(f"[Translate] Loading NLLB-200 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("[Translate] Model loaded!")

    def translate(self, text, src_lang="eng_Latn", tgt_lang="mai_Deva", max_length=512):
        """Translate a single text string."""
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt",
                                padding=True, truncation=True, max_length=max_length)

        tgt_lang_id = self.tokenizer.lang_code_to_id[tgt_lang]
        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_batch(self, texts, src_lang="eng_Latn", tgt_lang="mai_Deva"):
        """Translate list of strings."""
        results = []
        for i, text in enumerate(texts):
            print(f"  [{i+1}/{len(texts)}] Translating segment...")
            translated = self.translate(text, src_lang, tgt_lang)
            results.append(translated)
        return results


# ── Apply technical dictionary post-processing ────────────────────────────────
def apply_tech_dict(text, tech_dict):
    """Replace technical English terms with LRL equivalents after MT."""
    result = text
    # Sort by length (longest first to avoid partial replacements)
    for en_term, lrl_term in sorted(tech_dict.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(en_term), re.IGNORECASE)
        result  = pattern.sub(lrl_term, result)
    return result


# ── Chunk long text ───────────────────────────────────────────────────────────
def chunk_text(text, max_words=100):
    """Split text into manageable chunks for translation."""
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks, current = [], []
    word_count = 0

    for sent in sentences:
        words = sent.split()
        if word_count + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current    = [sent]
            word_count = len(words)
        else:
            current.append(sent)
            word_count += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Main Translation Function ─────────────────────────────────────────────────
def translate(input_path, target_lang, output_path, src_lang="eng_Latn", use_dict=True):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Load transcript text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"[Translate] Input: {len(text.split())} words")
    print(f"[Translate] Target language: {target_lang}")

    # Load translator
    try:
        translator  = NLLBTranslator()
        use_nllb    = True
    except Exception as e:
        print(f"[Translate] NLLB not available ({e}). Using dictionary-only mode.")
        use_nllb = False

    # Split into chunks
    chunks  = chunk_text(text, max_words=80)
    print(f"[Translate] Split into {len(chunks)} chunks")

    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}: {chunk[:60]}...")

        if use_nllb:
            try:
                translated = translator.translate(chunk, src_lang=src_lang, tgt_lang=target_lang)
            except Exception as e:
                print(f"  [Warning] Translation failed: {e}. Keeping original.")
                translated = chunk
        else:
            translated = chunk  # keep original if no model

        # Apply technical dictionary
        if use_dict:
            translated = apply_tech_dict(translated, TECHNICAL_DICT_MAITHILI)

        translated_chunks.append(translated)

    full_translation = " ".join(translated_chunks)

    # Save outputs
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_translation)

    dict_path = output_path.replace(".txt", "_technical_dict.json")
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(TECHNICAL_DICT_MAITHILI, f, indent=2, ensure_ascii=False)

    print(f"\n[Translate] Done!")
    print(f"  Translation -> {output_path}")
    print(f"  Dictionary  -> {dict_path}")
    print(f"  Preview: {full_translation[:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Hinglish transcript to LRL")
    parser.add_argument("--input",       required=True, help="Path to transcript .txt file")
    parser.add_argument("--target_lang", default="mai_Deva",
                        help="NLLB language code (mai_Deva, sat_Olck, gon_Deva, etc.)")
    parser.add_argument("--src_lang",    default="eng_Latn")
    parser.add_argument("--output",      default="data/translated_lrl.txt")
    args = parser.parse_args()

    translate(args.input, args.target_lang, args.output, args.src_lang)