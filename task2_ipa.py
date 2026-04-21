"""
Task 2.1 - Hinglish → IPA Unified Representation
Converts code-switched Hindi-English transcript to IPA phonemes.
Uses epitran/g2p-en for English, custom mapping for Hindi.

Run:
    python task2_ipa.py --transcript data/transcript.json \
                        --lid data/lid_labels.json \
                        --output data/ipa_output.txt
"""

import os
import re
import json
import argparse


# ── Hindi Devanagari → IPA mapping ───────────────────────────────────────────
# Source: standard Hindi phonology
HINDI_IPA_MAP = {
    # Vowels
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː", "उ": "ʊ", "ऊ": "uː",
    "ए": "eː", "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː", "ऋ": "rɪ",
    # Matras (vowel signs)
    "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ", "ू": "uː",
    "े": "eː", "ै": "ɛː", "ो": "oː", "ौ": "ɔː",
    # Consonants
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʰ", "ङ": "ŋ",
    "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "dʒʰ", "ञ": "ɲ",
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʰ", "ण": "ɳ",
    "त": "t̪", "थ": "t̪ʰ", "द": "d̪", "ध": "d̪ʰ", "न": "n",
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʰ", "म": "m",
    "य": "j", "र": "r", "ल": "l", "व": "ʋ",
    "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "ɦ",
    "क्ष": "kʂ", "त्र": "t̪r", "ज्ञ": "dʒɲ",
    # Nasalization & misc
    "ं": "̃", "ः": "h", "्": "",  # halant (vowel killer)
    "।": " | ",  # danda (sentence end)
}

# Common Hinglish romanized words → IPA
HINGLISH_ROMAN_MAP = {
    "aur":    "ɔːr",    "hai":    "ɦɛː",   "hain":   "ɦɛ̃ː",
    "tha":    "t̪ʰaː",  "thi":    "t̪ʰiː",  "the":    "t̪ʰeː",
    "kya":    "kjaː",   "kyun":   "kjũː",  "kyunki": "kjũːki",
    "nahi":   "nɐɦĩː",  "nahin":  "nɐɦĩː", "hum":    "ɦʊm",
    "tum":    "t̪ʊm",   "aap":    "aːp",   "main":   "mɛ̃ː",
    "mein":   "mẽː",   "se":     "seː",   "ko":     "koː",
    "ka":     "kaː",   "ke":     "keː",   "ki":     "kiː",
    "ek":     "eːk",   "do":     "d̪oː",   "teen":   "t̪iːn",
    "bhi":    "bʰiː",  "lekin":  "leːkɪn", "par":    "pɐr",
    "matlab": "mɐt̪lɐb", "yani":  "jaːniː", "isliye": "ɪslɪjeː",
    "toh":    "t̪oː",   "woh":    "ʋoː",   "yeh":    "jeː",
}


# ── English G2P ───────────────────────────────────────────────────────────────
def english_to_ipa(text):
    """Convert English text to IPA using epitran or g2p-en fallback."""
    try:
        import epitran
        epi = epitran.Epitran("eng-Latn")
        return epi.transliterate(text)
    except Exception:
        pass

    try:
        from g2p_en import G2p
        g2p = G2p()
        phonemes = g2p(text)
        # Convert ARPAbet to IPA
        ipa = arpabet_to_ipa(" ".join(phonemes))
        return ipa
    except Exception:
        pass

    # Simple fallback: return text as-is with markers
    return f"/{text}/"


def arpabet_to_ipa(arpabet_str):
    """Convert ARPAbet phonemes to IPA."""
    ARPABET_IPA = {
        "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ",
        "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
        "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "ɡ",
        "HH": "h", "IH": "ɪ", "IY": "iː", "JH": "dʒ", "K": "k",
        "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ",
        "OY": "ɔɪ", "P": "p", "R": "r", "S": "s", "SH": "ʃ",
        "T": "t", "TH": "θ", "UH": "ʊ", "UW": "uː", "V": "v",
        "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
    }
    result = []
    for token in arpabet_str.split():
        base = re.sub(r"\d", "", token)  # strip stress markers
        result.append(ARPABET_IPA.get(base, token.lower()))
    return "".join(result)


# ── Hindi G2P ─────────────────────────────────────────────────────────────────
def hindi_devanagari_to_ipa(text):
    """Convert Hindi Devanagari text to IPA using character-level mapping."""
    result = []
    i = 0
    while i < len(text):
        # Check 2-char clusters first
        if i + 1 < len(text) and text[i:i+2] in HINDI_IPA_MAP:
            result.append(HINDI_IPA_MAP[text[i:i+2]])
            i += 2
        elif text[i] in HINDI_IPA_MAP:
            ipa_char = HINDI_IPA_MAP[text[i]]
            # Add schwa if consonant not followed by vowel sign or halant
            if ipa_char and text[i] not in "अआइईउऊएऐओऔऋ":
                next_char = text[i+1] if i+1 < len(text) else ""
                is_vowel_sign = next_char in "ािीुूेैोौृ"
                is_halant     = next_char == "्"
                if not is_vowel_sign and not is_halant and ipa_char not in "̃ːh":
                    result.append(ipa_char + "ə")
                else:
                    result.append(ipa_char)
            else:
                result.append(ipa_char)
            i += 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def hindi_roman_to_ipa(word):
    """Convert romanized Hindi word to IPA."""
    w = word.lower().strip(".,!?;:")
    if w in HINGLISH_ROMAN_MAP:
        return HINGLISH_ROMAN_MAP[w]
    # Character-level approximation for unknown roman Hindi
    roman_approx = {
        "aa": "aː", "ee": "iː", "oo": "uː", "ai": "ɛː", "au": "ɔː",
        "sh": "ʃ", "kh": "kʰ", "gh": "ɡʰ", "ch": "tʃ", "jh": "dʒʰ",
        "th": "t̪ʰ", "dh": "d̪ʰ", "ph": "pʰ", "bh": "bʰ",
        "ng": "ŋ", "ny": "ɲ", "ry": "rj",
    }
    result = w
    for combo, ipa in roman_approx.items():
        result = result.replace(combo, ipa)
    return result


# ── Determine language of each word ──────────────────────────────────────────
def get_word_lang(word, lid_data, word_start_time):
    """Look up LID labels to find the language at a given timestamp."""
    if lid_data is None:
        return "en"   # Default to English if no LID data

    for frame in lid_data.get("frames", []):
        if frame["start"] <= word_start_time < frame["end"]:
            return frame["lang"]
    return "en"


def is_devanagari(text):
    """Check if text contains Devanagari Unicode characters."""
    return any("\u0900" <= ch <= "\u097F" for ch in text)


# ── Main Conversion ───────────────────────────────────────────────────────────
def convert_to_ipa(transcript_path, lid_path, output_path):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Load transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Load LID data
    lid_data = None
    if lid_path and os.path.exists(lid_path):
        with open(lid_path, "r", encoding="utf-8") as f:
            lid_data = json.load(f)
        print(f"[IPA] Loaded LID labels from {lid_path}")

    print(f"[IPA] Converting {len(transcript['segments'])} segments to IPA...")

    ipa_segments = []
    full_ipa     = []

    for seg in transcript["segments"]:
        seg_ipa_parts = []
        words = seg.get("words", [])

        if not words:
            # No word-level timestamps, process whole segment
            text = seg["text"].strip()
            lang = seg.get("lang", "en")
            if is_devanagari(text):
                ipa = hindi_devanagari_to_ipa(text)
            elif lang == "hi":
                ipa_words = [hindi_roman_to_ipa(w) for w in text.split()]
                ipa = " ".join(ipa_words)
            else:
                ipa = english_to_ipa(text)
            seg_ipa_parts.append(ipa)
        else:
            for word_info in words:
                word       = word_info["word"].strip()
                word_start = word_info.get("start", 0.0)
                lang       = get_word_lang(word, lid_data, word_start)

                if not word:
                    continue

                if is_devanagari(word):
                    ipa = hindi_devanagari_to_ipa(word)
                elif lang == "hi":
                    ipa = hindi_roman_to_ipa(word)
                else:
                    ipa = english_to_ipa(word)

                seg_ipa_parts.append(ipa)

        seg_ipa = " ".join(seg_ipa_parts)
        ipa_segments.append({
            "id":    seg["id"],
            "start": seg["start"],
            "end":   seg["end"],
            "text":  seg["text"],
            "ipa":   seg_ipa
        })
        full_ipa.append(seg_ipa)

    full_ipa_str = " | ".join(full_ipa)  # | separates segments

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_ipa_str)

    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"ipa_segments": ipa_segments, "full_ipa": full_ipa_str}, f,
                  indent=2, ensure_ascii=False)

    print(f"[IPA] Saved to {output_path}")
    print(f"[IPA] Preview: {full_ipa_str[:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hinglish transcript to IPA")
    parser.add_argument("--transcript", required=True, help="Path to transcript JSON")
    parser.add_argument("--lid",        default=None,  help="Path to LID labels JSON")
    parser.add_argument("--output",     default="data/ipa_output.txt")
    args = parser.parse_args()

    convert_to_ipa(args.transcript, args.lid, args.output)