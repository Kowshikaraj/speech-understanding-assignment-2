# Speech Understanding Assignment

## Overview

This project implements a complete speech processing pipeline including speech enhancement, transcription, language identification, phonetic conversion, translation, voice cloning, and security analysis.

The system takes raw lecture audio as input and produces processed outputs such as text, phonetic representation, translated text, and synthesized speech.

## Tasks Implemented

### Task 1: Speech Processing

* Denoising (Spectral Subtraction)
* Speech-to-Text (Whisper)
* Language Identification (BiLSTM)

### Task 2: Text Processing

* IPA Conversion (Phonetic transcription)
* Translation to Low-Resource Language

### Task 3: Voice Cloning

* Speaker Embedding (MFCC fallback)
* Text-to-Speech (Fallback synthesis)
* Prosody Alignment (DTW)

### Task 4: Security and Robustness

* Spoof Detection (Real vs Fake audio)
* FGSM Adversarial Attack

## Project Structure

```bash
speech_project/
│
├── audio/
├── data/
├── models/
├── outputs/
│
├── task1_denoise.py
├── task1_transcribe.py
├── task1_lid.py
├── task2_ipa.py
├── task2_translate.py
├── task3_embed.py
├── task3_tts.py
├── task3_dtw.py
├── task4_spoof.py
├── task4_fgsm.py
│
├── .gitignore
└── README.md
```

## How to Run

Denoising

```bash
python task1_denoise.py --input audio/lecture.wav --output audio/lecture_clean.wav
```

Transcription

```bash
python task1_transcribe.py --input audio/lecture_clean.wav --syllabus data/syllabus.txt --output data/transcript.json
```

Language Identification

```bash
python task1_lid.py --mode predict --input audio/lecture_clean.wav --output data/lid_labels.json
```

IPA Conversion

```bash
python task2_ipa.py --transcript data/transcript.json --lid data/lid_labels.json --output data/ipa_output.txt
```

Translation

```bash
python task2_translate.py --input data/transcript.txt --target_lang mai_Deva --output data/translated_lrl.txt
```

Voice Embedding

```bash
python task3_embed.py --voice audio/student_voice_ref.wav --output models/speaker_embedding.pt
```

Text-to-Speech

```bash
python task3_tts.py --text data/translated_lrl.txt --speaker_ref audio/student_voice_ref.wav --output outputs/synth_flat.wav
```

DTW Alignment

```bash
python task3_dtw.py --source audio/lecture_clean.wav --target outputs/synth_flat.wav --output outputs/output_LRL_cloned.wav
```

FGSM Attack

```bash
python task4_fgsm.py --input audio/lecture_clean.wav --output outputs/adversarial_audio.wav
```

## Outputs

* lecture_clean.wav
* transcript.json
* ipa_output.txt
* translated_lrl.txt
* synth_flat.wav
* output_LRL_cloned.wav
* adversarial_audio.wav

## Notes

* Large files such as audio, model weights, and JSON outputs are excluded using .gitignore
* Some modules use fallback implementations due to Python 3.12 compatibility issues
* Text-to-Speech output is simplified due to dependency limitations

## Technologies Used

* Python
* PyTorch
* Librosa
* Whisper
* NumPy
* SoundFile

## Conclusion

This project demonstrates a complete pipeline for speech understanding, text processing, voice synthesis, and robustness testing. It highlights practical challenges such as dependency management, large file handling, and model evaluation.

## Author

Banoth.Sri Kowshika Raj
