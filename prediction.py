import argparse
import os
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import pandas as pd
import numpy as np
import re
import functools

TEST_ROOT = "data/test-random"
OUTPUT_ROOT = "output"
LEXICON_PATH = "data/train/lexicon.txt"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

TAIWANESE_ALLOWED = r"a-z\s"

CLEAN_RE = re.compile(f"[{TAIWANESE_ALLOWED}]+")
SPACE_RE = re.compile(r"\s+")

def clean_taiwanese(text: str) -> str:
    """Strip non-Taiwanese characters (e.g., Burmese, Sinhala)."""
    fragments = CLEAN_RE.findall(text)
    cleaned = " ".join(fragments)
    cleaned = SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()

def load_lexicon(path):
    """Lexicon format: word tag canonical_form"""
    lex = {}
    if not os.path.exists(path):
        print(f"Warning: lexicon file not found: {path}")
        return lex

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                surface, _, canonical = parts
                lex[surface.lower()] = surface.lower()

    print(f"Loaded lexicon with {len(lex)} entries")
    return lex

@functools.lru_cache(maxsize=50000)
def edit_distance(a, b):
    """DP edit distance with caching."""
    dp = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    return dp[-1][-1]

def closest_lexicon_word(word, lex_keys):
    """Return nearest lexicon word by edit distance."""
    return min(lex_keys, key=lambda w: edit_distance(word, w))

def correct_taiwanese_sentence(sentence, lex):
    """Clean → tokenize → lexicon correction → canonical sentence."""
    cleaned = clean_taiwanese(sentence)
    if not cleaned:
        return ""

    words = cleaned.split()
    corrected = []

    if not lex:
        # If lexicon missing, still return cleaned text
        return cleaned

    keys = list(lex.keys())

    for w in words:
        w_low = w.lower()
        if w_low in lex:
            corrected.append(lex[w_low])
        else:
            nearest = closest_lexicon_word(w_low, keys)
            corrected.append(lex[nearest])

    return " ".join(corrected)


def load_model(model_dir):
    print(f"Loading model from: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

def transcribe_audio(processor, model, audio_path, lexicon, max_length_sec=30):
    audio, sr = librosa.load(audio_path, sr=16000)

    max_samples = max_length_sec * sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(model.device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_length=225,
            num_beams=1
        )

    sentence = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    sentence = correct_taiwanese_sentence(sentence, lexicon)

    return sentence

def transcribe_directory(processor, model, lexicon):
    rows = []
    
    for file in sorted(os.listdir(TEST_ROOT)):
        if not file.lower().endswith(".wav"):
            continue

        full_path = os.path.join(TEST_ROOT, file)
        audio_id = os.path.splitext(file)[0]

        print(f"Processing: {file} ...")

        sentence = transcribe_audio(processor, model, full_path, lexicon)
        rows.append({"id": audio_id, "sentence": sentence})

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory where the trained model is saved.")
    
    args = parser.parse_args()
    lexicon = load_lexicon(LEXICON_PATH)
    processor, model = load_model(args.model_dir)

    model_choice = f"{args.model_dir.split('/')[-2]}_{args.model_dir.split('/')[-1]}"
    df = transcribe_directory(processor, model, lexicon)

    timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')
    output_csv = os.path.join(OUTPUT_ROOT, f"{model_choice}_{timestamp}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")
