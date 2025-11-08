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

@functools.lru_cache(maxsize=50000)
def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        prev = dp[:]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            dp[j] = min(
                prev[j] + 1,
                dp[j - 1] + 1,
                prev[j - 1] + cost
            )
    return dp[-1]

class prediction:
    def __init__(self, model_dir, use_lexicon=True):
        self.model_dir = model_dir
        self.use_lexicon = use_lexicon
        self.processor = None
        self.model = None
        self.valid_words = None

    def get_valid_words(self):
        # extract words from training CSV file
        df = pd.read_csv("data/train/train-toneless.csv")

        words_from_csv = []

        for sentence in df["text"].astype(str):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            words = sentence.split()
            words_from_csv.extend(words)
        words_from_csv = list(set(words_from_csv))
        print(f"Extracted {len(words_from_csv)} unique words from CSV.")

        # extracted words from lexicon.txt
        words_from_lexicon = []
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    words_from_lexicon.append(parts[0].lower())
                    words_from_lexicon.append(parts[1].lower())
                    words_from_lexicon.append(parts[2].lower())
        words_from_lexicon = list(set(words_from_lexicon))
        words_from_lexicon.remove('inull')

        print(f"Extracted {len(words_from_lexicon)} unique words from lexicon.")

        words = list(set(words_from_csv + words_from_lexicon))
        print(f"Unique words count: {len(words)}")
        return words

    def correct_taiwanese_sentence(self, sentence):
        if not self.use_lexicon:
            return sentence

        # normalize
        sentence = sentence.lower().strip()
        if not sentence:
            return ""

        tokens = sentence.split()
        corrected = []

        for w in tokens:
            # keep exact match
            if w in self.valid_words:
                corrected.append(w)
                continue

            # fallback: find closest word
            nearest = min(
                self.valid_words,
                key=lambda vw: levenshtein(w, vw)
            )

            corrected.append(nearest)

        return " ".join(corrected)

    def load_model(self, model_dir):
        print(f"Loading model from: {model_dir}")
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir)
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def transcribe_audio(self, audio_path, max_length_sec=30):
        audio, sr = librosa.load(audio_path, sr=16000)

        max_samples = max_length_sec * sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.model.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                max_length=225,
                num_beams=3,
            )

        sentence = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        sentence = self.correct_taiwanese_sentence(sentence)

        return sentence

    def transcribe_directory(self):
        rows = []
        
        for file in sorted(os.listdir(TEST_ROOT)):
            if not file.lower().endswith(".wav"):
                continue

            full_path = os.path.join(TEST_ROOT, file)
            audio_id = os.path.splitext(file)[0]

            print(f"Processing: {file} ...")

            sentence = self.transcribe_audio(full_path)
            rows.append({"id": audio_id, "sentence": sentence})

        df = pd.DataFrame(rows)
        return df
    
    def run(self):
        self.valid_words = self.get_valid_words()
        self.load_model(self.model_dir)
        model_choice = f"{self.model_dir.split('/')[-2]}_{self.model_dir.split('/')[-1]}"
        df = self.transcribe_directory()

        timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')
        output_csv = os.path.join(OUTPUT_ROOT, f"{model_choice}_{timestamp}.csv")
        df.to_csv(output_csv, index=False)
        print(f"Saved predictions to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory where the trained model is saved.")
    parser.add_argument("--use_lexicon", action="store_true", help="Whether to use lexicon for correction. Default is True if lexicon file exists.")
    args = parser.parse_args()

    
    predictor = prediction(model_dir=args.model_dir, use_lexicon=args.use_lexicon)
    predictor.run()
