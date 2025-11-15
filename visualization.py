import os
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import pandas as pd
import numpy as np
import re
import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
import argparse

from collections import Counter
from jiwer import wer, cer
from jiwer import process_words

substitutions = Counter()
insertions    = Counter()
deletions     = Counter()
correct       = Counter()

DATA = "data/train/train/hybrid_TPGBIR"
DATA_CSV = os.path.join(DATA, "metadata.csv")
OUTPUT_ROOT = "visualization"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

class visualization:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.processor = None
        self.model = None
        self.valid_words = None

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
        return sentence

    def transcribe_directory(self):
        rows = []

        wav_files = sorted([f for f in os.listdir(DATA) if f.lower().endswith(".wav")])

        # take randomly for faster processing during testing
        np.random.seed(42)
        wav_files = list(np.random.choice(wav_files, size=min(5000, len(wav_files)), replace=False))
        
        for file in tqdm(wav_files, desc="Processing audio", unit="file"):
            full_path = os.path.join(DATA, file)
            audio_id = os.path.splitext(file)[0]

            sentence = self.transcribe_audio(full_path)     
            rows.append({"id": f"{audio_id}.wav", "sentence": sentence})

        df = pd.DataFrame(rows)
        return df
    
    def generate_confusion_matrix(self, predicted_df, output_path):
        """Generate character-level confusion matrix visualization"""
        # Load ground truth
        gt_df = pd.read_csv(DATA_CSV)
        # change column name to match
        gt_df = gt_df.rename(columns={"file_name": "id", "transcription": "sentence"})

        # Merge predicted and ground truth on id
        merged = predicted_df.merge(gt_df, on='id', suffixes=('_pred', '_gt'))
        
        merged.to_csv("merged_predictions.csv", index=False)

        def align_words(reference, hypothesis):
            """Return aligned word pairs using Levenshtein backtracking."""
            m, n = len(reference), len(hypothesis)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            for i in range(1, m + 1):
                ref_word = reference[i - 1]
                for j in range(1, n + 1):
                    hyp_word = hypothesis[j - 1]
                    cost = 0 if ref_word == hyp_word else 1
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,         # deletion
                        dp[i][j - 1] + 1,         # insertion
                        dp[i - 1][j - 1] + cost,  # substitution or match
                    )

            aligned = []
            i, j = m, n
            while i > 0 or j > 0:
                if i > 0 and j > 0:
                    cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
                    if dp[i][j] == dp[i - 1][j - 1] + cost:
                        aligned.append((reference[i - 1], hypothesis[j - 1]))
                        i -= 1
                        j -= 1
                        continue
                if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                    aligned.append((reference[i - 1], "<DEL>"))
                    i -= 1
                else:
                    aligned.append(("<INS>", hypothesis[j - 1]))
                    j -= 1

            aligned.reverse()
            return aligned

        y_true = []
        y_pred = []
        csv_out = []
        for _, row in merged.iterrows():
            gt_words = row['sentence_gt'].strip().split()
            pred_words = row['sentence_pred'].strip().split()
            if not gt_words and not pred_words:
                continue
            aligned_pairs = align_words(gt_words, pred_words)
            for gt_word, pred_word in aligned_pairs:
                y_true.append(gt_word)
                y_pred.append(pred_word)
            a, b = zip(*aligned_pairs)
            csv_out.append({
                "id": row['id'],
                "ground_truth": a,
                "prediction": b,
            })
        aligned_df = pd.DataFrame(csv_out)

        with open(os.path.join(OUTPUT_ROOT,"word_alignments.txt"), "w") as f:
            for _, row in aligned_df.iterrows():
                f.write("\n")
                f.write(f"File name: {row['id']}\n")
                f.write("True: ")
                f.write(" ".join(row['ground_truth']) + "\n")
                f.write("Pred: ")
                f.write(" ".join(row['prediction']) + "\n")
                f.write("\n")

        if not y_true:
            raise ValueError("No valid word alignments found to build confusion matrix.")
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        off_diagonal_rows = cm.sum(axis=1) - np.diag(cm)
        off_diagonal_cols = cm.sum(axis=0) - np.diag(cm)
        label_mask = (off_diagonal_rows > 0) | (off_diagonal_cols > 0)

        if not np.any(label_mask):
            raise ValueError("All words were recognized correctly; confusion matrix would be empty.")
        labels = [label for label, keep in zip(labels, label_mask) if keep]
        cm = cm[np.ix_(label_mask, label_mask)]

        row_sums = cm.sum(axis=1, keepdims=True)
        annot_values = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

        # Sort by diagonal values (lowest to highest)
        diag_values = np.diag(annot_values)
        sort_idx = np.argsort(diag_values)

        # Reorder confusion matrix, percentages, and labels
        cm = cm[np.ix_(sort_idx, sort_idx)]
        annot_values = annot_values[np.ix_(sort_idx, sort_idx)]
        labels = [labels[i] for i in sort_idx]

        # only take top n for better visualization
        n = 50
        if len(labels) > n:
            labels = labels[:n]
            cm = cm[:n, :n]
            annot_values = annot_values[:n, :n]

        plt.figure(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.6)))
        # heatmap color follows the normalized values
        sns.heatmap(annot_values, annot=cm, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Word-level Confusion Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
            
    def run(self):
        self.load_model(self.model_dir)
        model_choice = f"{self.model_dir.split('/')[-2]}_{self.model_dir.split('/')[-1]}"
        df = self.transcribe_directory()
        output_confusion = os.path.join(OUTPUT_ROOT, f"{model_choice}.png")
        self.generate_confusion_matrix(df, output_confusion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory where the trained model is saved.")
    args = parser.parse_args()
    visualizer = visualization(args.model_dir)
    visualizer.run()
