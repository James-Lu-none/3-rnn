import argparse
import os
import librosa
import torch
import evaluate
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from models import *
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt

DATA_ROOT = "data/train"
MODEL_ROOT = "model"
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Use processor's built-in padding for input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # Use tokenizer's built-in padding for labels (creates attention_mask)
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present at the start
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
    
def levenshtein(a: str, b: str) -> int:
    """Compute character-level Levenshtein distance."""
    m, n = len(a), len(b)

    if m == 0:
        return n
    if n == 0:
        return m

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]
class AudioDataset(Dataset):

    def __init__(self, dataframe, audio_dir, processor, max_length=40):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = str(row['file_name'])
        transcription = row['transcription']
        
        if not file_name.endswith('.wav'):
            return self._get_dummy_sample()
        
        try:
            audio, sr = librosa.load(os.path.join(self.audio_dir, file_name), sr=16000)
            max_samples = int(self.max_length * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            )

            labels = self.processor.tokenizer(
                transcription,
                return_tensors="pt",
            )

            return {
                "input_features": inputs.input_features.squeeze(0),
                "labels": labels.input_ids.squeeze(0)
            }

        except Exception as e:
            print(f"error: {file_name}: {e}")
            return {
                "input_features": torch.zeros(80, 1),
                "labels": torch.tensor([self.processor.tokenizer.eos_token_id], dtype=torch.long)
            }
class Train:
    def __init__(self, dataset = None, model_state_path=None, model_choice=None):
        self.model_choice = model_choice
        self.model_state_path = model_state_path
        self.dataset = dataset
        self.train_dataset = None
        self.val_dataset = None
        self.processor = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None


    def load_model(self):
        try:
            model_fn = globals()[self.model_choice]
            self.processor, self.model = model_fn()
        except KeyError:
            raise ValueError(f"Unknown model choice: {self.model_choice}")
        # self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_data(self):
        dataset_dir = f"{DATA_ROOT}/{self.dataset}"
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        print(f"Loading dataset from: {dataset_dir}")
        df = pd.read_csv(metadata_path)
        print(f"Total samples in dataset: {len(df)}")

        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

        # Load folder containing metadata.csv + wav files
        self.train_dataset = AudioDataset(train_df, dataset_dir, self.processor, max_length=30)
        self.val_dataset = AudioDataset(val_df, dataset_dir, self.processor, max_length=30)

    def setup_trainer(self):
        from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

        os.environ["WANDB_PROJECT"] = "whisper-finetune-project"
        args = Seq2SeqTrainingArguments(
            output_dir=f"{MODEL_ROOT}/{self.model_choice}",

            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,

            max_grad_norm=1.0,

            learning_rate=1e-5,
            warmup_steps=500,
            lr_scheduler_type="linear",
            weight_decay=0.01,

            num_train_epochs=5,

            eval_strategy="steps",
            eval_steps=300,
            save_strategy="steps",
            save_steps=300,
            save_total_limit=5,

            logging_steps=50,
            logging_dir="./logs",

            load_best_model_at_end=True,
            metric_for_best_model="levenshtein",
            greater_is_better=False,

            fp16=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,

            predict_with_generate=True,
            generation_max_length=225,
            push_to_hub=False,
            
            generation_config=self.model.generation_config,
        )
        
        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids

            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

            pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            distances = []
            for p, r in zip(pred_str, label_str):
                dist = levenshtein(p, r)
                norm = dist / max(1, len(r))  # character-level normalized Levenshtein
                distances.append(norm)

            mean_lev = float(sum(distances) / len(distances))

            return {"levenshtein": mean_lev}
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
        )
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.processor.feature_extractor,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    def run(self):
        self.load_model()
        self.load_data()
        self.setup_trainer()

        print("Training started...")
        self.trainer.train()

        timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')

        best_lev = self.trainer.state.best_metric
        best_lev_str = f"{best_lev:.4f}" if best_lev is not None else "NA"
        save_dir = f"{MODEL_ROOT}/{self.model_choice}/{timestamp}_{best_lev_str}"
        print(f"Saving model to {save_dir}...")
        
        os.makedirs(save_dir, exist_ok=True)
        self.trainer.save_model(save_dir)
        self.processor.save_pretrained(save_dir)

        print("Saving metrics to CSV and plots...")
        log_history = self.trainer.state.log_history
        json_path = os.path.join(save_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(log_history, f, indent=2)

        df = pd.DataFrame(log_history)

        csv_path = os.path.join(save_dir, "metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to: {csv_path}")

        plt.figure()
        if "loss" in df.columns:
            plt.plot(df["step"], df["loss"], label="train_loss")
        if "eval_loss" in df.columns:
            plt.plot(df["step"], df["eval_loss"], label="eval_loss")

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training / Evaluation Loss")
        plt.legend()
        plt.tight_layout()
        loss_plot_path = os.path.join(save_dir, "loss_plot.png")
        plt.savefig(loss_plot_path, dpi=150)
        plt.close()
        print(f"Loss plot saved to: {loss_plot_path}")

        plt.figure()
        plt.plot(df["step"], df["eval_levenshtein"], marker="o")
        plt.xlabel("Step")
        plt.ylabel("Normalized Levenshtein")
        plt.title("Levenshtein Distance over Training")
        plt.tight_layout()
        lev_plot_path = os.path.join(save_dir, "levenshtein_plot.png")
        plt.savefig(lev_plot_path, dpi=150)
        plt.close()
        print(f"Levenshtein plot saved to: {lev_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_choice", type=str, required=True)
    args = parser.parse_args()

    trainer = Train(
        dataset=args.dataset,
        model_choice=args.model_choice
    )

    trainer.run()
