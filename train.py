# train.py
import argparse
import os
import torch
import evaluate
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from models import *
from preprocess import prepare_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_feats, return_tensors="pt"
        )

        raw_labels = [f["labels"] for f in features]
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": raw_labels},
            padding=True,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

class Train:
    def __init__(self, dataset = None, model_state_path=None, model_choice=None):
        self.model_choice = model_choice
        self.model_state_path = model_state_path
        self.dataset = dataset
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
        
    def load_data(self):
        ds = load_dataset("audiofolder", data_files={"train": self.dataset})
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ds = ds["train"].train_test_split(test_size=0.2)

        print("Mapping train...")
        self.train_dataset = ds["train"].map(
            lambda x: prepare_dataset(x, self.processor, augment=True),
            remove_columns=ds["train"].column_names,
            num_proc=32,
            desc="train",
        )

        print("Mapping eval...")
        self.eval_dataset = ds["test"].map(
            lambda x: prepare_dataset(x, self.processor, augment=False),
            remove_columns=ds["test"].column_names,
            num_proc=32,
            desc="eval",
        )


    def setup_trainer(self):
        from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

        os.environ["WANDB_PROJECT"] = "whisper-finetune-project"
        args = Seq2SeqTrainingArguments(
            model_choice=self.model_choice,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=5000,
            fp16=True,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            eval_strategy="steps",
            predict_with_generate=True,
            generation_max_length=225,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
        )

        wer_metric = evaluate.load("wer")

        def compute_metrics(pred):
            pred_ids = pred.predictions
            lbl_ids = pred.label_ids
            lbl_ids[lbl_ids == -100] = self.processor.tokenizer.pad_token_id

            pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
            lbl_str = self.processor.batch_decode(lbl_ids, skip_special_tokens=True)

            return {"wer": wer_metric.compute(predictions=pred_str, references=lbl_str)}

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.processor.feature_extractor,
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(self.processor),
            compute_metrics=compute_metrics,
        )

    def run(self):
        self.load_data()
        self.setup_trainer()

        print("Training started...")
        self.trainer.train()

        print("Saving model...")
        self.trainer.save_model(f"{self.model_choice}/final")
        self.processor.save_pretrained(f"{self.model_choice}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_choice", type=str, default="whisper_runs")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

    trainer = Train(
        dataset=args.dataset,
        model_choice=args.model_choice
    )

    trainer.run()
