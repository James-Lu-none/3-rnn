import argparse
import os
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import pandas as pd
import numpy as np

TEST_ROOT = "data/test-random"
OUTPUT_ROOT = "output"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def load_model(model_dir):
    print(f"Loading model from: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

def transcribe_audio(processor, model, audio_path, max_length_sec=30):
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

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text

def transcribe_directory(processor, model):
    rows = []
    
    for file in sorted(os.listdir(TEST_ROOT)):
        if not file.lower().endswith(".wav"):
            continue

        full_path = os.path.join(TEST_ROOT, file)
        audio_id = os.path.splitext(file)[0]  # "1004370.wav" → "1004370"

        print(f"Processing: {file} ...")

        sentence = transcribe_audio(processor, model, full_path)
        rows.append({"id": audio_id, "sentence": sentence})

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory where the trained model is saved.")
    
    args = parser.parse_args()
    model_choice = f"{args.model_dir.split('/')[-2]}_{args.model_dir.split('/')[-1]}"
    processor, model = load_model(args.model_dir)
    df = transcribe_directory(processor, model)

    timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')
    output_csv = os.path.join(OUTPUT_ROOT, f"{model_choice}_{timestamp}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")
