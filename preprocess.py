import argparse
import glob, os
import numpy as np
from scipy.io import wavfile
from audiomentations import Compose, SomeOf, AddGaussianNoise, AddGaussianSNR, TimeStretch, PitchShift, Shift, AddBackgroundNoise, AddShortNoises, PolarityInversion, ApplyImpulseResponse
from audiomentations.core.audio_loading_utils import load_sound_file
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf
import pandas as pd
from tqdm import tqdm

BASE = "./data/train/train"
InPath = os.path.join(BASE, "train")
InLabelPath = os.path.join(BASE, "train-toneless.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--n_augmentations", type=int, required=True)
args = parser.parse_args()
if args.dataset not in ["TPG", "TPGB", "TPGBIR"]:
    raise ValueError("Invalid dataset name. Choose from: TPG, TPGB, TPGBIR")
os.system(f"rm -rf ./data/train/train/{args.dataset}")
    
print("generating augmented dataset:", args.dataset)
OutPath = os.path.join(BASE, args.dataset)
OutLabelPath = os.path.join(OutPath, "metadata.csv")

background_noises_path = os.path.join(BASE, "background_noises")
short_noises_path = os.path.join(BASE, "short_noises")
rir_path = os.path.join(BASE, "rir")
os.makedirs(OutPath, exist_ok=True)
sr = 16000

augment1 = naf.Sometimes([
    naa.VtlpAug(sampling_rate=sr, zone=(0.0, 1.0), coverage=1.0, factor=(0.9, 1.1)),
    ], aug_p=0.4)

augment_TPG = Compose([
    AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.2),
    TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
])

augment_TPGB = Compose([
    AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.2),
    TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
    AddBackgroundNoise(
        sounds_path=background_noises_path,
        min_snr_db=10,
        max_snr_db=30.0,
        p=0.4),
])

augment_TPGBIR = Compose([
    AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.2),
    TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
    AddBackgroundNoise(
        sounds_path=background_noises_path,
        min_snr_db=10,
        max_snr_db=30.0,
        p=0.4),
    # AddShortNoises(
    # sounds_path=short_noises_path,
    # min_snr_db=10,
    # max_snr_db=30.0,
    # noise_rms="relative_to_whole_input",
    # min_time_between_sounds=2.0,
    # max_time_between_sounds=8.0,
    # p=0.3),
    ApplyImpulseResponse(
            ir_path=rir_path, p=0.4
        )
])

inLabels = pd.read_csv(InLabelPath)
outLabels = []

files = [f for f in os.listdir(InPath) if f.endswith(".wav")]
for file in tqdm(files, desc="Processing files", unit="file"):
    try:
        file_id = int(os.path.splitext(file)[0])
    except ValueError:
        tqdm.write(f"[WARNING] Skipping non-numeric filename: {file}")
        continue

    row = inLabels[inLabels["id"] == file_id]
    if row.empty:
        tqdm.write(f"[WARNING] No label found for file {file}")
        continue
    text = row.iloc[0]["text"]

    samples, sample_rate = load_sound_file(
        os.path.join(InPath, file), sample_rate=None
    )
    # Augment/transform/perturb the audio data
    for i in range(args.n_augmentations):
        file_aug = f"{os.path.splitext(file)[0]}_aug{i+1}.wav"
        augmented_samples1 = augment1.augment(samples)
        # augment1 may return a list; ensure we pass a single array to augment2
        src = augmented_samples1[0] if isinstance(augmented_samples1, (list, tuple)) else augmented_samples1
        augment2 = None
        match args.dataset:
            case "TPG":
                augment2 = augment_TPG
            case "TPGB":
                augment2 = augment_TPGB
            case "TPGBIR":
                augment2 = augment_TPGBIR
        augmented_samples2 = augment2(samples=src, sample_rate=sample_rate)
        wavfile.write(
            os.path.join(OutPath, file_aug), rate=sample_rate, data=augmented_samples2
        )
        outLabels.append({
            "file_name": file_aug,
            "transcription": text
        })

outLabels = pd.DataFrame(outLabels)
outLabels.to_csv(OutLabelPath, index=False)
print("output labels saved to:", OutLabelPath)