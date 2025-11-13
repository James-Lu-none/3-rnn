import os
import pandas as pd
from tqdm import tqdm

BASE = "./data/train/train"
dict_sentence_dir = os.path.join(BASE, "dict-sentence")
dict_sentence_csv = os.path.join(BASE, "dict-sentence.csv")
dict_word_dir = os.path.join(BASE, "dict-word")
dict_word_csv = os.path.join(BASE, "dict-word.csv")
train_dir = os.path.join(BASE, "train")
train_csv = os.path.join(BASE, "train-toneless.csv")

train_df = pd.read_csv(train_csv)
dict_sentence_df = pd.read_csv(dict_sentence_csv)
dict_word_df = pd.read_csv(dict_word_csv)

hybrid_dir = os.path.join(BASE, "hybrid")

os.system(f"rm -rf {hybrid_dir}")
os.makedirs(hybrid_dir, exist_ok=True)
hybrid_csv = os.path.join(hybrid_dir, "metadata.csv")

def check_conflicts(df, name):
    if df['id'].duplicated().any():
        duplicated_ids = df[df['id'].duplicated()]['id'].unique()
        print(f"Conflicting IDs found in {name}: {duplicated_ids}")
    else:
        print(f"No conflicting IDs in {name}.")

def duplicate_data(df, src_dir, dist_dir, n_duplicates):
    print(f"Duplicating data from {src_dir} to {dist_dir} with {n_duplicates} duplicates each.")
    os.makedirs(dist_dir, exist_ok=True)
    new_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Duplicating data from {src_dir}"):
        file_id = row['id']
        text = row['text']
        src_file_path = os.path.join(src_dir, f"{file_id}.wav")
        for i in range(n_duplicates):
            new_id = f"{file_id}_dup{i+1}"
            dist_file_path = os.path.join(dist_dir, f"{new_id}.wav")
            os.system(f"cp {src_file_path} {dist_file_path}")
            new_rows.append({'id': new_id, 'text': text})
    new_df = pd.DataFrame(new_rows)
    return new_df

dup_train_df = duplicate_data(train_df, train_dir, hybrid_dir, 5)
hybrid_df = pd.concat([dup_train_df, dict_sentence_df, dict_word_df], ignore_index=True)
check_conflicts(hybrid_df, "hybrid dataset")

print(f"Total entries in train dataset after duplication: {len(dup_train_df)}")
print(f"Total entries in dict-sentence dataset: {len(dict_sentence_df)}")
print(f"Total entries in dict-word dataset: {len(dict_word_df)}")

print(f"Total entries in hybrid dataset: {len(hybrid_df)}")
hybrid_df.to_csv(hybrid_csv, index=False)
print(f"Hybrid dataset created at {hybrid_dir} with metadata file {hybrid_csv}.")

