from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import json

BASE = "./data/train/train"
LEXICON_PATH = os.path.join(BASE, "lexicon.txt")
OUTPUT_DIR = "taiwanese_tokenizer"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


words = []

with open(LEXICON_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            words.append(parts[0].lower())
            words.append(parts[1].lower())
            words.append(parts[2].lower())
# removes duplicate words
words = list(set(words))
# removes iNULL
words.remove('inull')

# Remove duplicates and sort
words = sorted(list(set(words)))

print(f"Loaded {len(words)} syllables.")


special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

# Assign ID: special tokens first
vocab = {}
idx = 0

for tok in special_tokens:
    vocab[tok] = idx
    idx += 1

for w in words:
    vocab[w] = idx
    idx += 1

print(f"Total vocab size: {len(vocab)}")


tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# Whisper expects BOS/EOS handling
tokenizer.post_processor = TemplateProcessing(
    single="<s> $0 </s>",
    pair="<s> $A </s> $B </s>",
    special_tokens=[
        ("<s>", vocab["<s>"]),
        ("</s>", vocab["</s>"]),
    ],
)
tokenizer.save(f"{OUTPUT_DIR}/tokenizer.json")

# Also save vocab.json for Transformers
with open(f"{OUTPUT_DIR}/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

with open(f"{OUTPUT_DIR}/tokenizer_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "tokenizer_class": "WhisperTokenizerFast",
        "model_type": "wordlevel",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
    }, f, indent=2)

special_map = {
    "bos_token": {"content": "<s>", "lstrip": False, "rstrip": False, "single_word": False},
    "eos_token": {"content": "</s>", "lstrip": False, "rstrip": False, "single_word": False},
    "unk_token": {"content": "<unk>", "lstrip": False, "rstrip": False, "single_word": False},
    "pad_token": {"content": "<pad>", "lstrip": False, "rstrip": False, "single_word": False},
}
with open(f"{OUTPUT_DIR}/special_tokens_map.json", "w") as f:
    json.dump(special_map, f, indent=2)
print("✅ Tokenizer generated in:", OUTPUT_DIR)
