#!/bin/bash

src_dir=data/train/train
dst_dir=data/train/fixed_train
mkdir -p "$dst_dir"
for f in "$src_dir"/*.wav; do
    sox "$f" -r 16000 -e signed-integer -b 16 -c 1 "$dst_dir/$(basename "$f")"
done