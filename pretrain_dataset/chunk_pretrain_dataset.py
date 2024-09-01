import json
from pathlib import Path
import zstandard as zstd
from litdata import optimize, TokensLoader
from functools import partial
import os
from typing import List, Optional
import sentencepiece
import sys
import torch


from transformers import AutoTokenizer

def tokenize_fn(filepath, tokenizer=None):
    with open(filepath, "r", encoding="utf-8") as f:
        for row in f:
            text = json.loads(row)["target"]
            text_ids = tokenizer.encode(text)
            yield torch.tensor(text_ids)


if __name__ == "__main__":
    input_dir = "pretrain_dataset/jsons/"
    inputs = [str(file) for file in Path(f"{input_dir}").rglob("*.jsonl")]  

    tokenizer = AutoTokenizer.from_pretrained("tokenization/expanded_tokenizer/")

    outputs = optimize(
        fn=partial(tokenize_fn, tokenizer=tokenizer),
        inputs=inputs,
        output_dir="pretrain_dataset/chunks/",
        chunk_size=(8192 * 128),
        item_loader=TokensLoader(),
    )
