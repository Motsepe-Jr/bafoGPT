#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Initialize new tokenizer for continual pre-training
"""

import argparse
import json
import os
from typing import List, Union

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import AutoTokenizer


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)

def expand_vocab_tokenizer(
    source_tokenizer_dir: Union[str, os.PathLike], target_tokenizer_dir: Union[str, os.PathLike], new_tokens: List[str]
) -> None:
    """Expand tokenizer for continue pre-training."""
    if os.path.exists(target_tokenizer_dir):
        raise RuntimeError(f"Find existed directory {target_tokenizer_dir}")

    source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_dir,  use_fast=False)
    source_sp_processor = source_tokenizer.sp_model
    source_spm = sp_pb2_model.ModelProto()
    source_spm.ParseFromString(source_sp_processor.serialized_model_proto())

    print(f"Source tokenizer size: {len(source_sp_processor)}")

    source_spm_tokens = set([p.piece for p in source_spm.pieces])
    for piece in new_tokens:
        assert isinstance(piece, str), f"Invalid token({piece}) type {type(piece)}"
        if piece in source_spm_tokens:
            # Skip existed token.
            continue
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        source_spm.pieces.append(new_p)

    # (multiple of 512)
    current_vocab_size = len(source_spm.pieces)
    padded_vocab_size = find_multiple(current_vocab_size, 512)

    # Add padding tokens to reach the target vocabulary size
    for i in range(padded_vocab_size - current_vocab_size):
        padding_token = f"<padded_vocab_{i}>"
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = padding_token
        new_p.score = 0
        source_spm.pieces.append(new_p)

    print(f"Expanded tokenizer size: {len(source_spm.pieces)}")

    # Save
    os.makedirs(target_tokenizer_dir)
    target_tokenizer_model_path = os.path.join(target_tokenizer_dir, "tokenizer.model")
    with open(file=target_tokenizer_model_path, mode="wb") as fp:
        fp.write(source_spm.SerializeToString())

def main():

    source_tokenizer_dir = "tokenization/gemma_tokenizer/"
    target_tokenizer_dir =  "tokenization/expanded_tokenizer/"
    expand_tokens_file = "tokenization/zulu_tokenizer/zulu_tokens.json"

    with open(expand_tokens_file, mode="r", encoding="utf-8") as fp_reader:
        expand_tokens_data = json.load(fp_reader)

    expand_tokens = []
    for token in expand_tokens_data:
        if token in expand_tokens:
            continue
        expand_tokens.append(token)
    expand_tokens.sort(key=lambda t: len(t), reverse=False)

    expand_vocab_tokenizer(
        source_tokenizer_dir=source_tokenizer_dir,
        target_tokenizer_dir=target_tokenizer_dir,
        new_tokens=expand_tokens,
    )


if __name__ == "__main__":
    main()