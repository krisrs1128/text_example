#!/usr/bin/env python
import torch
from torchtext import data
from torch import nn

tokenizer = lambda x: x.split()

def author_encode(x):
    if x == "wordsworth":
        return 1
    return 0

def build_data(data_path, **kwargs):
    TEXT = data.Field(sequential=True, tokenize=tokenizer)
    LABEL = data.Field(sequential=False, use_vocab=False, preprocessing=author_encode)
    texts = data.TabularDataset(
        data_path,
        format="csv",
        fields = [("author", LABEL), ("gutenberg_id", None), ("sentence", TEXT), ("train", None), ("n_words", None)]
    )
    train, test = texts.split()
    train, val = train.split(split_ratio=0.5)
    TEXT.build_vocab(train, **kwargs)
    return train, val, test, TEXT.vocab

