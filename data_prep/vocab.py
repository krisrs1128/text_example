#!/usr/bin/env python
"""
Build a torchtext vocabulary object.

One of the tricker parts of working with text is figuring out how to transform
what starts as raw text into arrays of numbers that can processed by a
computer. A few kinds of steps are especially common,

  1) Removing stopwords
  2) Lemmatization
  3) Removing punctuation or extra whitespace
  4) Handling capitalization / proper nouns
  5) Replacing very rare words with some sort of "Unknown" character
  6) Embedding words using a pretrained embedder

In theory, you could do all this on your own, using numpy / torch arrays. On
the other hand, using some of the utilities in torchtext can make life much
easier, by providing implementations / wrappers for some of these steps. This
is the approach we follow here.
"""
import torch
from torchtext import data
from torch import nn

tokenizer = lambda x: x.split()
author_encode = lambda x: 1 if x == "wordsworth" else 0

def build_data(data_path, **kwargs):
    """
    Preprocess into Vocabulary Object

    :param data_path: The path to the raw CSV file containing the gutenberg
      data.
    :return A tuple with the following elements. Each is an already
      preprocessed string of text.
      - train: The sentences in the training split.
      - val: The sentences in the validation split.
      - test: The sentences in the test split.
      - vocab: A vocabulary object, indexing words and removing rare ones.
    """
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
