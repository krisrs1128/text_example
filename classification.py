#!/usr/bin/env python
import torch
import pandas as pd


vocab = Voc("poems")

sentences = pd.read_csv("data_prep/sentences.csv")

for _, row in sentences.iterrows():
    vocab.add_sentence(row["sentence"])

# woe = 119

vocab.replace_rare(10)
vocab.reindex()
sentences["sentence"]
sentences = sentences.sort_values("n_words", ascending=False)

x = inputVar(list(sentences["sentence"][:10].values), vocab)
