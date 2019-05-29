#!/usr/bin/env python
"""
Toy Text Experiment

This takes the output from data_prep/gutenberg.R and runs a simple
classification task on it (is the author wordsworth or keats?). The main steps are,
    - Read in the data
    - Preprocess the text, and put it in an iterator
    - Embed the words and define a model
    - Train the model
    - Evaluate the errors
"""
from data_prep.vocab import build_data
from pipeline.model import RNN
from pipeline.train import train
from torch import nn
from torchtext import data
import json
import torch

# Parameters related to data
data_path = "/data/sentences.csv"
opts = json.load(open("config.json"))
hyper = opts["model"]
train_opts = opts["train"]
embedding_info = opts["processing"]["embedding"]
x_train, _, _, vocab = build_data(data_path, **embedding_info)

# Define an iterator that makes sure batches have similar lengths
train_iter = data.Iterator(x_train, batch_size=opts["train"]["batch_size"], sort_key = lambda x: len(x.Text))
embedding = nn.Embedding(len(vocab), hyper["embedding_dim"])
embedding.weight.data.copy_(vocab.vectors)

# Define the model and optimization framework
model = RNN(**hyper)
optimizer = torch.optim.Adam(model.parameters())
loss_fun = nn.BCEWithLogitsLoss()

# Put everything on the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_fun = loss_fun.to(device)

# train, and save a model after every 5 epochs
for epoch in range(opts["train"]["n_epochs"]):
    model, train_loss = train(model, train_iter, optimizer, loss_fun, embedding, device)
    print("\tTrain Loss: {}".format(train_loss))

    if epoch % 5 == 0:
        torch.save(model.state_dict(), 'model_{}.pt'.format(epoch))
