#!/usr/bin/env python
from torchtext import data
import torch
from torch import nn
import time
from model import train, RNN
from vocab import build_vocab

data_path = "/data/sentences.csv"
n_epochs = 100
embedding_info = {
    "vectors": "glove.6B.50d",
    "vectors_cache": "/data/glove"
}
x_train, _, _, vocab = build_data(data_path, **embedding_info)


train_iter = data.Iterator(x_train, batch_size=32, sort_key = lambda x: len(x.Text))
embedding = nn.Embedding(len(vocab), 50)
embedding.weight.data.copy_(vocab.vectors)
model = RNN(50, 100, 1, 3, 0.2)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

for epoch in range(n_epochs):
    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion, embedding, device)
    end_time = time.time()

    torch.save(model.state_dict(), 'tut2-{}-model.pt'.format(epoch))
    print("\tTrain Loss: {}".format(train_loss))

for epoch in range(n_epochs):
    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion)
    end_time = time.time()

    torch.save(model.state_dict(), 'tut2-{}-model.pt'.format(epoch))
    print("\tTrain Loss: {}".format(train_loss))
