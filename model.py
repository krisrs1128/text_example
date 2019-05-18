#!/usr/bin/env python
import torch.nn as nn

def train(model, iterator, optimizer, criterion, embedding, device):
    epoch_loss = 0
    model.train()

    for elem in iterator:
        optimizer.zero_grad()
        x = embedding(elem.sentence)
        x = x.to(device)
        y = elem.author.float().to(device)

        _, _, y_hat = model(x)

        loss = criterion(y_hat.squeeze(1), y)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = 32
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h_1n, h_n = self.rnn(x)
        linear_hn = self.fc(h_n[-1, :])
        return h_1n, h_n, torch.sigmoid(linear_hn)
