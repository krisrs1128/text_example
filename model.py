#!/usr/bin/env python
"""
Definitions for common Text Models

It's standard practice to have either a file or a folder with a few nn.Module
class definitions. Each of these classes needs two methods,

(1) An __init__, which asks for all the user-defined parameters required in
  subsequent training, and
(2) A forwards method, which passes input x through the network, producing a
  prediction y_hat and any intermediate featurizations h that the user might
  want access to later on.
"""
import torch.nn as nn

class RNN(nn.Module):
    """
    Vanilla RNN Model

    """
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        """
        Initialize an RNN model (with GRU cells)

        :param embedding_dim: The model expects input that are already
          embedded (or k-hot coded) torch tensors.
        :param hidden_dim: The dimension of the learned feature
          representation, just before passing into a linear layer for
          classification..
        :param output_dim: The dimension of the output of the final linear
          layer.
        :param num_layers: The number of layers individual RNN cells.
        :param dropout: The dropout rate at the final linear layer.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = 32

        # GRU cell definition
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # final linear layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Pass input through the network

        :param x: The input, already embedded sentence, to pass into the
          network.
        :return: A tuple with the following elements,
          - h_1n: The hidden representations, as they evolve over the sequence.
          - h_n: The hidden representation, just before the linear
              transformation. Has dimension number of layers x hidden dimension.
          - linear_hn: The post-linear model transformations, which can be
              passed into a final logit for classification.
        """
        h_1n, h_n = self.rnn(x)
        linear_hn = self.fc(h_n[-1, :])
        return h_1n, h_n, linear_hn
