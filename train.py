#!/usr/bin/env python
"""
Training Utilities

It's standard to wrap up some of the details of pytorch model training in some
external utilities.
"""
import torch.nn as nn

def train(model, iterator, optimizer, loss_fun, embedding, device):
    """
    Train Model for an Epoch

    :param model: The model to update, a nn.Module class.
    :param iterator: A torch data iterator, from which training examples will
      be drawn.
    :param optimizer: The object that updates weights over iterations.
    :param loss_fun: A function that computes a loss, to guide weight updates.
    :param embedding: An nn.Embedding that can be used to convert a text
      sentence into its numerical equivalent.
    :param device: Either the CPU or GPU device on which to put the model when
      making updates.
    :return: A tuple with the following elements,
      - model: The model, with updated parameters.
      - epoch_loss: The average loss over the epoch
    """
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

    return model, epoch_loss / len(iterator)
