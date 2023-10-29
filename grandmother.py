"""# https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Choose relatively prime sizes for inner layers so they're easy to track.

IN_FEATURES = 3
MID_1_FEATURES = 23
MID_2_FEATURES = 13
OUT_FEATURES = 2


def define_model():
    model = nn.Sequential(
        nn.Linear(in_features=IN_FEATURES, out_features=MID_1_FEATURES),
        nn.ReLU(),
        nn.Linear(in_features=MID_1_FEATURES, out_features=MID_2_FEATURES),
        nn.ReLU(),
        nn.Linear(in_features=MID_2_FEATURES, out_features=OUT_FEATURES),
        nn.Sigmoid(),
    )
    return model


# The size of a batch is implicit. In this example,
# you should pass in a PyTorch tensor of shape
# (BATCH_SIZE, IN_FEATURES) into the first layer and
# expect a tensor of shape (BATCH_SIZE, MID_1_FEATURES)
# in return.


BATCH_SIZE = 15
EPOCHS = 37


x_train = torch.randn(EPOCHS, BATCH_SIZE, IN_FEATURES)
y_train = torch.zeros(EPOCHS, BATCH_SIZE, OUT_FEATURES)


def set_up_training_data():
    for e in range(EPOCHS):
        for b in range(BATCH_SIZE):
            y_train[e, b, 0] = x_train[e, b, 0]
            y_train[e, b, 1] = x_train[e, b, 2]


def train_model():
    model = define_model()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for e in range(EPOCHS):
        y_pred = model(x_train[e, :, :])
        loss = loss_fn(y_pred, y_train[e, :, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def evaluate_model(model, x):
    result = model(x)
    return result


set_up_training_data()
model = train_model()
x_test = torch.tensor([[3.0, 4.0, 5.0]])
result = evaluate_model(model, x_test)
print(result)