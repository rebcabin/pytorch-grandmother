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
from torch.utils.tensorboard import SummaryWriter

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
    """Do a projection of the first and third elements of each
    x vector into the first and second elements of each y vector."""
    for e in range(EPOCHS):
        for b in range(BATCH_SIZE):
            y_train[e, b, 0] = x_train[e, b, 0]
            y_train[e, b, 1] = x_train[e, b, 2]


def train_model():
    model_ = define_model()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model_.parameters(), lr=0.1)

    # TODO: non-working visualization of the loss curve
    writer = SummaryWriter('runs/grandmother_experiment_1')

    for e in range(EPOCHS):
        y_pred = model_(x_train[e, :, :])
        loss = loss_fn(y_pred, y_train[e, :, :])
        writer.add_scalars(main_tag='Training Loss',
                           tag_scalar_dict={'Loss': loss},
                           global_step=e)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.flush()
    writer.close()

    return model_


def evaluate_model(model_, x):
    result_ = model_(x)
    return result_


set_up_training_data()
model = train_model()
# normalized input (sum of squared is ~1.0)
x_test = torch.tensor([[.6, .4, .6928]])
result = evaluate_model(model, x_test)
print(result)