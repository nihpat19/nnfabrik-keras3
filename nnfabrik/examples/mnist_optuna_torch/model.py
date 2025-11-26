from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import keras


# class MNISTModel(keras.Model):
#     def __init__(self, out_dim: int, h_dim: int = 5) -> None:
#         super().__init__()
#
#         self.fc1 = keras.layers.Dense(h_dim,activation="relu")
#         self.fc2 = keras.layers.Dense(out_dim,activation="softmax")
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         return self.fc2(x)

def get_mnist_model(in_dim, out_dim, h_dim):
    inputs = keras.Input(shape=(in_dim,),name="digits")
    x1 = keras.layers.Dense(h_dim, activation='relu')(inputs)
    outputs = keras.layers.Dense(out_dim,activation='softmax',name="predictions")(x1)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def mnist_model_fn(dataloaders: Dict, seed: int, **config) -> keras.Model:
    """
    Builds a model object for the given config
    Args:
        dataloaders: a dictionary of data loaders
        seed: random seed (e.g. for model initialization)
    Returns:
        Instance of torch.nn.Module
    """
    # get the input and output dimension for the model
    first_input, first_output = next(iter(dataloaders["train"]))
    in_dim = int(np.prod(first_input.shape[1:]))
    out_dim = 10

    torch.manual_seed(seed)  # for reproducibility (almost)
    model = get_mnist_model(in_dim, out_dim, h_dim=config.get("h_dim", 5))

    return model
