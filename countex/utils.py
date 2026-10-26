import random
import torch

import numpy as np


def get_device():
    """
    :return: the device on which the model is being trained.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    Set the seed for random, numpy, and torch's default seeds.

    :param seed: int
    :return: void
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
