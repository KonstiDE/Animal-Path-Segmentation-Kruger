import os

from matplotlib import pyplot as plt

import config.config as cfg

import numpy as np
import pickle as pkl

from objects.dataframe import DataFrame


def view_dataframe():
    sub = cfg.get_test_path(version="dynamic_buffer")

    files = os.listdir(sub)

    for file in files:
        with open(os.path.join(cfg.get_frame_path(), sub, file), "rb") as f:
            frame = pkl.load(f)

            frame.show_rgb()
            frame.show_overlay()
            f.close()
            input()


if __name__ == '__main__':
    view_dataframe()
