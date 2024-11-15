import shutup
import random

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

shutup.please()


class DataFrame:
    def __init__(self, index_w, index_h, data_stack, overlay, src_part, classification):
        self.index_w = index_w
        self.index_h = index_h
        self.src_part = src_part
        self.data_stack = data_stack
        self.overlay = overlay
        self.classification = classification

    def show_rgb(self):
        plt.imshow(self.data_stack.transpose((1, 2, 0)).astype('uint8')[:, :, 0:3])
        plt.show()

    def show_overlay(self):
        cs = [(random.random(), random.random(), random.random()) for _ in range(len(np.unique(self.overlay)))]
        cs[0] = (0, 0, 0)

        plt.imshow(self.overlay.squeeze(0), cmap=ListedColormap(cs))
        plt.colorbar()
        plt.show()
