import numpy as np
from scipy.interpolate import interpolate

import config.config as cfg

from objects.dataframe import DataFrame


def slice_n_dice(data, mask, t, part):
    if len(mask.shape) < 3:
        mask = np.expand_dims(mask, 0)

    assert data.shape[1:3] == mask.shape[1:3]

    channels, rows, cols = data.shape
    frames = []

    for r in range(0, rows, t):
        for c in range(0, cols, t):
            tile_data = data[:, r:r + t, c:c + t]
            tile_mask = mask[:, r:r + t, c:c + t]
            class_y = np.max(tile_mask)
            if tile_data.shape[1] != tile_data.shape[2]:
                padded_data = np.zeros((tile_data.shape[0], t, t))
                padded_mask = np.zeros((1, t, t))
                padded_data[:, :tile_data.shape[1], :tile_data.shape[2]] = tile_data
                padded_mask[:, :tile_mask.shape[1], :tile_mask.shape[2]] = tile_mask

                df = DataFrame(
                    index_w=int(r / t),
                    index_h=int(c / t),
                    data_stack=padded_data,
                    overlay=padded_mask,
                    src_part=part,
                    classification=class_y
                )

            else:
                df = DataFrame(
                    index_w=int(r / t),
                    index_h=int(c / t),
                    data_stack=tile_data,
                    overlay=tile_mask,
                    src_part=part,
                    classification=class_y
                )

            frames.append(df)

            print(str.format("Sliced ({:.0f}|{:.0f})", r / t, c / t))

    return frames


def raw_slicing(data, t):
    rows, cols, channels = data.shape
    frames = []

    for r in range(0, rows, t):
        for c in range(0, cols, t):
            tile_data = data[r:r + t, c:c + t, :]
            if tile_data.shape != (t, t, channels):
                padded_data = np.zeros((t, t, channels))
                padded_data[:tile_data.shape[0], :tile_data.shape[1]:] = tile_data

                frames.append(padded_data)
            else:
                frames.append(tile_data)

    return frames


def check_integrity(data_frames):
    valid_frames = []

    for df in data_frames:
        if np.min(df.data_stack) > 0 and not np.all(df.overlay == 0):
            valid_frames.append(df)

    if len(valid_frames) == 0:
        raise Warning("Variable valid_frame is empty after check_integrity() was called.")

    return valid_frames


def check_array_integrity(array):
    return len(np.unique(array)) > 1
