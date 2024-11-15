"""
    Script to build up dataframes from data from Koger et al.
"""

import os

from PIL import *
import matplotlib.pyplot as plt
import numpy as np
import config.config as cfg

from preprocessing.slicing import slice_n_dice, check_integrity

from preprocessing.export import export_to_npz


def walk(sites):
    for site in sites:
        vec_paths = {}

        for feature_name, feature_id in cfg.get_stolsnek_features().items():
            path = os.path.join(cfg.get_raw_data_path_koger(), "digitized/{}/".format(site), feature_name)

            if os.path.isfile(path):
                vec_paths[path] = feature_id

        big_stack, big_overlay = overlay(
            tif_path=os.path.join(cfg.get_raw_data_path_koger(), "result_{}.tif".format(site)),
            vec_paths=vec_paths
        )
        frames = slice_n_dice(big_stack, big_overlay, cfg.blob_size(), site)
        frames = check_integrity(frames)
        # fix_missing_vals_(frames)
        export_to_npz(frames=frames, export_tif=False, name="koger")


def overlay(tif_path, vec_paths, rast_paths):
    pass

    return np.zeros((1, 1, 1024, 1024))
