import os.path

import numpy as np
from ray import tune

# Configuration of hyper-parameters
config = {
    "base_path": "C:/Users/s371513/PycharmProjects/apseg/",
    "raw_data_path_stolsnek": "E:/paper",
    "raw_data_path_koger": "E:/koger",
    "frame_path": "E:/frames/",
    "split": {
        'train': (0.7, "train/"),
        'validation': (0.2, "validation/"),
        'test': (0.1, "test/")
    },
    "stolsnek_features": {
        "static_buffer": {
            "buffer_track_mean.gpkg": 1,
            "crossover_area.gpkg": 1,
        },
        "dynamic_buffer": {
            "buffer_track_wide_1.gpkg": 1,
            "buffer_track_mid_0.6.gpkg": 1,
            "buffer_track_narrow_0.3.gpkg": 1,
            "crossover_area.gpkg": 1,
        },
        "hybrid_buffer": {
            "buffer_track_wide_1.gpkg": 1,
            "buffer_track_mid_1.gpkg": 1,
            "buffer_track_narrow_1.gpkg": 1,
            "crossover_area.gpkg": 1,
        },
    },
    "stolsnek_rasters": {
        "dsm_resampled.tif"
    },
    "no_data_value": -2147483648,
    "num_workers": 4,
    "pin_memory": True,
    "blob_size": 512,
    "r_dll": r"C:\Program Files\R\R-4.3.1\bin\x64",
    "r_home": r"C:\Program Files\R\R-4.3.1",
    "ray_config": {
        "apseg_version": tune.grid_search(["static_buffer", "dynamic_buffer", "hybrid_buffer"]),
        "lr": tune.grid_search([1e-04]),
        "batch_size": tune.grid_search([4])
    }
}


def get_frame_path():
    return config["frame_path"]


def get_train_path(version):
    return os.path.join(get_frame_path(), version, config["split"]["train"][1])


def get_validation_path(version):
    return os.path.join(get_frame_path(), version, config["split"]["validation"][1])


def get_test_path(version):
    return os.path.join(get_frame_path(), version, config["split"]["test"][1])


def batch_size():
    return config["batch_size"]


def num_workers():
    return config["num_workers"]


def pin_memory():
    return config["pin_memory"]


def blob_size():
    return config["blob_size"]


def get_base_path():
    return config["base_path"]


def get_raw_data_path_stolsnek():
    return config["raw_data_path_stolsnek"]


def get_raw_data_path_koger():
    return config["raw_data_path_koger"]


def get_r_home():
    return config["r_home"]


def get_r_x64_dll():
    return config["r_dll"]


def get_stolsnek_features():
    return config["stolsnek_features"]


def get_stolsnek_rasters():
    return config["stolsnek_rasters"]


def get_koger_sites():
    return config["koger_sites"]


def get_split_map():
    return config["split"]


def get_no_data_value():
    return config["no_data_value"]



def get_ray_config():
    return config["ray_config"]

