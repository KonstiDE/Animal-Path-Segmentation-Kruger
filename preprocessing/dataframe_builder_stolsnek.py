"""
    Script to build up dataframes from big tifs
"""

import os

import numpy as np
import config.config as cfg
import rasterio as rio

os.environ["R_HOME"] = cfg.get_r_home()
os.environ["PATH"] = cfg.get_r_x64_dll() + ";" + os.environ["PATH"]

from rpy2.robjects.packages import importr

from preprocessing.slicing import slice_n_dice, check_integrity

from preprocessing.export import export_to_npz

rba = importr("base")
rsf = importr("sf")
rtr = importr("terra")


def walk(parts):
    for part in parts:
        for gt_version in cfg.get_stolsnek_features().items():

            vec_paths = {}
            rast_paths = []

            for feature_name, feature_id in gt_version[1].items():
                path = os.path.join(cfg.get_raw_data_path_stolsnek(), "{}".format(part), feature_name)

                if os.path.isfile(path):
                    vec_paths[path] = feature_id

            for raster_name in cfg.get_stolsnek_rasters():
                path = os.path.join(cfg.get_raw_data_path_stolsnek(), "{}".format(part), raster_name)

                if os.path.isfile(path):
                    rast_paths.append(path)

            big_stack, big_overlay = overlay(
                tif_path=os.path.join(cfg.get_raw_data_path_stolsnek(), "{}".format(part), "rgb.tif"),
                vec_paths=vec_paths,
                rast_paths=rast_paths
            )
            frames = slice_n_dice(big_stack, big_overlay, cfg.blob_size(), part)
            frames = check_integrity(frames)
            # fix_missing_vals_(frames)
            export_to_npz(frames=frames, name="stolsnek", version=gt_version[0])


def overlay(tif_path, vec_paths, rast_paths):
    tif = rtr.rast(tif_path)

    dim = list(rba.floor(rba.dim(tif)[0:2]))
    dim = [int(x) for x in dim]

    track_rasters = []
    input_rasters = []

    for (track_file_path, class_id) in vec_paths.items():
        track_file_entry = rtr.vect(track_file_path)
        where = rtr.rasterize(track_file_entry, tif, background=0)

        track_raster = np.array(rtr.values(where)).reshape(dim)

        track_rasters.append(track_raster)

    for raster_file_path in rast_paths:
        raster_file_entry = rio.open(raster_file_path).read()

        input_rasters.append(raster_file_entry)

    overlay_raster = np.sum(np.stack(track_rasters), axis=0)
    overlay_raster[overlay_raster > 1] = 1

    # torch transpose
    rgb = np.array(rtr.values(tif)).reshape([dim[0], dim[1], 4])[:, :, 0:3].transpose((2, 0, 1))

    if len(input_rasters) > 0:
        if len(input_rasters) == 1:
            input_additionals = np.dstack(input_rasters)
        else:
            input_additionals = np.stack(input_rasters, axis=0).squeeze(1)

        rgb = np.concatenate((rgb, input_additionals))

    # numpy transpose
    # plt.imshow(rgb.transpose((1, 2, 0)))
    # plt.show()

    return rgb, overlay_raster
