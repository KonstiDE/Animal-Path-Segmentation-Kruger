import os

import numpy as np
import pickle as pkl
import rasterio as rio

from rasterio.windows import Window

import config.config as cfg


# Export function to save a dataframe helper object to an .npz compressed format
def export_to_npz(frames, name, version):
    for df in frames:
        data = df.data_stack
        overlay = df.overlay

        np.savez(
            os.path.join(os.path.join(
                cfg.get_frame_path(),
                str(version),
                "df_{}_{}_{}_{}.npz".format(
                    df.src_part, df.index_w, df.index_h, name
                )
            )),
            red=data[0],
            green=data[1],
            blue=data[2],
            # dom=data[3],
            gt=overlay
        )
