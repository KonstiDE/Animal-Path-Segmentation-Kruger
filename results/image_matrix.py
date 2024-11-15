import math

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skl
import warnings

import torch

import config.config as cfg

from provider.dataset_provider import get_dataset
from model.unet.unet_model import AsppUNET

warnings.filterwarnings("ignore")

MODEL_PATH_STATIC = "E:/paper/ray_results/all_batched4_lbf_thickness9/run_df373_00000_0_apseg_version=static_buffer,batch_size=4,lr=0.0001_2024-06-25_22-31-16/checkpoint_000026/model_epoch27.pt"
MODEL_PATH_HYBRID = "E:/paper/ray_results/all_batched4_lbf_thickness9/run_df373_00002_2_apseg_version=hybrid_buffer,batch_size=4,lr=0.0001_2024-06-25_22-31-16/checkpoint_000013/model_epoch14.pt"
MODEL_PATH_DYNAMIC = "E:/paper/ray_results/all_batched4_lbf_thickness9/run_df373_00001_1_apseg_version=dynamic_buffer,batch_size=4,lr=0.0001_2024-06-25_22-31-16/checkpoint_000037/model_epoch38.pt"

BATCH_SIZE = 1
DEVICE = "cuda:0"
px = 1 / plt.rcParams['figure.dpi']


def perform_tests(loaders, models, sample_ids=None):
    if sample_ids is None:
        sample_ids = ["part1_11_21_stolsnek", "part1_28_35_stolsnek", "part1_28_37_stolsnek"]

    for height in range(30, 42):
        fig, axs = plt.subplots(len(sample_ids), 2 + len(models), figsize=(29, height))

        h = 0

        for sample_id in sample_ids:
            first_done = False

            for i in range(len(models)):
                data, target = loaders[i].__getitem_by_name__(sample_id)

                target = target.squeeze(0)
                target[target > 1] = 0

                if not first_done:
                    first_done = True

                    data = data.squeeze(0)

                    red = data[0]
                    green = data[1]
                    blue = data[2]

                    red_normalized = (red * (1 / red.max()))
                    green_normalized = (green * (1 / green.max()))
                    blue_normalized = (blue * (1 / blue.max()))

                    beauty = np.dstack((red_normalized, green_normalized, blue_normalized))

                    im = axs[h, 0].imshow(beauty)
                    axs[h, 0].set_xticklabels([])
                    axs[h, 0].set_yticklabels([])

                    im = axs[h, 1].imshow(target, cmap="viridis")
                    axs[h, 1].set_xticklabels([])
                    axs[h, 1].set_yticklabels([])
                    cbar = plt.colorbar(im, ax=axs[h, 1])
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontsize(26)

                data = data.unsqueeze(0)
                prediction = models[i](data).squeeze(0).squeeze(0).detach().cpu()
                prediction = torch.sigmoid(prediction)

                mae = skl.mean_absolute_error(target, prediction)
                mse = skl.mean_squared_error(target, prediction)

                im = axs[h, 2 + i].imshow(prediction, cmap="viridis")
                axs[h, 2 + i].set_xticklabels([])
                axs[h, 2 + i].set_yticklabels([])
                cbar = plt.colorbar(im, ax=axs[h, 2 + i])
                im.set_clim(0, max(prediction.max(), target.max()))
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(26)
                axs[h, 2 + i].set_xlabel("MAE: {:.2f}\nRMSE: {:.2f}".format(mae, math.sqrt(mse)), fontsize=30)

            h += 1

        plt.tight_layout()
        plt.savefig("matrix_height{}.png".format(height), dpi=400)


def setup():
    loader_static = get_dataset(cfg.get_test_path(version="static_buffer"))
    loader_hybrid = get_dataset(cfg.get_test_path(version="hybrid_buffer"))
    loader_dynamic = get_dataset(cfg.get_test_path(version="dynamic_buffer"))

    unet_static = AsppUNET(in_channels=4)
    unet_hybrid = AsppUNET(in_channels=4)
    unet_dynamic = AsppUNET(in_channels=4)

    unet_static.load_state_dict(torch.load(MODEL_PATH_STATIC)['net_state_dict'])
    unet_hybrid.load_state_dict(torch.load(MODEL_PATH_HYBRID)['net_state_dict'])
    unet_dynamic.load_state_dict(torch.load(MODEL_PATH_DYNAMIC)['net_state_dict'])

    unet_static.eval()
    unet_hybrid.eval()
    unet_dynamic.eval()

    perform_tests(
        [loader_static, loader_hybrid, loader_dynamic],
        [unet_static, unet_hybrid, unet_dynamic],
        [
            "part8_21_49_stolsnek",
            "part1_11_21_stolsnek",
            "part1_28_35_stolsnek",
            "part1_28_37_stolsnek",
            "part8_27_48_stolsnek",
            "part1_12_31_stolsnek",
            "part1_9_12_stolsnek"
        ]
        # [
        #     #urban:
        #     "ndom50_32350_5684_1_nw_2019_9~SENTINEL2X_20190215-000000-000_L3A_T32ULB_C_V1-2.npz",
        #     "ndom50_32345_5699_1_nw_2018_10~SENTINEL2X_20180515-000000-000_L3A_T32ULB_C_V1-2.npz",
        #     #suburban:
        #     "ndom50_32340_5690_1_nw_2018_1~SENTINEL2X_20180515-000000-000_L3A_T32ULB_C_V1-2.npz",
        #     "ndom50_32336_5697_1_nw_2018_12~SENTINEL2X_20180515-000000-000_L3A_T32ULB_C_V1-2.npz",
        #     #idustrial:
        #     "ndom50_32312_5747_1_nw_2018_6~SENTINEL2X_20180515-000000-000_L3A_T32ULC_C_V1-2.npz",
        #     #rural/countryside:
        #     "ndom50_32352_5753_1_nw_2018_14~SENTINEL2X_20180315-000000-000_L3A_T32ULC_C_V1-2.npz",
        #     #vegetation:
        #     "ndom50_32351_5650_1_nw_2019_8~SENTINEL2X_20190615-000000-000_L3A_T32ULB_C_V1-2.npz",
        # ]
    )


if __name__ == '__main__':
    setup()
