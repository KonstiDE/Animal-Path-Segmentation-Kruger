import shutup
shutup.please()

import os.path

import matplotlib.pyplot as plt
import numpy as np

import torch
import config.config as cfg

from tqdm import tqdm
from provider.dataset_provider import get_loader
from model.unet.unet_model import AsppUNET

from metrics.linebuff_accuracy import linebuff_accuracy

from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure

device = "cuda:0"


def test(checkpoint_path, pt_file, test_data_path):
    torch.cuda.empty_cache()

    result_path = os.path.join(checkpoint_path, "results")

    if os.path.isdir(checkpoint_path) and not os.path.isdir(result_path):
        os.mkdir(result_path)

    unet = AsppUNET(in_channels=4, out_channels=1, aspp=False)
    unet.load_state_dict(torch.load(checkpoint_path + pt_file)['net_state_dict'])
    unet.to(device)

    unet.eval()
    torch.no_grad()

    mae_torch = MeanAbsoluteError().to(device)
    mse_torch = MeanSquaredError().to(device)
    ssim_torch = StructuralSimilarityIndexMeasure().to(device)

    loader = get_loader(test_data_path, 1, shuffle=False)

    loop = tqdm(loader)

    accs = []
    maes = []
    mses = []
    ssims = []

    for (data, target, file) in loop:
        data = data.to(device)
        target = target.to(device)

        prediction = unet(data)

        lbf = linebuff_accuracy(prediction, target, device=device)

        prediction = torch.sigmoid(prediction)

        mae = mae_torch(prediction, target)
        mse = mse_torch(prediction, target)
        ssim = ssim_torch(prediction, target)

        mae_torch.reset()
        mse_torch.reset()
        ssim_torch.reset()

        prediction = prediction.squeeze(0).squeeze(0).detach().cpu()
        target = target.squeeze(0).squeeze(0).detach().cpu()

        data = data.squeeze(0).cpu().numpy()
        red = data[0]
        red_normalized = (red * (255 // red.max())).astype(np.uint8)
        green = data[1]
        green_normalized = (green * (255 // green.max())).astype(np.uint8)
        blue = data[2]
        blue_normalized = (blue * (255 // blue.max())).astype(np.uint8)

        beauty = np.dstack((red_normalized, green_normalized, blue_normalized))

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        im = axs[0].imshow(beauty)
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])
        # plt.colorbar(im, ax=axs[0])

        im = axs[1].imshow(prediction, cmap="viridis")
        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])
        # im.set_clim(0, max(prediction.max(), target.max()))
        plt.colorbar(im, ax=axs[1])

        im = axs[2].imshow(target, cmap="viridis")
        axs[2].set_xticklabels([])
        axs[2].set_yticklabels([])
        plt.colorbar(im, ax=axs[2])

        accs.append(lbf)
        maes.append(mae)
        mses.append(mse)
        ssims.append(ssim)

        plt.savefig(os.path.join(result_path, file[0].replace("df_", "").replace(".npz", "") + ".png"))
        plt.close(fig)

    print(str(sum(accs) / len(accs)))
    print(str(sum(maes) / len(maes)))
    print(str(sum(mses) / len(mses)))
    print(str(sum(ssims) / len(ssims)))


if __name__ == '__main__':
    test(os.path.join(
        "E:/paper/ray_results/all_batched4_lbf_thickness9/run_df373_00001_1_apseg_version=dynamic_buffer,batch_size=4,lr=0.0001_2024-06-25_22-31-16/checkpoint_000037/"),
        "model_epoch38.pt",
        cfg.get_test_path(version="dynamic_buffer")
    )
