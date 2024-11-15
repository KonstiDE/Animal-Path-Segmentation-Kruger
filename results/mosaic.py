import os

import rasterio as rio
import shutup
import PIL.Image
import torch

import config.config as cfg
import numpy as np

from tqdm import tqdm

from skimage.morphology import skeletonize
from preprocessing.slicing import raw_slicing
from model.unet.unet_model import AsppUNET
from PIL import Image

from preprocessing.slicing import check_array_integrity

os.environ["R_HOME"] = cfg.get_r_home()
os.environ["PATH"] = cfg.get_r_x64_dll() + ";" + os.environ["PATH"]

from rpy2.robjects.packages import importr

rba = importr("base")
rsf = importr("sf")
rtr = importr("terra")

PIL.Image.MAX_IMAGE_PIXELS = None
shutup.please()


def generate_result(model, part_path, dsm_path, version):
    tif = rtr.rast(part_path)

    dim = list(rba.floor(rba.dim(tif)[0:2]))
    dim = [int(x) for x in dim]

    rgb = np.array(rtr.values(tif)).reshape([dim[0], dim[1], 4])[:, :, 0:3]
    #dsm = rio.open(dsm_path).read()

    #rgb = np.concatenate((rgb, np.expand_dims(dsm.squeeze(0), 2)), axis=2)

    mosaic_width_index = (rgb.shape[0] // cfg.blob_size()) + 1
    mosaic_height_index = (rgb.shape[1] // cfg.blob_size()) + 1

    arrays = raw_slicing(rgb, cfg.blob_size())

    result_image = Image.new("L", (mosaic_height_index * cfg.blob_size(), mosaic_width_index * cfg.blob_size()))

    loop = tqdm(arrays)

    cw = 0
    ch = 0
    for data in loop:
        data = torch.permute(torch.Tensor(data), (2, 0, 1)).unsqueeze(0).cuda()

        data = model(data)
        data = torch.sigmoid(data)

        data = data.squeeze(0).squeeze(0).detach().cpu().numpy()
        data *= 255

        if cw == mosaic_height_index:
            ch += 1
            cw = 0

        result_image.paste(Image.fromarray(data.astype(int)), (cw * cfg.blob_size(), ch * cfg.blob_size()))

        cw += 1

    result_image = result_image.crop((0, 0, rgb.shape[1], rgb.shape[0]))

    matrix = np.array(result_image.getdata())
    matrix = matrix.reshape((1, rgb.shape[0], rgb.shape[1]))

    old_reference = rio.open(part_path)

    mosaic = rio.open(
        'mosaic_' + version + '_buffer.tif',
        'w',
        driver='GTiff',
        width=old_reference.width,
        height=old_reference.height,
        count=1,
        dtype=str(old_reference.dtypes[0]),
        crs=old_reference.crs,
        transform=old_reference.transform
    )

    mosaic.write(matrix)
    mosaic.close()


if __name__ == '__main__':
    model = AsppUNET(in_channels=3, out_channels=1, aspp=False).cuda()

    state_dict = torch.load(os.path.join(
        "E:/paper/ray_results/all_batched4_lbf_thickness9_spectral/run_d00c7_00002_2_apseg_version=hybrid_buffer,batch_size=4,lr=0.0001_2024-07-18_19-18-55/checkpoint_000015/model_epoch16.pt"
    ))["net_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    generate_result(
        model=model,
        part_path="E:/paper/part2_validation/clipped_part2.tif",
        dsm_path="E:/paper/part2_validation/clipped_dsm_part2_resampled.tif",
        version="hybrid"
    )
