import numpy as np

from tqdm import tqdm
from scipy import interpolate

# Sometimes, no-data-values can occur and can be interpolated with this script. This was NOT the case for our study
# but is a process carried around in all of our repositories.
def fix_missing_vals_(data_frames):
    for df in tqdm(data_frames):
        interpolate_nan_bands_(df.data_stack)


# Actual interpolation process
def interpolate_nan_bands_(array):
    for band_index in range(array.shape[2]):
        b = np.isnan(array[:, :, band_index])

        if np.any(~np.isnan(b)):
            x = np.arange(0, b.shape[1])
            y = np.arange(0, b.shape[0])

            b = np.ma.masked_invalid(b)
            xx, yy = np.meshgrid(x, y)

            x1 = xx[~b.mask]
            y1 = yy[~b.mask]

            array[:, :, band_index] = interpolate.griddata((x1, y1), b[~b.mask].ravel(), (xx, yy), method='nearest')


if __name__ == '__main__':
    a = np.random.randn(512, 512, 4)

    a[70, 342, 0] = np.nan
    print(a[70, 342, 0])

    interpolate_nan_bands_(a)
    print(a[70, 342, 0])
