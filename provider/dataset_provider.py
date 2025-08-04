import os
import random
import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision.transforms import v2

random.seed(251199)


# The .npz frames that we saved after overlay are loaded here and being prepared for pytorch. (converted to tensors)
class NrwDataSet(Dataset):
    def __init__(self, data_dir, load_amount):

        self.data_dir = data_dir
        self.dataset = []
        self.class_targets = []

        frame_files = os.listdir(data_dir)

        c = 0
        for frame_file in frame_files:
            frame = np.load(os.path.join(data_dir, frame_file), allow_pickle=True)

            # Get all channels
            red = frame["red"]
            green = frame["green"]
            blue = frame["blue"]
            dsm = frame["dom"]

            # Stack input data and get the ground truth that was previously rasterized within the overlay-function
            data = np.stack((red, green, blue, dsm))
            mask = frame["gt"]

            self.dataset.append((data, mask, frame_file))
            # self.class_targets.append(frame.classification)

            # transformed = transforms(data, mask)

            # self.dataset.append((transformed[0], transformed[1], frame_file))
            # self.class_targets.append(frame.classification)

            c += 1

            # Small check for tests if a tinier dataset should be loaded
            if load_amount > 0:
                if c == load_amount:
                    break

    def __len__(self):
        return len(self.dataset)


    # Get single items by name. Useful for testing by viewing a dataframe with the view_dataframe method in the
    # utils directory
    def __getitem_by_name__(self, name_id):
        frame = np.load(os.path.join(self.data_dir, "df_" + name_id + ".npz"), allow_pickle=True)

        red = frame["red"]
        green = frame["green"]
        blue = frame["blue"]
        dsm = frame["dom"]

        data = np.stack((red, green, blue, dsm))
        mask = frame["gt"]

        return torch.Tensor(data), mask

    # Troch's default gget_item method
    def __getitem__(self, index):
        data_tuple = self.dataset[index]

        data = torch.Tensor(data_tuple[0])
        mask = torch.Tensor(data_tuple[1])
        frame_file = data_tuple[2]

        return data, mask, frame_file


# Create dataset of train, val or test
def get_loader(npz_dir, batch_size, num_workers=2, pin_memory=True, shuffle=True, load_amount=0):
    train_ds = NrwDataSet(npz_dir, load_amount)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )
    return train_loader


# Debug method if we want to get directly the data slices, not the whole loader
def get_dataset(npz_dir, load_amount=0):
    return NrwDataSet(npz_dir, load_amount)
