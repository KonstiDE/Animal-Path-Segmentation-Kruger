import numpy as np

import torch
import torch.nn.functional as f

import time


def secondary_buffer_torch(mask_gt, kernel, thickness) -> torch.Tensor:
    out = f.conv2d(mask_gt, kernel, stride=1, padding=thickness // 2)
    out[out > 1] = 1

    return out


def linebuff_accuracy(pred, target, device="cuda:0", thickness=9):
    kernel = torch.ones((1, 1, thickness, thickness), dtype=torch.float).to(device)
    accs = []

    # Loop over samples in batch
    for i in range(pred.shape[0]):
        channel_data = pred[i]
        channel_target = target[i]

        aoi = torch.flatten(secondary_buffer_torch(channel_target, kernel, thickness).squeeze(0))

        min_val = torch.min(channel_data)
        max_val = torch.max(channel_data)

        if max_val - min_val > 0:
            channel_data = torch.div(channel_data - min_val, max_val - min_val)

        # plt.imshow(aoi.squeeze(0).detach().cpu().numpy(), cmap='gray')
        # plt.show()
        #
        # plt.imshow(channel_data.squeeze(0).detach().cpu().numpy(), cmap='gray')
        # plt.show()
        #
        # plt.imshow(channel_target.squeeze(0).detach().cpu().numpy(), cmap='gray')
        # plt.show()

        channel_data = torch.flatten(channel_data[0])
        channel_target = torch.flatten(channel_target[0])

        accs.append(1 - (torch.sum(
            torch.abs((channel_target[aoi == 1] - channel_data[aoi == 1]))
        ) / torch.numel(aoi[aoi == 1])))

    return torch.mean(torch.stack(accs)).item()


if __name__ == '__main__':
    data, mask = torch.Tensor(
        np.stack([np.zeros((512, 512)), np.zeros((512, 512)), np.zeros((512, 512)), np.zeros((512, 512))])
    ).unsqueeze(1), torch.Tensor(
        np.stack([np.eye(512), np.eye(512), np.eye(512), np.eye(512)])
    ).unsqueeze(1)

    stamp = time.time()
    accuracy = linebuff_accuracy(data.cuda(), mask.cuda(), thickness=9)

    print("Took: {:.3f}s".format(time.time() - stamp))

    print("Bounding Box Accuracy: {:.3f}".format(float(accuracy)))
