import os
import torch
import config.config as cfg

import matplotlib.pyplot as plt


def load_graphs_from_checkpoint(model_path, epoch):
    if os.path.isfile(os.path.join(model_path, "model_epoch" + str(epoch) + ".pt")):
        checkpoint = torch.load(os.path.join(model_path, "model_epoch" + str(epoch) + ".pt"), map_location='cpu')
        overall_training_loss = checkpoint['training_losses']
        overall_validation_loss = checkpoint['validation_losses']

        overall_training_linebuff = checkpoint['training_linebuffs']
        overall_validation_linebuff = checkpoint['validation_linebuffs']

        plt.figure()
        plt.plot(overall_training_loss, 'blue', label="Training loss")
        plt.plot(overall_validation_loss, 'red', label="Validation loss")
        plt.legend(loc="upper right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

        plt.figure()
        plt.plot(overall_training_linebuff, 'green', label="Training LBA")
        plt.plot(overall_validation_linebuff, 'orange', label="Validation LBA")
        plt.legend(loc="lower right", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.show()

    else:
        print("No model found within {} and epoch {}".format(
            model_path,
            str(epoch)
        ))


if __name__ == '__main__':
    load_graphs_from_checkpoint(os.path.join(
        cfg.get_base_path(),
        "zzz_BCEWithLogitsLoss_Adam_AsppUNET_5e-05"
    ), 30)
