import os
import tempfile

import torch
import shutup
import sys

import torch.nn as nn
import torch.optim as optim
import statistics as s
import config.config as cfg

from utils.stopper import PatienceStopper

from provider.dataset_provider import get_loader
from model.unet.unet_model import UNET

from metrics.linebuff_accuracy import linebuff_accuracy

from ray import tune
from ray import train as raytrain
from ray.train import Checkpoint
from ray.experimental.tqdm_ray import tqdm

sys.path.append(os.getcwd())
device = "cuda:0"
shutup.please()


def train(loader, loss_fn, optimizer, scaler, model):
    torch.enable_grad()
    model.train()

    running_loss = []
    running_line = []

    loop = tqdm(loader)

    for (data, target, _) in loop:
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device)
        data = model(data)

        target = target.to(device)

        with torch.cuda.amp.autocast():
            loss = loss_fn(data, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_value = loss.item()

        running_line.append(linebuff_accuracy(data, target))
        running_loss.append(loss_value)

    return s.mean(running_loss), s.mean(running_line)


def valid(loader, loss_fn, model):
    model.eval()

    running_loss = []
    running_line = []

    loop = tqdm(loader)

    for (data, target, _) in loop:
        data = data.to(device)
        data = model(data)

        target = target.to(device)

        with torch.no_grad():
            loss = loss_fn(data, target)

        loss_value = loss.item()

        running_line.append(linebuff_accuracy(data, target))
        running_loss.append(loss_value)

    return s.mean(running_loss), s.mean(running_line)


def run(ray_config):
    torch.cuda.empty_cache()

    # Setup
    model = UNET(in_channels=3, out_channels=1, aspp=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ray_config["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    epochs_done = 0

    model.to(device)

    # Get training data
    train_loader = get_loader(cfg.get_train_path(ray_config["apseg_version"]),
                              ray_config["batch_size"], cfg.num_workers(), cfg.pin_memory())

    # Get validation data
    valid_loader = get_loader(cfg.get_validation_path(ray_config["apseg_version"]),
                              ray_config["batch_size"], cfg.num_workers(), cfg.pin_memory())

    # Loop over all samples (epochs can be infinitly high, because early-stopping anyway steps in after some time)
    for epoch in range(epochs_done + 1, 101):
        training_loss, training_lbf = train(train_loader, loss_fn, optimizer, scaler, model)
        validation_loss, validation_lbf = valid(valid_loader, loss_fn, model)

        # Ray report
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        metrics = {
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_linebuff": training_lbf,
            "validation_linebuff": validation_lbf
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                checkpoint_data,
                os.path.join(temp_checkpoint_dir, "model_epoch{}.pt".format(epoch)),
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            raytrain.report(metrics, checkpoint=checkpoint)


if __name__ == '__main__':
    # scheduler = ASHAScheduler(max_t=100, grace_period=1, reduction_factor=2)

    stopper = PatienceStopper(
        metric="validation_loss",
        mode="min",
        patience=5
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run),
            resources={"cpu": 24, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            metric="validation_loss",
            mode="min",
            num_samples=1
        ),
        param_space=cfg.get_ray_config(),
        run_config=raytrain.RunConfig(
            stop=stopper,
            local_dir="E:/paper/ray_results",
        )
    )
    results = tuner.fit()

    best_result = results.get_best_result("validation_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["validation_loss"]))
    print("Best trial final validation accuracy: {}".format(best_result.metrics["validation_lbf"]))
