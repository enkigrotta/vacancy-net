"""CIFAR-10 data loaders."""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def get_cifar10_loaders(cfg):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=transform,
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, test_loader
