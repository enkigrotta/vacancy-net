"""Train BaselineVQVAE on CIFAR-10."""

import argparse

import torch

from config import Config
from model_baseline import BaselineVQVAE
from data import get_cifar10_loaders
from metrics import codebook_utilization, reconstruction_mse
from utils import TrainingLogger, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs from Config')
    parser.add_argument('--device', type=str, default=None,
                        help='cpu / cuda / auto')
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device

    torch.manual_seed(cfg.seed)
    device = cfg.resolve_device()

    train_loader, test_loader = get_cifar10_loaders(cfg)

    model = BaselineVQVAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)

    # Initialize codebook via k-means on first batch
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            z_e = model.encoder(x)
            z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, cfg.latent_dim)
            model.initialize_codebook(z_flat)
            break

    logger = TrainingLogger(cfg.baseline_log_dir, 'baseline')
    logger.log_event('config', **{k: str(v) for k, v in vars(cfg).items()
                                  if not k.startswith('_')})

    for epoch in range(cfg.num_epochs):
        model.train()
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            out = model(x)
            opt.zero_grad()
            out['L_total'].backward()
            opt.step()

            if i % cfg.log_interval == 0:
                util = codebook_utilization(out['indices'], model.K)
                logger.log_step(
                    epoch=epoch, batch=i,
                    L_recon=float(out['L_recon'].item()),
                    L_commit=float(out['L_commit'].item()),
                    L_total=float(out['L_total'].item()),
                    codebook_util=util,
                )
                print(f'ep{epoch} b{i} L_total={out["L_total"].item():.4f} util={util:.2f}')

        # End-of-epoch eval
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(test_loader))
            x = x.to(device)
            out = model(x)
            mse = reconstruction_mse(x, out['x_hat'])
            util = codebook_utilization(out['indices'], model.K)
            logger.log_event('eval', epoch=epoch, mse=mse, util=util)
            print(f'=== epoch {epoch} eval: mse={mse:.4f} util={util:.2f} ===')

        if (epoch + 1) % cfg.save_interval == 0:
            save_checkpoint(model, f'{cfg.baseline_log_dir}/ckpt_ep{epoch}.pt',
                            epoch=epoch)

    save_checkpoint(model, f'{cfg.baseline_log_dir}/ckpt_final.pt',
                    epoch=cfg.num_epochs - 1)
    print('Done. Logs in', logger.path)


if __name__ == '__main__':
    main()
