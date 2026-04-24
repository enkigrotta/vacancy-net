"""Train ∅-NET (VacancyNet) on CIFAR-10."""

import argparse

import torch

from config import Config
from model_vacancy_net import VacancyNet
from data import get_cifar10_loaders
from metrics import codebook_utilization, reconstruction_mse
from utils import TrainingLogger, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device

    torch.manual_seed(cfg.seed)
    device = cfg.resolve_device()

    train_loader, test_loader = get_cifar10_loaders(cfg)

    model = VacancyNet(cfg).to(device)
    model.setup_gradient_engine()

    # Δ₀_initial: k-means on first K_means_batches batches
    z_flat = model.collect_init_data(train_loader, cfg.kmeans_batches, device)
    if z_flat.numel() > 0:
        model.vacancy.initialize_from_data(z_flat)

    logger = TrainingLogger(cfg.vacancy_net_log_dir, 'vacancy_net')
    logger.log_event('config', **{k: str(v) for k, v in vars(cfg).items()
                                  if not k.startswith('_')})

    for epoch in range(cfg.num_epochs):
        model.train()
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            step_res = model.train_step(x, epoch=epoch, batch=i)

            if step_res['replicant_event'] is not None:
                logger.log_event('replicant', **step_res['replicant_event'])
                ev = step_res['replicant_event']
                print(f'⟳ ep{epoch} b{i}: K {ev["K_old"]}->{ev["K_new"]} '
                      f'(splits={ev["n_splits"]} merges={ev["n_merges"]} '
                      f'shifts={ev["n_shifts"]} res={ev["n_resurrections"]})')

            if i % cfg.log_interval == 0:
                logger.log_step(
                    epoch=epoch, batch=i,
                    L_recon=step_res['L_recon'],
                    L_commit=step_res['L_commit'],
                    L_total=step_res['L_total'],
                    K_eff=step_res['K_eff'],
                    tension=step_res['tension'],
                    valve_shed=step_res['valve_shed'],
                )
                print(f'ep{epoch} b{i} L={step_res["L_total"]:.4f} '
                      f'K={step_res["K_eff"]} tension={step_res["tension"]}')

            # Periodic Δ observation
            if model.observer.is_due():
                try:
                    metrics = model.observer.observe(
                        model.encoder, model.vacancy, model.decoder, x,
                    )
                    logger.log_event('delta_observation', epoch=epoch, batch=i, **metrics)
                except Exception as e:
                    logger.log_event('delta_observation_error',
                                     epoch=epoch, batch=i, error=str(e))

        # End-of-epoch eval
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(test_loader))
            x = x.to(device)
            out = model.forward(x)
            mse = reconstruction_mse(x, out['x_hat'])
            util = codebook_utilization(out['indices'], model.vacancy.K_eff)
            logger.log_event('eval', epoch=epoch, mse=mse, util=util,
                             K_eff=model.vacancy.K_eff,
                             n_replicant_events=model._replicant_count)
            print(f'=== epoch {epoch} eval: mse={mse:.4f} util={util:.2f} '
                  f'K={model.vacancy.K_eff} ⟳_total={model._replicant_count} ===')

        if (epoch + 1) % cfg.save_interval == 0:
            save_checkpoint(model, f'{cfg.vacancy_net_log_dir}/ckpt_ep{epoch}.pt',
                            epoch=epoch, K_eff=model.vacancy.K_eff)

    save_checkpoint(model, f'{cfg.vacancy_net_log_dir}/ckpt_final.pt',
                    epoch=cfg.num_epochs - 1, K_eff=model.vacancy.K_eff)
    print('Done. Logs in', logger.path)


if __name__ == '__main__':
    main()
