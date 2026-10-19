"""
trainer.py  —  training loop for the jigsaw OT-CFM model

Optimisation is identical to the sorting task: AdamW (3e-4, wd 1e-4),
cosine anneal to 1e-5, 300 epochs, batch 256, mixed precision, grad-clip 1.0.
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from metrics import evaluate


def train_epoch(model, loader, optimizer, scaler, device) -> float:
    model.train()
    total, nb = 0.0, 0

    for patches, gt_slot, obs_blk, slot_blk, log_orbit in loader:
        patches  = patches.to(device,  non_blocking=True)
        obs_blk  = obs_blk.to(device,  non_blocking=True)
        slot_blk = slot_blk.to(device, non_blocking=True)
        gt_slot  = gt_slot.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",
                                enabled=CFG.USE_AMP and device.type == "cuda"):
            loss = model.cfm_loss(patches, obs_blk, slot_blk, gt_slot)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        nb    += 1

    return total / max(nb, 1)


def train(model, train_loader, dist_loader, device,
          epochs: int = None, k_samples: int = None):
    """
    Train for `epochs` epochs and return the history.  The final-epoch model is
    used for all reported results; no checkpoint is selected.  Every
    CFG.EVAL_EVERY epochs the K-sample metrics on dist_loader are printed for
    monitoring only.
    """
    epochs    = epochs    or CFG.N_EPOCHS
    k_samples = k_samples or CFG.K_SAMPLES

    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR,
                            weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=CFG.LR_MIN)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=CFG.USE_AMP and device.type == "cuda")

    history = dict(epoch=[], train_loss=[], validity=[], distinct=[],
                   marginal_tv=[], rmse=[], accuracy=[], correct=[])
    hdr = (f"{'Ep':>5}  {'Loss':>10}  {'Valid':>8}  {'Dist@K':>8}  "
           f"{'margTV':>8}  {'RMSE':>8}  {'Corr':>7}  {'Acc':>7}  {'Time':>7}")
    print(f"\n{hdr}")
    print("─" * len(hdr))
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()

        if epoch % CFG.EVAL_EVERY == 0 or epoch == 1 or epoch == epochs:
            raw = getattr(model, "_orig_mod", model)
            r = evaluate(raw, dist_loader, device, k_samples=k_samples)
            model.train()
            history["epoch"].append(epoch)
            history["train_loss"].append(loss)
            for k in ("validity", "distinct", "marginal_tv", "rmse", "accuracy", "correct"):
                history[k].append(r[k])
            print(f"{epoch:>5}  {loss:>10.5f}  {r['validity']*100:>7.2f}%  "
                  f"{r['distinct']*100:>7.2f}%  {r['marginal_tv']:>8.4f}  "
                  f"{r['rmse']:>8.5f}  {r['correct']*100:>6.2f}%  "
                  f"{r['accuracy']*100:>6.2f}%  {time.time()-t0:>6.0f}s")

    print("─" * len(hdr))
    print(f"Done ({time.time()-t0:.0f}s)\n")
    return history
