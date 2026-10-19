"""
trainer.py
──────────
Training loop for the FlowModel.

Handles:
  - AMP (automatic mixed precision)
  - Gradient clipping
  - Cosine annealing LR schedule
  - Periodic evaluation (clean acc + ambiguous metrics)
  - History dict for plotting
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import CFG
from metrics import clean_accuracy, ambiguous_metrics


# ══════════════════════════════════════════════════════════════════
# 1.  SINGLE EPOCH
# ══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scaler, device) -> float:
    """
    One full pass over the training set.

    Each batch:
      1. Encode images (CNN)
      2. Compute OT-CFM loss (noisy X0 + OT coupling + velocity MSE)
      3. Backprop with AMP + grad clip

    Returns:
        mean CFM loss over the epoch
    """
    model.train()
    total_loss = 0.

    for images, P_a, P_b, alpha, is_ambig in loader:
        images = images.to(device, non_blocking=True)
        P_a    = P_a.to(device,    non_blocking=True)
        P_b    = P_b.to(device,    non_blocking=True)
        alpha  = alpha.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda",
                                 enabled=CFG.USE_AMP and device.type == "cuda"):
            loss = model.cfm_loss(images, P_a, P_b, alpha)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════
# 2.  FULL TRAINING RUN
# ══════════════════════════════════════════════════════════════════

def train(model, train_loader, cln_loader, amb_loader, device) -> dict:
    """
    Train for CFG.N_EPOCHS epochs and return the full history dict.

    Evaluation every 10 epochs (or epoch 1) on:
      - clean_acc   : exact-match on unambiguous sequences (1 ODE sample)
      - coverage    : fraction of ambiguous inputs where both modes found
      - calibration : |freq(mode_a) - alpha|

    Args:
        model:        FlowModel
        train_loader: DataLoader (mixed ambiguous + clean)
        cln_loader:   DataLoader (clean only, for accuracy eval)
        amb_loader:   DataLoader (ambiguous only, for coverage eval)
        device:       torch.device

    Returns:
        history dict with lists indexed by epoch
    """
    optimizer = optim.AdamW(model.parameters(),
                             lr=CFG.LR,
                             weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.N_EPOCHS, eta_min=CFG.LR_MIN)
    scaler    = torch.amp.GradScaler(
        "cuda", enabled=CFG.USE_AMP and device.type == "cuda")

    history = {
        "epoch"      : [],
        "train_loss" : [],
        "clean_acc"  : [],
        "coverage"   : [],
        "calibration": [],
        "any_correct": [],
    }

    # Print header
    print(f"\n{'Ep':>5}  {'Loss':>9}  {'ClnAcc':>8}  "
          f"{'Cov@%d' % CFG.K_SAMPLES:>8}  {'CalErr':>8}  {'Time':>7}")
    print("─" * 62)

    t0 = time.time()

    for epoch in range(1, CFG.N_EPOCHS + 1):

        # ── Train ────────────────────────────────────────────────
        loss = train_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()

        # ── Evaluate every 10 epochs (+ epoch 1) ─────────────────
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            cln_acc = clean_accuracy(model, cln_loader, device)
            amb     = ambiguous_metrics(model, amb_loader, device,
                                        k_samples=CFG.K_SAMPLES)
            model.train()

            history["epoch"].append(epoch)
            history["train_loss"].append(loss)
            history["clean_acc"].append(cln_acc)
            history["coverage"].append(amb["coverage"])
            history["calibration"].append(amb["calibration"])
            history["any_correct"].append(amb["any_correct"])

            elapsed = time.time() - t0
            print(f"{epoch:>5}  {loss:>9.4f}  {cln_acc*100:>7.1f}%  "
                  f"{amb['coverage']*100:>7.1f}%  "
                  f"{amb['calibration']:>8.4f}  "
                  f"{elapsed:>6.0f}s")

    print("─" * 62)
    print(f"Training complete  ({time.time()-t0:.0f}s total)\n")

    return history
