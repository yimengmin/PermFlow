"""
config.py
─────────
All hyperparameters in one place.
Import anywhere with:  from config import CFG
"""

class CFG:
    # ── Task ──────────────────────────────────────────────────────
    N          = 9      # sequence length (digits to sort)
    SEED       = 42

    # ── Dataset ───────────────────────────────────────────────────
    N_TRAIN    = 100000
    N_VAL      = 4000
    N_TEST_AMB = 2000   # ambiguous-only test set
    N_TEST_CLN = 2000   # clean-only test set
    FRAC_AMB   = 0.5    # fraction of training sequences that are ambiguous

    # Digit pairs used for blending.
    # Both digits in a pair are ≥ 3 apart so they occupy
    # genuinely different sorted positions in the sequence.
    BLEND_PAIRS = [
        (0, 4), (1, 5), (2, 6), (3, 7),
        (1, 6), (2, 7), (0, 5), (3, 8),
    ]

    # ── CNN encoder ───────────────────────────────────────────────
    IMG_SIZE   = 28
    D_ENC      = 64     # per-digit embedding dim out of CNN

    # ── Transformer ───────────────────────────────────────────────
    D_MODEL    = 128
    NHEADS     = 4
    NLAYERS    = 4
    FFN_MULT   = 4
    DROPOUT    = 0.10

    # ── OT-CFM flow ───────────────────────────────────────────────
    SIGMA_0    = 1.0    # ||X~0 - X0||_F (noise is normalised on the sphere of T)
    N_STEPS    = 10     # Euler ODE steps at inference
    K_SAMPLES  = 10     # stochastic draws for coverage/calibration eval

    # ── Optimisation ─────────────────────────────────────────────
    LR           = 3e-4
    WEIGHT_DECAY = 1e-4
    N_EPOCHS     = 300
    BATCH_SIZE   = 256
    N_WORKERS    = 2
    USE_AMP      = True
    GRAD_CLIP    = 1.0
    LR_MIN       = 1e-5  # cosine annealing floor


    # ── Output ────────────────────────────────────────────────────
    DATA_DIR   = "./data"
    OUT_DIR    = "./outputs"
    CKPT_PATH  = "./outputs/flow.pt"   # overwritten per run by train.py
