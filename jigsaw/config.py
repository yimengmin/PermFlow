"""
config.py  —  all hyperparameters for the jigsaw-puzzle ambiguity experiment
════════════════════════════════════════════════════════════════════════════

TASK
────
An image is cut into an n = g×g grid of patches, the patches are shuffled by
a permutation drawn uniformly from S_n, and the model must predict the
permutation that restores the original arrangement.  This is the jigsaw
benchmark of SymmetricDiffusers (Zhang, Yang & Liao, ICLR 2025), unchanged:

    MNIST     g ∈ {4,5,6},  60,000 train / 10,000 test

Patch geometry follows their `chunk_image()` exactly: patch side is
`image_size // g` and the right/bottom remainder is discarded (e.g. g=6 on
MNIST uses the top-left 24×24 of the 28×28 image).  No resizing.

THE AMBIGUITY
─────────────
Patches i and j are *exchangeable* when their pixel content is exactly equal.
This partitions the n patches into classes B₁,…,B_r of sizes m₁,…,m_r, and
the set of permutations giving a perfect reconstruction is an orbit

    𝒪 = { P : P sends each patch to a slot whose original content equals it }
    |𝒪| = ∏_b m_b!

On plain MNIST the background is exactly zero-valued, so a 5×5 or 6×6 grid
typically contains a large class of all-zero patches and |𝒪| is large.  Their
Noisy MNIST adds i.i.d. Gaussian noise with σ = 0.01 to every pixel, which
destroys exact equality and forces |𝒪| = 1 — stated in their README as
ensuring background pieces are distinctive.  σ is therefore *their* knob, and
it interpolates continuously between a fully ambiguous and a fully identified
task.

TWO ARMS
────────
    Published setting   Noisy MNIST, σ = 0.01   |𝒪| = 1
    Orbit arm           plain MNIST, σ = 0      |𝒪| = ∏ m_b!

Set the arm with  --sigma_pix  (see train.py --help).
"""


class CFG:
    # ── Task ──────────────────────────────────────────────────────
    DATASET    = "mnist"
    GRID       = 4         # g — pieces per side; n = g*g patches
    SIGMA_PIX  = 0.0       # data-space Gaussian pixel noise σ
                           #   0.00 → main arm (ambiguous, |𝒪| = ∏ m_b!)
                           #   0.01 → their NoisyMNIST control (|𝒪| = 1)

    # Keep the noised image in the image domain: round back to 8 bits, as
    # MNIST natively is.  This matters only for the σ sweep.
    #
    # In float32 *any* σ > 0 destroys byte-equality outright, so |𝒪| jumps
    # from ∏ m_b! to 1 between σ = 0 and σ = 10⁻⁴ and the ablation degenerates
    # into a step at 0⁺.  With 8-bit re-quantisation a background pixel only
    # leaves 0 when the noise exceeds half a level (1/510), so the fraction of
    # flipped pixels is 2Φ(−1/(510σ)) and the ambiguity decays smoothly across
    # exactly the specified sweep σ ∈ {0, 10⁻⁴, 10⁻³, 10⁻²}.
    #
    # Neither endpoint arm is affected: at σ = 0 quantisation is the identity,
    # and at σ = 0.01 (their control) 84% of pixels flip, so |𝒪| = 1 either
    # way.
    QUANTISE   = True
    QUANT_LEVELS = 255     # 8-bit

    SEED       = 42        # model init / training noise
    DATA_SEED  = 42        # shuffle + pixel noise; fixed across model seeds so
                           # every seed is trained and tested on the same data

    # ── Dataset sizes ─────────────────────────────────────────────
    # MNIST 60,000/10,000 is the benchmark's own split, applied in data.py; N_TRAIN/N_TEST below
    # only *cap* them (-1 = use the whole split).
    N_TRAIN    = -1
    N_TEST     = -1
    # The K-sample distributional metrics (validity, distinct modes, marginal
    # TV) cost K ODE integrations per image, so they run on a fixed prefix of
    # the test set.  Single-sample metrics (accuracy, RMSE, MAE, Kendall-τ)
    # always use the full test split.
    N_TEST_DIST = 2000

    # ── Patch encoder (matched to SymmetricDiffusers' UnscrambleMnistCNN) ──
    #   conv(32, 3×3, s1, p1) → BatchNorm → ReLU → MaxPool(2) → 2 FC → d_model
    CNN_CHANNELS = 32
    CNN_KERNEL   = 3
    CNN_STRIDE   = 1
    CNN_PADDING  = 1

    # ── Transformer (matched: 7 layers, 8 heads, d=128, FFN 512, drop 0.1) ──
    D_MODEL    = 128
    NHEADS     = 8
    NLAYERS    = 7
    D_FFN      = 512
    DROPOUT    = 0.10
    TOKEN_POS  = True      # learnable position embedding on the patch tokens

    # ── OT-CFM flow ───────────────────────────────────────────────
    SIGMA_0    = 1.0       # flow-space noise magnitude on X₀
    N_STEPS    = 10        # Euler ODE steps at inference (S)
    K_SAMPLES  = 10        # trajectories drawn per image (K)

    # ── Optimisation (identical to the sorting task) ──────────────
    LR           = 3e-4
    WEIGHT_DECAY = 1e-4
    N_EPOCHS     = 300
    BATCH_SIZE   = 256
    N_WORKERS    = 2
    USE_AMP      = True
    GRAD_CLIP    = 1.0
    LR_MIN       = 1e-5    # cosine annealing floor

    EVAL_EVERY   = 10      # epochs between validation evaluations

    # ── Paths ─────────────────────────────────────────────────────
    DATA_DIR   = "./data"               # MNIST is downloaded here
    CACHE_DIR  = "./cache"              # precomputed patches + partitions
    OUT_DIR    = "./outputs"

    # ── Derived ───────────────────────────────────────────────────
    @classmethod
    def image_size(cls) -> int:
        return 28

    @classmethod
    def in_channels(cls) -> int:
        return 1

    @classmethod
    def n(cls) -> int:
        return cls.GRID * cls.GRID

    @classmethod
    def patch_size(cls) -> int:
        """Their convention: floor-divide and discard the remainder."""
        return cls.image_size() // cls.GRID

    @classmethod
    def tag(cls) -> str:
        """
        Checkpoint / result tag (σ₀ is appended only when it differs from 1).
        """
        sig = f"{cls.SIGMA_PIX:g}".replace(".", "p").replace("-", "m")
        t   = f"{cls.DATASET}_g{cls.GRID}_sig{sig}"
        if abs(cls.SIGMA_0 - 1.0) > 1e-9:
            t += "_s0" + f"{cls.SIGMA_0:g}".replace(".", "p")
        if cls.SEED != 42:
            t += f"_seed{cls.SEED}"
        return t
