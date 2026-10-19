"""
data.py
───────
MNIST loading and dataset construction.

Each training sequence has N digits. With probability FRAC_AMB, one
digit is a BLENDED image: α·img(da) + (1-α)·img(db) where |da-db| ≥ 3.
This creates a bimodal label: both P*_a and P*_b are valid orderings.

For clean sequences P*_a == P*_b (no ambiguity).
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import CFG

try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("[data.py] torchvision not found — using synthetic digit data")


# ══════════════════════════════════════════════════════════════════
# 1.  LOAD MNIST IMAGES GROUPED BY DIGIT
# ══════════════════════════════════════════════════════════════════

def load_mnist_by_digit(train: bool = True) -> dict:
    """
    Returns:
        dict: digit (0-9) → FloatTensor [M, 1, 28, 28] in [0, 1]
    """
    if HAS_TORCHVISION:
        dset = datasets.MNIST(
            CFG.DATA_DIR, train=train, download=True,
            transform=transforms.ToTensor()
        )
        by_digit = {d: [] for d in range(10)}
        for img, label in dset:
            by_digit[label].append(img)              # img: [1, 28, 28]
        return {d: torch.stack(v) for d, v in by_digit.items() if v}
    else:
        # ── Synthetic fallback ────────────────────────────────────
        # Each "digit" is a 28×28 image whose brightness is
        # proportional to digit value. Ordering is preserved.
        rng = np.random.default_rng(CFG.SEED)
        n_per = 5000 if train else 1000
        by_digit = {}
        for d in range(10):
            base  = torch.full((n_per, 1, 28, 28), d / 9. * 0.8)
            noise = torch.from_numpy(
                rng.normal(0, 0.08, (n_per, 1, 28, 28)).astype(np.float32))
            by_digit[d] = (base + noise).clamp(0., 1.)
        return by_digit


# ══════════════════════════════════════════════════════════════════
# 2.  PERMUTATION MATRIX FROM A LIST OF DIGIT VALUES
# ══════════════════════════════════════════════════════════════════

def make_perm_matrix(digit_values: list) -> torch.Tensor:
    """
    Given digit values at each sequence position, return P where
    P[i, j] = 1 iff element i belongs at sorted rank j.

    Example: digit_values = [3, 1, 4]
      sorted order: 1 < 3 < 4  →  ranks = [1, 0, 2]
      P = [[0,1,0],   (element 0 goes to rank 1)
           [1,0,0],   (element 1 goes to rank 0)
           [0,0,1]]   (element 2 goes to rank 2)
    """
    n    = len(digit_values)
    idx  = sorted(range(n), key=lambda i: digit_values[i])
    rank = [0] * n
    for pos, orig in enumerate(idx):
        rank[orig] = pos
    P = torch.zeros(n, n)
    for i, r in enumerate(rank):
        P[i, r] = 1.
    return P


# ══════════════════════════════════════════════════════════════════
# 3.  FILLER DIGIT SELECTION
# ══════════════════════════════════════════════════════════════════

def choose_filler_digits(da: int, db: int, k: int, rng) -> list:
    """
    Choose k filler digits so that da and db land at DIFFERENT
    sorted positions. We ensure at least one digit falls strictly
    between da and db in value.

    Args:
        da, db: the two blend digits
        k:      number of filler digits needed (= N - 1)
        rng:    numpy rng
    """
    lo, hi     = min(da, db), max(da, db)
    candidates = [d for d in range(10) if d != da and d != db]

    between = [d for d in candidates if lo < d < hi]
    outside = [d for d in candidates if d <= lo or d >= hi]
    rng.shuffle(between)
    rng.shuffle(outside)

    # Include at least one digit between lo and hi if possible
    if between and k >= 1:
        n_between = int(rng.integers(1, min(len(between), k) + 1))
        fillers   = between[:n_between] + outside[:k - n_between]
    else:
        fillers = outside[:k]

    # Pad if we ran short (shouldn't happen for N=6, 10 digits)
    all_others = [d for d in range(10) if d not in (da, db)]
    while len(fillers) < k:
        d = all_others[int(rng.integers(len(all_others)))]
        if d not in fillers:
            fillers.append(d)
    return fillers[:k]


# ══════════════════════════════════════════════════════════════════
# 4.  DATASET
# ══════════════════════════════════════════════════════════════════

class MNISTSortingDataset(Dataset):
    """
    Each item returns:
        images   : [N, 1, 28, 28]  float32 in [0,1]
        P_star_a : [N, N]           target permutation for blend-as-da
        P_star_b : [N, N]           target permutation for blend-as-db
        alpha    : scalar float     blend weight (P*_a prob ~ alpha)
        is_ambig : bool             True if this sequence has a blend

    For clean sequences (is_ambig=False):  P_star_a == P_star_b.
    For ambiguous sequences:               P_star_a != P_star_b.
    """

    def __init__(self,
                 mnist_by_digit: dict,
                 n_samples: int,
                 n: int,
                 rng,
                 include_ambiguous: bool = True,
                 frac_ambiguous: float = CFG.FRAC_AMB):
        self.n     = n
        self.items = []

        for _ in range(n_samples):
            is_amb = include_ambiguous and (rng.random() < frac_ambiguous)

            if is_amb:
                self.items.append(
                    self._make_ambiguous(mnist_by_digit, n, rng))
            else:
                self.items.append(
                    self._make_clean(mnist_by_digit, n, rng))

    # ── Ambiguous sequence ────────────────────────────────────────
    def _make_ambiguous(self, mnist_by_digit, n, rng) -> dict:
        # Pick blend pair
        pair_idx = int(rng.integers(len(CFG.BLEND_PAIRS)))
        da, db   = CFG.BLEND_PAIRS[pair_idx]

        # Alpha ~ Beta(2,2) concentrated around 0.5, clipped to (0.2, 0.8)
        alpha = float(np.clip(rng.beta(2., 2.), 0.2, 0.8))

        # Blend image: position 0 is always the ambiguous slot
        img_a = mnist_by_digit[da][int(rng.integers(len(mnist_by_digit[da])))]
        img_b = mnist_by_digit[db][int(rng.integers(len(mnist_by_digit[db])))]
        blend = alpha * img_a + (1. - alpha) * img_b    # [1, 28, 28]

        # N-1 filler digits
        fillers = choose_filler_digits(da, db, n - 1, rng)
        filler_imgs = [
            mnist_by_digit[d][int(rng.integers(len(mnist_by_digit[d])))]
            for d in fillers
        ]

        images = [blend] + filler_imgs                  # list of [1,28,28]

        # Both valid permutations
        P_a = make_perm_matrix([da] + fillers)
        P_b = make_perm_matrix([db] + fillers)

        return dict(images=torch.stack(images), P_a=P_a, P_b=P_b,
                    alpha=alpha, is_ambig=True)

    # ── Clean sequence ────────────────────────────────────────────
    def _make_clean(self, mnist_by_digit, n, rng) -> dict:
        # Choose n distinct digits, shuffle presentation order
        digits = sorted(rng.choice(10, size=n, replace=False).tolist())
        perm   = rng.permutation(n).tolist()
        pres   = [digits[perm[i]] for i in range(n)]   # shuffled order

        images = [
            mnist_by_digit[d][int(rng.integers(len(mnist_by_digit[d])))]
            for d in pres
        ]
        P = make_perm_matrix(pres)
        return dict(images=torch.stack(images), P_a=P, P_b=P,
                    alpha=1.0, is_ambig=False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return (
            it["images"],                                        # [N,1,28,28]
            it["P_a"],                                           # [N,N]
            it["P_b"],                                           # [N,N]
            torch.tensor(it["alpha"],    dtype=torch.float32),  # scalar
            torch.tensor(it["is_ambig"], dtype=torch.bool),     # scalar
        )


# ══════════════════════════════════════════════════════════════════
# 5.  DATALOADER FACTORY
# ══════════════════════════════════════════════════════════════════

def build_loaders(seed: int = CFG.SEED):
    """
    Build all four DataLoaders:
        train_loader  — mixed (50% ambiguous)
        val_loader    — mixed
        amb_loader    — 100% ambiguous  (for coverage/calibration eval)
        cln_loader    — 100% clean      (for standard accuracy eval)
    """
    print("Loading MNIST images...")
    mnist_train = load_mnist_by_digit(train=True)
    mnist_test  = load_mnist_by_digit(train=False)

    rng = np.random.default_rng(seed)

    print("Building datasets...")
    train_dset = MNISTSortingDataset(mnist_train, CFG.N_TRAIN, CFG.N, rng,
                                     include_ambiguous=True,
                                     frac_ambiguous=CFG.FRAC_AMB)
    val_dset   = MNISTSortingDataset(mnist_test,  CFG.N_VAL,   CFG.N, rng,
                                     include_ambiguous=True,
                                     frac_ambiguous=CFG.FRAC_AMB)
    amb_dset   = MNISTSortingDataset(mnist_test,  CFG.N_TEST_AMB, CFG.N, rng,
                                     include_ambiguous=True,
                                     frac_ambiguous=1.0)
    cln_dset   = MNISTSortingDataset(mnist_test,  CFG.N_TEST_CLN, CFG.N, rng,
                                     include_ambiguous=False,
                                     frac_ambiguous=0.0)

    kw = dict(
        batch_size        = CFG.BATCH_SIZE,
        num_workers       = CFG.N_WORKERS,
        pin_memory        = True,
        persistent_workers= (CFG.N_WORKERS > 0),
    )
    train_loader = DataLoader(train_dset, shuffle=True,  **kw)
    val_loader   = DataLoader(val_dset,   shuffle=False, **kw)
    amb_loader   = DataLoader(amb_dset,   shuffle=False, **kw)
    cln_loader   = DataLoader(cln_dset,   shuffle=False, **kw)

    print(f"  Train: {len(train_dset):>6,}  |  Val: {len(val_dset):>5,}  |"
          f"  Amb: {len(amb_dset):>5,}  |  Cln: {len(cln_dset):>5,}")

    return train_loader, val_loader, amb_loader, cln_loader
