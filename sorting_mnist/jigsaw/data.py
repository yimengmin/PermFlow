"""
data.py  —  jigsaw datasets with a precomputed ambiguity partition
══════════════════════════════════════════════════════════════════

Pipeline, per image, computed once offline and cached to disk:

  1.  load the image in [0, 1]
  2.  add i.i.d. Gaussian pixel noise N(0, σ²) and round back to 8 bits
                                              (σ = CFG.SIGMA_PIX, CFG.QUANTISE)
  3.  crop to g·p × g·p with p = image_size // g and cut into n = g² patches
      in raster order                              (their `chunk_image`)
  4.  draw τ ~ Uniform(S_n) and shuffle:  obs[i] = orig[τ(i)]
  5.  partition the patches by *exact* content → slot_blk, and hence 𝒪

Step 2 comes before step 3 on purpose: the noise is a property of the image,
so the observed patches carry it and exact equality between background
patches is destroyed — which is precisely what their NoisyMNIST does and why
σ = 0.01 forces |𝒪| = 1.

A dataset item is

    patches   [n, C, p, p] float32  the shuffled patches — the model input
    gt_slot   [n]          int64    gt_slot[i] = τ(i), one member of 𝒪
    obs_blk   [n]          int64    content class of observed patch i
    slot_blk  [n]          int64    content class of the patch at slot j
    log_orbit scalar       float32  log|𝒪| in nats

The original (unshuffled) patches are not stored: orig[τ(i)] = obs[i]
recovers them exactly, and metrics.py does so.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import CFG
from orbit import orbit_log_size, partition_by_content

try:
    from torchvision import datasets
    HAS_TORCHVISION = True
except ImportError:                                     # pragma: no cover
    HAS_TORCHVISION = False


# ══════════════════════════════════════════════════════════════════
# 1.  RAW IMAGES
# ══════════════════════════════════════════════════════════════════

def load_images(dataset: str, train: bool) -> torch.Tensor:
    """
    Returns [N, C, H, W] float32 in [0, 1], in the benchmark's own splits:
        MNIST     60,000 / 10,000
    """
    if not HAS_TORCHVISION:                             # pragma: no cover
        raise RuntimeError("torchvision is required to load MNIST")

    if dataset == "mnist":
        dset = datasets.MNIST(CFG.DATA_DIR, train=train, download=True)
        x = dset.data.unsqueeze(1).float() / 255.0           # [N,1,28,28]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    return x.contiguous()


# ══════════════════════════════════════════════════════════════════
# 2.  CHUNKING  (identical to SymmetricDiffusers' chunk_image)
# ══════════════════════════════════════════════════════════════════

def add_pixel_noise(images: torch.Tensor, sigma: float, rng,
                    quantise: bool = None) -> torch.Tensor:
    """
    Their NoisyMNIST construction: i.i.d. Gaussian noise on every pixel.

    The image is then rounded back to 8 bits (CFG.QUANTISE), because MNIST is
    an 8-bit image and because that is what makes σ a *continuous*
    knob.  In float32 any σ > 0 breaks byte-equality immediately and |𝒪| drops
    from ∏ m_b! to 1 between σ = 0 and σ = 10⁻⁴; quantised, a background pixel
    only leaves 0 once the noise exceeds half a level, so the ambiguity decays
    smoothly over σ ∈ [10⁻⁴, 10⁻²].  Both endpoint arms are unchanged by it.
    """
    if sigma <= 0:
        return images
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, images.shape).astype(np.float32))
    out = images + noise
    q = CFG.QUANTISE if quantise is None else quantise
    if q:
        L = CFG.QUANT_LEVELS
        out = torch.round(out.clamp(0.0, 1.0) * L) / L
    return out


def chunk_images(images: torch.Tensor, g: int) -> torch.Tensor:
    """
    [N, C, H, W] → [N, g², C, p, p] with p = H // g, raster order, remainder
    on the right/bottom discarded.
    """
    N, C, H, W = images.shape
    p = H // g
    x = images[:, :, : g * p, : g * p]
    x = x.reshape(N, C, g, p, g, p)                # N,C,gh,p,gw,p
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()   # N,gh,gw,C,p,p
    return x.reshape(N, g * g, C, p, p)


# ══════════════════════════════════════════════════════════════════
# 3.  DATASET
# ══════════════════════════════════════════════════════════════════

class JigsawDataset(Dataset):
    """
    Shuffled patches plus the ambiguity partition of each instance.
    """

    def __init__(self,
                 images:    torch.Tensor,
                 g:         int,
                 sigma_pix: float,
                 seed:      int,
                 verbose:   bool = True,
                 quantise:  bool = None,
                 dataset:   str = None):
        self.g = g
        self.n = g * g

        rng = np.random.default_rng(seed)

        # ── pixel noise, then chunk ──────────────────────────────
        images = add_pixel_noise(images, sigma_pix, rng, quantise)
        patches_orig = chunk_images(images, g)          # [N, n, C, p, p]
        N = patches_orig.shape[0]

        # ── shuffle ──────────────────────────────────────────────
        tau = np.stack([rng.permutation(self.n) for _ in range(N)])   # [N,n]
        tau_t = torch.from_numpy(tau).long()
        idx = tau_t.view(N, self.n, 1, 1, 1).expand_as(patches_orig)
        self.patches = torch.gather(patches_orig, 1, idx).contiguous()
        self.gt_slot = tau_t                            # gt_slot[i] = τ(i)

        # ── exact-content partition (offline, once) ──────────────
        orig_np  = patches_orig.numpy()
        slot_blk = np.empty((N, self.n), dtype=np.int64)
        log_orb  = np.empty(N, dtype=np.float32)
        for k in range(N):
            slot_blk[k] = partition_by_content(orig_np[k])
            log_orb[k]  = orbit_log_size(slot_blk[k])
        self.slot_blk  = torch.from_numpy(slot_blk)
        self.obs_blk   = torch.gather(self.slot_blk, 1, tau_t)
        self.log_orbit = torch.from_numpy(log_orb)

        if verbose:
            amb = (self.log_orbit > 0).float().mean().item()
            print(f"    n={self.n:<3d} images={N:>6,}  σ={sigma_pix:g}  "
                  f"ambiguous={amb*100:5.1f}%  "
                  f"mean log|O|={self.log_orbit.mean():.3f}")

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, i):
        return (self.patches[i], self.gt_slot[i],
                self.obs_blk[i], self.slot_blk[i], self.log_orbit[i])


# ══════════════════════════════════════════════════════════════════
# 4.  CACHED CONSTRUCTION
# ══════════════════════════════════════════════════════════════════

def _cache_path(dataset: str, split: str, g: int,
                sigma_pix: float, seed: int, cap: int) -> str:
    sig = f"{sigma_pix:g}".replace(".", "p").replace("-", "m")
    q   = "q8" if CFG.QUANTISE else "f32"
    return os.path.join(
        CFG.CACHE_DIR,
        f"jigsaw_{dataset}_{split}_g{g}_sig{sig}_{q}_seed{seed}_n{cap}.pt")


def build_dataset(dataset: str, split: str, g: int, sigma_pix: float,
                  seed: int, cap: int = -1, use_cache: bool = True):
    """
    Build (or load from cache) one split.  `cap` truncates the split; -1 keeps
    all of it.
    """
    os.makedirs(CFG.CACHE_DIR, exist_ok=True)
    path = _cache_path(dataset, split, g, sigma_pix, seed, cap)

    if use_cache and os.path.exists(path):
        blob = torch.load(path, weights_only=False)
        print(f"  [cache] {os.path.basename(path)}")
        dset = JigsawDataset.__new__(JigsawDataset)
        dset.__dict__.update(blob)
        return dset

    images = load_images(dataset, train=(split == "train"))
    if cap > 0:
        images = images[:cap]
    dset = JigsawDataset(images, g, sigma_pix,
                         seed=seed + (0 if split == "train" else 10_000),
                         dataset=dataset)
    if use_cache:
        torch.save(dset.__dict__, path)
    return dset


def build_loaders(dataset: str = None, g: int = None, sigma_pix: float = None,
                  seed: int = None, batch_size: int = None,
                  use_cache: bool = True):
    """
    Returns (train_loader, test_loader, test_dist_loader).

    test_dist_loader is a prefix of the test split (CFG.N_TEST_DIST images)
    used for the K-sample distributional metrics, which cost K ODE
    integrations per image.
    """
    dataset   = dataset   or CFG.DATASET
    g         = g         or CFG.GRID
    seed      = seed      if seed      is not None else CFG.DATA_SEED
    sigma_pix = sigma_pix if sigma_pix is not None else CFG.SIGMA_PIX
    batch_size = batch_size or CFG.BATCH_SIZE

    train_cap = CFG.N_TRAIN
    test_cap  = CFG.N_TEST

    print("Building jigsaw datasets...")
    train_dset = build_dataset(dataset, "train", g, sigma_pix, seed,
                               train_cap, use_cache)
    test_dset  = build_dataset(dataset, "test",  g, sigma_pix, seed,
                               test_cap,  use_cache)

    n_dist    = min(CFG.N_TEST_DIST, len(test_dset))
    dist_dset = torch.utils.data.Subset(test_dset, range(n_dist))

    kw = dict(batch_size=batch_size, num_workers=CFG.N_WORKERS,
              pin_memory=True, persistent_workers=(CFG.N_WORKERS > 0))
    train_loader = DataLoader(train_dset, shuffle=True,  drop_last=True,  **kw)
    test_loader  = DataLoader(test_dset,  shuffle=False, drop_last=False, **kw)
    dist_loader  = DataLoader(dist_dset,  shuffle=False, drop_last=False, **kw)

    print(f"  Train: {len(train_dset):>6,}  |  Test: {len(test_dset):>6,}  |"
          f"  Test(dist): {n_dist:>5,}")
    return train_loader, test_loader, dist_loader
