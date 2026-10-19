"""
model.py
────────
Neural network architecture.

DigitEncoder     : CNN  [B, N, 1, 28, 28] → [B, N, D_ENC]
VelocityTransformer : Transformer  [B, N, D_ENC] → [B, N, N]  (log-score / velocity)
FlowModel        : full model wrapping encoder + transformer + ODE integration
"""

import torch
import torch.nn as nn
from config import CFG
from flow_utils import (tangent_project, sample_tangent_noise,
                         make_noisy_X0, compute_cfm_loss, couple)


# ══════════════════════════════════════════════════════════════════
# 1.  CNN DIGIT ENCODER
# ══════════════════════════════════════════════════════════════════

class DigitEncoder(nn.Module):
    """
    Small CNN that independently encodes each digit image.

    Input : [B, N, 1, 28, 28]
    Output: [B, N, D_ENC]

    All N images in a batch are processed in a single fused call by
    reshaping to [B*N, 1, 28, 28] before the conv layers.
    """

    def __init__(self, d_enc: int = CFG.D_ENC):
        super().__init__()
        self.d_enc = d_enc
        self.cnn = nn.Sequential(
            # Block 1: 28×28 → 14×14
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 14×14 → 7×7
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 7×7 → 1×1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),                            # [B*N, 64]
        )
        self.proj = nn.Sequential(
            nn.Linear(64, d_enc),
            nn.LayerNorm(d_enc),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, N, 1, 28, 28]
        returns: [B, N, D_ENC]
        """
        B, N = images.shape[:2]
        flat = images.reshape(B * N, 1, CFG.IMG_SIZE, CFG.IMG_SIZE)
        feat = self.cnn(flat)                        # [B*N, 64]
        enc  = self.proj(feat)                       # [B*N, D_ENC]
        return enc.reshape(B, N, self.d_enc)


# ══════════════════════════════════════════════════════════════════
# 2.  TRANSFORMER VELOCITY / SCORE NETWORK
# ══════════════════════════════════════════════════════════════════

class VelocityTransformer(nn.Module):
    """
    Transformer that maps per-digit encodings to an N×N log-score matrix.

    Architecture:
      1. Project D_ENC → D_MODEL
      2. Add learnable position embeddings (slot identity)
      3. N layers of pre-LN self-attention + FFN
      4. Bilinear output head: V[i,j] = <h_i, rank_emb[j]> / sqrt(d)

    The bilinear head is the key inductive bias:
      h_i   encodes "what element i looks like after seeing all others"
      r_j   encodes "what rank-j looks like"
      V[i,j] = similarity between element i and rank j

    For the flow model, V is passed through tangent_project() to get
    a valid velocity in T(DS_n).
    """

    def __init__(self,
                 n:        int = CFG.N,
                 d_enc:    int = CFG.D_ENC,
                 d_model:  int = CFG.D_MODEL,
                 nlayers:  int = CFG.NLAYERS,
                 nheads:   int = CFG.NHEADS,
                 ffn_mult: int = CFG.FFN_MULT,
                 dropout:  float = CFG.DROPOUT,
                 use_xt:   bool  = False,
                 use_t:    bool  = False):
        super().__init__()
        self.n = n
        self.use_xt = use_xt
        self.use_t  = use_t
        if use_t:
            # time embedding added to the state token
            self.t_proj = nn.Sequential(
                nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        if use_xt:
            # current state X_t as one extra token (same design in all three tasks)
            self.xt_proj = nn.Sequential(
                nn.Linear(n * n, d_model),
                nn.LayerNorm(d_model),
            )
        assert d_model % nheads == 0, \
            f"D_MODEL={d_model} must be divisible by NHEADS={nheads}"

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_enc, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable slot-position embeddings
        # (tells the network which sequence slot each token occupies)
        self.pos_emb = nn.Embedding(n, d_model)

        # Transformer encoder — pre-LayerNorm for training stability
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nheads,
            dim_feedforward = d_model * ffn_mult,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,    # Pre-LN: more stable than post-LN
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers          = nlayers,
            enable_nested_tensor= False,
        )
        self.final_ln = nn.LayerNorm(d_model)

        # Rank embeddings and scale
        self.rank_emb = nn.Embedding(n, d_model)
        self._scale   = d_model ** -0.5

    def forward(self, enc: torch.Tensor, X_t: torch.Tensor = None,
                t: torch.Tensor = None) -> torch.Tensor:
        """
        enc: [B, N, D_ENC]  — per-digit CNN embeddings
        X_t: [B, N, N]      — current state (only if use_xt)
        returns: [B, N, N]  — raw score matrix (log_alpha / velocity)
        """
        B = enc.shape[0]

        h = self.input_proj(enc)                          # [B, N, D_MODEL]
        h = h + self.pos_emb.weight.unsqueeze(0)          # [B, N, D_MODEL]
        if self.use_xt:
            xt = self.xt_proj(X_t.reshape(B, -1).to(h.dtype))
            if self.use_t:
                xt = xt + self.t_proj(t.reshape(B, 1).to(h.dtype))
            xt = xt.unsqueeze(1)
            h  = torch.cat([xt, h], dim=1)                 # [B, N+1, D_MODEL]
        h = self.transformer(h)                            # [B, N, D_MODEL]
        h = self.final_ln(h)                               # [B, N, D_MODEL]
        if self.use_xt:
            h = h[:, 1:, :]                                # drop state token

        # Bilinear: score[i,j] = <h_i, rank_emb_j> / sqrt(d)
        scores = h @ self.rank_emb.weight.T * self._scale  # [B, N, N]
        return scores


# ══════════════════════════════════════════════════════════════════
# 3.  FULL FLOW MODEL
# ══════════════════════════════════════════════════════════════════

class FlowModel(nn.Module):
    """
    PermFlow for visual sorting.

    use_xt=True (PermFlow): the velocity network sees (context, X_t, t); the
    path is X_t = (1-t) X~0 + t P*, the regression target P* - X~0, and P* is
    chosen by the coupling ("nearest", "calibrated", or the ablation "random").
    use_xt=False (state-independent ablation): v = f(context), constant along
    a trajectory, trained with the nearest coupling on the mean direction
    P* - X0 (compute_cfm_loss in flow_utils.py).
    """

    def __init__(self,
                 n:        int   = CFG.N,
                 d_enc:    int   = CFG.D_ENC,
                 n_steps:  int   = CFG.N_STEPS,
                 sigma_0:  float = CFG.SIGMA_0,
                 use_xt:   bool  = True,
                 coupling: str   = "nearest",
                 project:  bool  = True):
        super().__init__()
        assert use_xt or (coupling == "nearest" and project), \
            "the state-independent ablation uses the nearest coupling with projection"
        self.n        = n
        self.n_steps  = n_steps
        self.sigma_0  = sigma_0
        self.use_xt   = use_xt
        self.coupling = coupling
        self.project  = project
        self.encoder  = DigitEncoder(d_enc)
        self.velocity = VelocityTransformer(n=n, d_enc=d_enc, use_xt=use_xt,
                                            use_t=use_xt)

    def _proj(self, U):
        return tangent_project(U) if self.project else U

    # ── Training ──────────────────────────────────────────────────
    def cfm_loss(self, images, P_star_a, P_star_b, alpha=None):
        """CFM loss; images [B,N,1,28,28], P_star_a / P_star_b [B,N,N], alpha [B]."""
        enc = self.encoder(images)                              # [B, N, D_ENC]
        if not self.use_xt:
            return compute_cfm_loss(velocity_net_out=self.velocity(enc),
                                    P_star_a=P_star_a, P_star_b=P_star_b,
                                    sigma=self.sigma_0, n=self.n)
        B, dev, dt_ = images.shape[0], images.device, P_star_a.dtype
        X0, X0_noisy, eps = make_noisy_X0(B, self.n, self.sigma_0, dev, dt_)
        P   = couple(X0_noisy, eps, P_star_a, P_star_b, alpha, self.coupling)
        t   = torch.rand(B, device=dev, dtype=dt_).view(B, 1, 1)
        X_t = (1. - t) * X0_noisy + t * P
        v   = self._proj(self.velocity(enc, X_t, t.view(B)))
        return ((v - (P - X0_noisy)) ** 2).mean()

    # ── Inference ─────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self,
               images:    torch.Tensor,
               k_samples: int = CFG.K_SAMPLES) -> list:
        """
        Run the ODE k_samples times from different noisy starts.

        Each run uses a fresh ε → different ODE trajectory → potentially
        different final permutation matrix.  Aggregating k_samples reveals
        the multimodal structure of the target distribution.

        Args:
            images:    [B, N, 1, 28, 28]
            k_samples: number of independent noise seeds

        Returns:
            list of k_samples tensors, each [B, N, N]
            (soft permutation matrices, before hungarian rounding)
        """
        B, n   = images.shape[0], self.n
        device = images.device
        dtype  = images.dtype
        dt     = 1.0 / self.n_steps

        # Encode once — shared across all k_samples
        enc = self.encoder(images)                       # [B, N, D_ENC]

        results = []
        for _ in range(k_samples):
            # Fresh noisy start for each sample
            X0_base = torch.full((B, n, n), 1./n, device=device, dtype=dtype)
            eps     = sample_tangent_noise(B, n, device, dtype)
            # eps already lives in T(DS_n); adding it to the centroid
            # preserves row/col sums = 1. Re-projecting the sum would
            # zero out the centroid (tangent_project((1/n)·11ᵀ) = 0).
            X = X0_base + self.sigma_0 * eps

            # Euler integration of the velocity field
            # Note: velocity does NOT depend on X (no data leakage risk here
            # because the flow is one-shot: v = f(encode(images)))
            if self.use_xt:
                for s in range(self.n_steps):
                    t = torch.full((B,), s * dt, device=device, dtype=dtype)
                    X = X + dt * self._proj(self.velocity(enc, X, t))
            else:
                v = self._proj(self.velocity(enc))       # [B, N, N]  (constant)
                for _ in range(self.n_steps):
                    X = X + dt * v

            results.append(X.detach())

        return results   # list of k tensors, each [B, N, N]

    def forward(self, images, P_star_a, P_star_b, alpha=None):
        """Alias for training — returns cfm_loss."""
        return self.cfm_loss(images, P_star_a, P_star_b, alpha)
