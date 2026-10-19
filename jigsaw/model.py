"""
model.py  —  patch encoder + X_t-conditioned velocity transformer
═════════════════════════════════════════════════════════════════

ENCODER  (capacity-matched to SymmetricDiffusers' UnscrambleMnistCNN)
─────────────────────────────────────────────────────────────────────
Per patch, independently:

    conv(32, 3×3, s1, p1) → BatchNorm → ReLU → MaxPool(2) → flatten
    → Linear(d_conv, (d_conv+d)/2) → ReLU → Linear(·, d)        d = 128

exactly their layer stack and their hidden width, so the comparison is not
confounded by encoder capacity.

VELOCITY NETWORK
────────────────
The velocity is conditioned on the current state X_t and the time t:

So:  flatten X_t → project to d_model → prepend as a state token to the n
patch embeddings → 7-layer transformer (8 heads, d 128, FFN 512, drop 0.1) →
drop the state token → bilinear head against learnable slot embeddings →
apply C, so the velocity has zero row and column sums and the iterate stays
in the unit-row/column-sum affine space A_n.  No non-negativity is imposed at
any point — X_t may have negative entries throughout.

    v(X_t) = C · [ h(X_t)ᵀ E_slot / √d ] · C,      C = I − (1/n)11ᵀ
"""

import torch
import torch.nn as nn

from config import CFG
from flow_utils import (cfm_loss, make_noisy_X0, orbit_couple,
                        sample_tangent_noise, tangent_project)


# ══════════════════════════════════════════════════════════════════
# 1.  PATCH ENCODER
# ══════════════════════════════════════════════════════════════════

class PatchEncoder(nn.Module):
    """
    [B, n, C, p, p] → [B, n, d_model], all n·B patches in one fused call.
    """

    def __init__(self,
                 in_channels: int = 1,
                 patch_size:  int = 7,
                 d_model:     int = CFG.D_MODEL,
                 channels:    int = CFG.CNN_CHANNELS):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size  = patch_size
        self.d_model     = d_model

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, CFG.CNN_KERNEL,
                      CFG.CNN_STRIDE, CFG.CNN_PADDING),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # MaxPool(2) halves the patch (floor), matching their arithmetic.
        d_conv  = (patch_size // 2) ** 2 * channels
        d_hid   = (d_conv + d_model) // 2
        self.mlp = nn.Sequential(
            nn.Linear(d_conv, d_hid),
            nn.ReLU(inplace=True),
            nn.Linear(d_hid, d_model),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, n = patches.shape[:2]
        x = patches.reshape(B * n, *patches.shape[2:])
        x = self.conv(x).flatten(1)
        x = self.mlp(x)
        return x.reshape(B, n, self.d_model)


# ══════════════════════════════════════════════════════════════════
# 2.  X_t-CONDITIONED VELOCITY TRANSFORMER
# ══════════════════════════════════════════════════════════════════

class VelocityTransformer(nn.Module):
    """
    (patch embeddings h, current position X_t) → velocity [B, n, n].

    Tokens:  [ xt_token | patch_0 | … | patch_{n-1} ]
    """

    def __init__(self,
                 n:        int   = 16,
                 d_model:  int   = CFG.D_MODEL,
                 nlayers:  int   = CFG.NLAYERS,
                 nheads:   int   = CFG.NHEADS,
                 d_ffn:    int   = CFG.D_FFN,
                 dropout:  float = CFG.DROPOUT,
                 token_pos: bool = CFG.TOKEN_POS,
                 use_xt:   bool  = True,
                 use_t:    bool  = False):
        super().__init__()
        self.n      = n
        self.use_xt = use_xt
        self.use_t  = use_t
        if use_t:
            # time embedding added to the state token; the path X_t = (1-t)X~0 + tP*'s
            # optimal field (P* − X)/(1 − t) depends on t explicitly
            self.t_proj = nn.Sequential(
                nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        if use_xt:
            self.xt_proj = nn.Sequential(
                nn.Linear(n * n, d_model),
                nn.LayerNorm(d_model),
            )
        self.token_pos = nn.Embedding(n, d_model) if token_pos else None

        layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nheads,
            dim_feedforward = d_ffn,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=nlayers, enable_nested_tensor=False)
        self.final_ln = nn.LayerNorm(d_model)

        self.slot_emb = nn.Embedding(n, d_model)
        self._scale   = d_model ** -0.5

    def forward(self,
                enc: torch.Tensor,                    # [B, n, d_model]
                X_t: torch.Tensor = None,             # [B, n, n]
                t:   torch.Tensor = None              # [B]
                ) -> torch.Tensor:
        B = enc.shape[0]
        h = enc
        if self.token_pos is not None:
            h = h + self.token_pos.weight.unsqueeze(0)

        if self.use_xt:
            xt_token = self.xt_proj(X_t.reshape(B, -1))
            if self.use_t:
                xt_token = xt_token + self.t_proj(t.reshape(B, 1).to(xt_token.dtype))
            xt_token = xt_token.unsqueeze(1)
            h = torch.cat([xt_token, h], dim=1)        # [B, n+1, d]

        h = self.final_ln(self.transformer(h))
        if self.use_xt:
            h = h[:, 1:, :]                            # drop the state token

        return h @ self.slot_emb.weight.T * self._scale     # [B, n, n]


# ══════════════════════════════════════════════════════════════════
# 3.  FULL FLOW MODEL
# ══════════════════════════════════════════════════════════════════

class FlowModel(nn.Module):
    """
    PermFlow for the jigsaw task.

    Training : cfm_loss(patches, obs_blk, slot_blk) → scalar
    Inference: sample(patches, K)                    → list of K [B, n, n]
    """

    def __init__(self,
                 n:            int   = 16,
                 in_channels:  int   = 1,
                 patch_size:   int   = 7,
                 d_model:      int   = CFG.D_MODEL,
                 n_steps:      int   = CFG.N_STEPS,
                 sigma_0:      float = CFG.SIGMA_0,
                 coupling:     str   = "penalised",
                 project:      bool  = True):
        super().__init__()
        # Path X_t = (1−t)X̃₀ + tP*, regression target P* − X̃₀, velocity sees (X_t, t).
        self.n        = n
        self.n_steps  = n_steps
        self.sigma_0  = sigma_0
        self.coupling = coupling      # "penalised" (orbit coupling) | "label" (ablation)
        self.project  = project       # False = ablation: raw f_θ, no C(·)

        self.encoder  = PatchEncoder(in_channels, patch_size, d_model)
        self.velocity = VelocityTransformer(n=n, d_model=d_model, use_t=True)

    def _proj(self, U: torch.Tensor) -> torch.Tensor:
        return tangent_project(U) if self.project else U

    def _couple(self, X0_noisy, obs_blk, slot_blk, gt_slot):
        """
        "label" is the single-label ablation: always the one sampled τ,
        ignoring the rest of 𝒪, as a method without orbit coupling would.
        """
        if self.coupling == "label":
            P = torch.zeros_like(X0_noisy)
            P.scatter_(2, gt_slot.unsqueeze(-1), 1.0)
            return P
        return orbit_couple(X0_noisy, obs_blk, slot_blk, self.coupling)

    # ── Training ──────────────────────────────────────────────────
    def cfm_loss(self,
                 patches:  torch.Tensor,     # [B, n, C, p, p]
                 obs_blk:  torch.Tensor,     # [B, n]
                 slot_blk: torch.Tensor,     # [B, n]
                 gt_slot:  torch.Tensor = None
                 ) -> torch.Tensor:
        """
        1. encode the patches
        2. X̃₀ = (1/n)J + σ₀·T(ε)            (fresh ε every batch)
        3. P* = argmin_{P ∈ 𝒪} ‖X̃₀ − P‖_F   (fresh coupling every batch)
        4. X_t = (1−t)·X̃₀ + t·P*
        5. v_pred = C f(enc, X_t) C
        6. ‖v_pred − (P* − X₀)‖²
        """
        B = patches.shape[0]
        device, dtype = patches.device, patches.dtype

        enc = self.encoder(patches)                            # [B, n, d]

        X0, X0_noisy, _ = make_noisy_X0(B, self.n, self.sigma_0, device, dtype)
        P_star = self._couple(X0_noisy, obs_blk, slot_blk, gt_slot)

        t    = torch.rand(B, device=device, dtype=dtype).view(B, 1, 1)
        X_t  = (1.0 - t) * X0_noisy + t * P_star

        v_pred = self._proj(self.velocity(enc, X_t, t.view(B)))
        return cfm_loss(v_pred, X0_noisy, P_star)        # target P* − X̃₀

    # ── Inference ─────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, patches: torch.Tensor,
               k_samples: int = CFG.K_SAMPLES) -> list:
        """
        K independent Euler-integrated trajectories from K noisy starts.
        Returns a list of K soft [B, n, n] matrices (Hungarian rounding is
        done downstream, in metrics.hungarian_round).
        """
        B, n   = patches.shape[0], self.n
        device, dtype = patches.device, patches.dtype
        dt     = 1.0 / self.n_steps

        enc = self.encoder(patches)                 # encode once, reuse

        out = []
        for _ in range(k_samples):
            X0 = torch.full((B, n, n), 1. / n, device=device, dtype=dtype)
            X  = X0 + self.sigma_0 * sample_tangent_noise(B, n, device, dtype)
            for s in range(self.n_steps):
                t = torch.full((B,), s * dt, device=device, dtype=dtype)
                X = X + dt * self._proj(self.velocity(enc, X, t))
            out.append(X.detach())
        return out

    def forward(self, patches, obs_blk, slot_blk, gt_slot=None):
        return self.cfm_loss(patches, obs_blk, slot_blk, gt_slot)


# ══════════════════════════════════════════════════════════════════
# 4.  ABLATION: STATE-INDEPENDENT VELOCITY
# ══════════════════════════════════════════════════════════════════

class StateIndependentFlowModel(FlowModel):
    """
    Ablation (Section 5.2): the same model with the X_t state token and the
    time input removed, i.e. v = f(patches) alone, trained on the mean
    direction P* - X0.  Trajectories are parallel straight lines, so samples
    differ only through the initial noise.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity = VelocityTransformer(
            n=self.n, d_model=self.encoder.d_model, use_xt=False)

    def cfm_loss(self, patches, obs_blk, slot_blk, gt_slot=None):
        B = patches.shape[0]
        device, dtype = patches.device, patches.dtype
        enc = self.encoder(patches)
        X0, X0_noisy, _ = make_noisy_X0(B, self.n, self.sigma_0, device, dtype)
        P_star = self._couple(X0_noisy, obs_blk, slot_blk, gt_slot)
        v_pred = self._proj(self.velocity(enc, None))
        return cfm_loss(v_pred, X0, P_star)

    @torch.no_grad()
    def sample(self, patches, k_samples: int = CFG.K_SAMPLES) -> list:
        B, n = patches.shape[0], self.n
        device, dtype = patches.device, patches.dtype
        enc = self.encoder(patches)
        v   = self._proj(self.velocity(enc, None))
        out = []
        for _ in range(k_samples):
            X0 = torch.full((B, n, n), 1. / n, device=device, dtype=dtype)
            X  = X0 + self.sigma_0 * sample_tangent_noise(B, n, device, dtype)
            out.append((X + v).detach())     # constant field ⇒ closed form
        return out
