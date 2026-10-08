"""
soh.pidr_features
=================
PID-R features of the frozen NOCE-Net recurrent states (Appendix
"PID-R Feature Extraction from GRU Hidden States").

For a state sequence ``H = [h_1, ..., h_N]`` (``h_c`` in R^{d_h}):

* Proportional ``P_c = h_c``
* Derivative   ``D_c = h_c - h_{c-1}`` (``D_1 = 0``)
* Integral     ``I_c = lambda * I_{c-1} + (1 - lambda) * h_c`` (EMA with a
  learnable per-dimension decay ``lambda`` in (0.5, 0.999), initialised at 0.95)
* Relative     ``R_c = h_c - H[0]`` (displacement from the first state of
  the sequence)

The concatenation ``phi_c = [P_c | D_c | I_c | R_c]`` (4 d_h) is the input
of the SOH heads in ``soh/direct_soh_head.py`` and ``soh/per_dataset_mlp.py``.
"""
import torch
import torch.nn as nn


class PIDFeatureExtractor(nn.Module):
    """
    Constructs PID-style temporal features from GRU hidden states:

        P(c)  = h_c                          proportional — current state
        D(c)  = h_c - h_{c-1}               derivative   — rate of change
        I(c)  = EMA of h over [0..c]        integral     — accumulated history
        R(c)  = h_c - H[0]                  relative     — drift from the first state

    The EMA is computed with a learnable per-dimension decay
    λ ∈ (lambda_min, lambda_max) (sigmoid reparameterisation):

        s_c = λ ⊙ s_{c-1} + (1 - λ) ⊙ h_c

    where ⊙ is element-wise multiplication and the recursion starts from
    the first state of the sequence.

    Output feature dim = 4 * d_h (P, D, I, R concatenated).

    Batched operation supports variable-length sequences via a boolean
    padding mask. Padded positions receive zero features and are excluded
    from any downstream loss via the returned mask.

    Args:
        d_h          : hidden state dimension
        lambda_init  : initial value for all λ entries (default 0.95)
        lambda_min   : minimum λ (prevents λ collapsing to 0)
        lambda_max   : maximum λ (prevents λ → 1, i.e. no update)
    """

    def __init__(
        self,
        d_h: int,
        lambda_init: float = 0.95,
        lambda_min:  float = 0.50,
        lambda_max:  float = 0.999,
    ):
        super().__init__()
        self.d_h        = d_h
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Unconstrained parameter; clamped to (lambda_min, lambda_max) in forward
        lambda_raw_init = torch.log(
            torch.tensor(lambda_init - lambda_min) /
            torch.tensor(lambda_max - lambda_init)
        )
        self.lambda_raw = nn.Parameter(
            lambda_raw_init * torch.ones(d_h)
        )

    @property
    def lambda_(self) -> torch.Tensor:
        """Returns λ clamped to (lambda_min, lambda_max) via sigmoid reparameterisation."""
        sig = torch.sigmoid(self.lambda_raw)
        return self.lambda_min + (self.lambda_max - self.lambda_min) * sig

    def forward_single(self, H: torch.Tensor) -> torch.Tensor:
        """
        Single-sequence forward pass (no padding).

        Args:
            H : (N, d_h)

        Returns:
            features : (N, 4 * d_h)  [P | D | I | R]
        """
        N, d_h = H.shape
        lam = self.lambda_                              # (d_h,)

        # --- Proportional (P) ------------------------------------------------
        P = H                                           # (N, d_h)

        # --- Derivative (D) --------------------------------------------------
        D = torch.zeros_like(H)
        D[1:] = H[1:] - H[:-1]                         # D[0] = 0

        # --- Integral via learnable EMA (I) ----------------------------------
        # Sequential recursion: s_c = lam * s_{c-1} + (1-lam) * h_c, starting from s = h_1
        I = torch.zeros_like(H)
        s = H[0].clone()                                # initialise at h_0
        for c in range(N):
            s = lam * s + (1.0 - lam) * H[c]
            I[c] = s

        # --- Relative (R) ----------------------------------------------------
        R = H - H[0:1, :]                               # (N, d_h)

        return torch.cat([P, D, I, R], dim=-1)          # (N, 4*d_h)

    def forward(
        self,
        H_batch: torch.Tensor,
        lengths: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """
        Batched forward pass with masking for variable-length sequences.

        The EMA recurrence is evaluated sequentially over the FEC axis and
        vectorised over the batch. At padded positions the EMA state is
        frozen, so batteries of different lengths in the same batch do not
        affect each other.

        Args:
            H_batch : (B, N_max, d_h) padded hidden states
            lengths : (B,)            actual sequence lengths

        Returns:
            features : (B, N_max, 4*d_h)  — padded positions are zeroed
            mask     : (B, N_max)  bool — True where position is valid (not padding)
        """
        B, N_max, d_h = H_batch.shape
        device = H_batch.device
        lam = self.lambda_.to(device)                   # (d_h,)

        # ----- Build boolean validity mask -----------------------------------
        # mask[b, n] = True if n < lengths[b]
        idx   = torch.arange(N_max, device=device).unsqueeze(0)   # (1, N_max)
        mask  = idx < lengths.unsqueeze(1)                         # (B, N_max)
        mask3 = mask.unsqueeze(-1).expand_as(H_batch)              # (B, N_max, d_h)

        # Zero out padding positions so they don't pollute the EMA
        H_safe = H_batch * mask3.float()

        # ----- Proportional (P) ----------------------------------------------
        P = H_safe                                      # (B, N_max, d_h)

        # ----- Derivative (D) ------------------------------------------------
        D = torch.zeros_like(H_safe)
        # Shift H_safe by 1 along time; first position stays 0
        D[:, 1:, :] = H_safe[:, 1:, :] - H_safe[:, :-1, :]
        # Zero derivative at the first real position is correct (no prior step)
        # Also zero derivative at the first padding position (already 0 from H_safe)

        # ----- Integral via EMA (I) -------------------------------------------
        # Sequential recursion over the FEC axis, vectorised over the batch:
        #   s_c = lam * s_{c-1} + (1 - lam) * h_c,  starting from s = h_1.
        I = torch.zeros_like(H_safe)                   # (B, N_max, d_h)
        # Initialise recurrent state from h_0 of each battery
        s = H_safe[:, 0, :].clone()                    # (B, d_h)

        for c in range(N_max):
            # Update EMA only for valid positions
            valid_c = mask[:, c].float().unsqueeze(-1)  # (B, 1)
            s_new   = lam * s + (1.0 - lam) * H_safe[:, c, :]
            # For padding positions keep s unchanged (so future valid steps
            # aren't corrupted by zeros)
            s = s_new * valid_c + s * (1.0 - valid_c)
            I[:, c, :] = s

        # Zero out padding positions
        I = I * mask3.float()

        # ----- Relative (R) --------------------------------------------------
        h0 = H_safe[:, 0:1, :].expand_as(H_safe)      # (B, N_max, d_h)
        R  = (H_safe - h0) * mask3.float()

        # ----- Concatenate ---------------------------------------------------
        features = torch.cat([P, D, I, R], dim=-1)     # (B, N_max, 4*d_h)

        return features, mask
