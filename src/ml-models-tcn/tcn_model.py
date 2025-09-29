"""
tcn_model.py

Title: Temporal Convolutional Network (TCN) model for patient-level predictions

Summary: 
- Processes timestamp-level ICU sequences (up to 96 hours per patient).
- input → causal convolutions → stacked into Temporal Residual Blocks → pooling → dense head
- Outputs patient-level outcomes:
    - Classification head → max_risk, median_risk (binary tasks).
    - Regression head → pct_time_high (continuous task).
- Uses causal dilated convolutions → can learn from temporal trends, spikes, and trajectories.
- Temporal residual blocks: 2x (conv + layer norm + activation + dropout) + downsample + residual add → stability, generalisation, prevent overfitting.
- Masked mean pooling → avoids padded timesteps corrupting pooled patient vectors.
- Two task heads → supports both classification (max_risk/median_risk) and regression (pct_time_high).  
- Purpose:
    - Shows how deep sequence modelling (TCN) is different from LightGBM (aggregated stats).
    - Captures temporal patterns (trends, spikes, deterioration trajectories) that classical ML (e.g. LightGBM) cannot see. 
Inputs:
- `x`: FloatTensor of shape (batch, seq_len, num_features).  
- `mask`: FloatTensor of shape (batch, seq_len), with 1.0 for valid timesteps, 0.0 for padding.  
Outputs:
- Dictionary:
    - `'logit'`: (batch,) → classification head (use BCEWithLogitsLoss).  
    - `'regression'`: (batch,) → regression head (use MSELoss).  
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import torch                    # Core PyTorch tensor operations and GPU acceleration
import torch.nn as nn           # High-level building blocks: Conv1d, Linear, Dropout, etc.
from typing import Optional     # Allows you to write "Optional[int]" → can be int or None

# -------------------------------------------------------------
# Step 1: Causal Convolution Layer
# -------------------------------------------------------------
'''
CausalConv1d class implements a causal 1D convolution: a convolution that never uses future timesteps (no leakage).
- Standard convolutions look left + right in time, which leaks future info.
- A causal convolution only looks backwards (past + present), not forwards.
- PyTorch’s nn.Conv1d = 1D convolution 
- CausalConv1d is a 1D causal convolution wrapper using symmetric padding and trimming (right side) so kernel only depends on the present and past (to avoid future leakage).
- Padding + trimming is length-preserving (input length = output length), so each timestamp maps cleanly to an output, important for stacking layers
- This lets us build Temporal Convolutional Networks (TCNs) that are causal and can use dilated kernels to get a large temporal receptive field without deep stacks (less layers).
'''
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size # number of input steps each convolution looks at
        self.dilation = dilation # spacing between each input that kernal looks at
        # padding so that output length = input length + padding_trim
        self.padding = (kernel_size - 1) * dilation # formula for calculating number of padding values to add in each layer to ensure output length = input length.
        # self.conv = nn.Conv1d(...) is the actual convolution layer object PyTorch provides. We wrap it so we can trim off extra padding later to enforce causality.
        # Conv1d expects shape (batch, channels, seq_len)
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,  # dilation = spacing between kernel taps
            padding=self.padding,
        )
    # forward pass
    def forward(self, x):
        # Input (x): (B, C_in, L) (batch, channels, seq_len)
        out = self.conv(x)  # applies the convolution, output sequence temporarily longer due to padding
        if self.padding != 0:
            # trim the rightmost padding (future) to make convolution causal (prevents "seeing into the future")
            # padding was added symmetrically by PyTorch, but we only want past padding (not future).
            out = out[:, :, : -self.padding]
        return out # Output: (B, C_out, L), same length as input (L), but possibly more channels (C_out), because convolutions can increase feature depth.

# -------------------------------------------------------------
# Step 2: Temporal Residual Block
# -------------------------------------------------------------
"""
A TemporalBlock is like one “unit” in the TCN stack, each block has: 
- 2 causal convolutions (so it looks back in time only)
- LayerNorm (normalisation to stabilise training)
- Dropout (to avoid overfitting)
- Skip connection (residual) to prevent vanishing gradients.
"""
class TemporalBlock(nn.Module):
    # define the temporal block
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):  # dropout = probability of “dropping” neurons to prevent overfitting.
        super().__init__()

        # -------------------------------------------------------------
        # Causal convolutions x2
        # -------------------------------------------------------------
        # Two stacked causal convolutions, each one learns temporal filters that extract patterns from the sequence, share the same dilation inside the block.
        # First causal conv
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        # Second causal conv
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)

        # -------------------------------------------------------------
        # Normalisation (LayerNorm)
        # -------------------------------------------------------------
        # After each convolution, the scale of activations might explode or vanish
        # LayerNorm normalises across the outputs of all kernels (channels) at each timestep, so every timestep’s feature vector has a stable scale (mean 0, var 1) → stabilises training   
        self.layernorm1 = nn.LayerNorm(out_ch)  # applied on (batch, seq_len, channels) after permute
        self.layernorm2 = nn.LayerNorm(out_ch)

        # -------------------------------------------------------------
        # Activation (ReLU) & Dropout
        # -------------------------------------------------------------
        self.activation = nn.ReLU()             # ReLU introduces non-linearity → lets the network model complex patterns (not just linear sums).
        self.dropout = nn.Dropout(dropout)      # Dropout randomly sets some activations to 0 during training → prevents overfitting and forces robustness.
        
        # -------------------------------------------------------------
        # Downsample & Residual Connection
        # -------------------------------------------------------------
        # Downsample when input channels ≠ output channels, 1×1 convolution adjusts dimensions to match, so residual addition is valid.
        # Residual connection = original input sequence added back to the output of convolutions after the temporal block finishes.
        if in_ch != out_ch:     # If input (in_ch) and output (out_ch) dimensions differ, we can’t just add them directly.
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1)   # Solution: a 1x1 convolution (just reshaping channels) to make them match.
        else: 
            self.downsample = None      # match channels for residual

    # -------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------
    # Save the original input → we’ll add it back later (residual connection).
    def forward(self, x):
        # input (x): (B, C_in, L) (batch, channels, seq_len)
        residual = x    # store for skip / residual connection

        # --- First conv → norm → ReLU → dropout ---
        # First convolution looks at local temporal patterns (via kernel + dilation) → then we normalise across channels → then activate (ReLU) → then apply dropout.
        out = self.conv1(x)                 # (B, out_ch, L)
        out = out.permute(0, 2, 1)          # (B, L, out_ch) → needed for LayerNorm
        out = self.layernorm1(out)
        out = out.permute(0, 2, 1)          # back to (B, out_ch, L)
        out = self.activation(out)
        out = self.dropout(out)

        # --- Second conv → norm → ReLU → dropout ---
        # Repeat the same pipeline with another convolution.
        # Two conv layers in a row = more expressive features per block.
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.layernorm2(out)
        out = out.permute(0, 2, 1) 
        out = self.activation(out)
        out = self.dropout(out)

        # --- Residual add ---
        # If input and output shapes differ → adjust the residual with a 1×1 conv
        # Add the input (residual) back to the processed output.
        # This skip connection ensures the network can still propagate gradients backward (shortcut), avoiding the vanishing gradient problem.
        if self.downsample is not None:
            residual = self.downsample(             # self.downsample(residual) = defined as a 1x1 convolution earlier, if number of channels (C) don't match, matrix will map C_in → C_out.
                residual[:, :, : out.shape[2]]      # residual[:, :, : out.shape[2]] = handle any small length (L) mismatch (our tensor may be shorter than residual after padding + trimming in causal conv)
            )  

        # residual add (C + L are equal in both out + residual)
        return self.activation(out + residual)

# -------------------------------------------------------------
# Step 3: TCN Model with Masked Pooling
# -------------------------------------------------------------
# Stacks multiple TemporalBlocks with exponentially increasing dilation.
# Then pools across time → dense head(s) for classification/regression.
class TCNModel(nn.Module):
    """
    Stacked TCN with masked global pooling and heads for classification/regression.

    Args:
        num_features: input feature dimension
        num_channels: list of channel sizes for each temporal block, e.g. [64, 128, 128]
        kernel_size: conv kernel size
        dropout: dropout rate
        head_hidden: hidden units in final dense head (optional)
    """

    def __init__(
        self,
        num_features: int,                       # number of input variables per timestep
        num_channels: list = [64, 128, 128],     # hidden channel sizes for blocks
        kernel_size: int = 3,                    # convolution kernel width
        dropout: float = 0.2,
        head_hidden: Optional[int] = 64,         # optional dense hidden layer before outputs
    ):
        super().__init__()
        self.num_features = num_features
        self.num_channels = num_channels

        # ---- Stack temporal blocks ----
        layers = []
        in_ch = num_features
        # dilation doubling per block
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i   # 1, 2, 4, ... doubles receptive field each block
            tb = TemporalBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            layers.append(tb)
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.feature_dim = num_channels[-1]     # last block output size

        # ---- Optional dense head shared by classification/regression (we return logits) ----
        if head_hidden is not None:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_out_dim = head_hidden
        else:
            self.head = None
            head_out_dim = self.feature_dim

        # ---- Task-specific heads separate final linear layers ----
        self.classifier = nn.Linear(head_out_dim, 1)   # binary classification (logit)
        self.regressor = nn.Linear(head_out_dim, 1)    # regression (pct_time_high)

    # ---------------------------------------------------------
    # Step 4: Masked pooling
    # ---------------------------------------------------------
    def masked_mean_pool(self, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
        """
        Pool across time but ignore padding.
        x: (B, L, C)
        mask: (B, L) with 1=real timestep, 0=padding
        """
        mask = mask.unsqueeze(-1)                  # (batch, seq_len, 1)
        x_masked = x * mask                        # zero out padded timesteps
        sums = x_masked.sum(dim=1)                 # sum over time → (batch, channels)
        counts = mask.sum(dim=1).clamp(min=eps)    # (batch, 1) avoid div by 0
        mean = sums / counts
        return mean

    # ---------------------------------------------------------
    # Step 5: Forward pass
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass through TCN

        Args:
            x: (batch, seq_len, num_features)
            mask: (batch, seq_len) float tensor {0,1} or None
        Returns: 
            dict {'logit': (batch,) tensor, 'regression': (batch,) tensor}
        """
        # Permute to (batch, channels, seq_len) for Conv1d
        # Input: x = (B, L, F), mask = (B, L)
        x_in = x.permute(0, 2, 1)   # (B, F, L) for Conv1d
        out = self.tcn(x_in)        # (B, C_last, L), apply stacked temporal blocks
        out = out.permute(0, 2, 1)  # (B, L, C_last)

        # Pool across time
        if mask is None:
            pooled = out.mean(dim=1)
        else:
            pooled = self.masked_mean_pool(out, mask)

        # Dense head
        if self.head is not None:
            pooled = self.head(pooled)

        # Task outputs
        logit = self.classifier(pooled)         # (B, 1)
        regression = self.regressor(pooled)     # (B, 1)

        return {
            "logit": logit.squeeze(-1),             # (B,)
            "regression": regression.squeeze(-1)    # (B,)
        }

# -------------------------------------------------------------
# Step 6: Quick unit test / smoke test
# -------------------------------------------------------------
if __name__ == "__main__":
    # small smoke test
    B = 4
    L = 96
    F = 173
    device = "cpu"
    model = TCNModel(num_features=F, num_channels=[64, 64, 128], kernel_size=3, dropout=0.2, head_hidden=64).to(device)

    # dummy input: batch x seq x features
    x = torch.randn(B, L, F, device=device)
    # create mask with variable lengths
    mask = torch.zeros(B, L, device=device)
    lengths = [96, 60, 12, 96]
    for i, ln in enumerate(lengths):
        mask[i, :ln] = 1.0

    out = model(x, mask)
    print("logit.shape:", out["logit"].shape)        # expected (B,)
    print("regression.shape:", out["regression"].shape)  # expected (B,)

    # Check numeric values
    assert out["logit"].shape == (B,)
    assert out["regression"].shape == (B,)
    print("TCN smoke test passed.")