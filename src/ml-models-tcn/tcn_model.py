"""
tcn_model.py

Title: Temporal Convolutional Network (TCN) model for patient-level predictions

Summary: 
- Processes timestamp-level ICU sequences (up to 96 hours per patient).
- Outputs patient-level outcomes:
    - Classification head → max_risk, median_risk (binary tasks).
    - Regression head → pct_time_high (continuous task).
- Uses causal dilated convolutions → can learn from temporal trends, spikes, and trajectories.
- Residual blocks + layer norm + dropout → stability, generalisation, prevent overfitting.
- Masked mean pooling → avoids padded timesteps corrupting pooled patient vectors.
- Two task heads → supports both classification (max_risk/median_risk) and regression (pct_time_high).  
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
# Standard convolutions look left + right in time, which leaks future info.
# A causal convolution only looks backwards (past + present), not forwards.
# 1D causal convolution wrapper using padding and trimming to avoid future leakage.
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # padding so that output length = input length + padding_trim
        self.padding = (kernel_size - 1) * dilation
        # Conv1d expects shape (batch, channels, seq_len)
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,  # dilation = spacing between kernel taps
            padding=self.padding,
        )

    def forward(self, x):
        # Input (x): (B, C_in, L) (batch, channels, seq_len)
        out = self.conv(x)  # temporarily longer due to padding
        if self.padding != 0:
            # trim the rightmost padding (future) to make convolution causal (prevents "seeing into the future")
            out = out[:, :, : -self.padding]
        return out # Output: (B, C_out, L)


# -------------------------------------------------------------
# Step 2: Temporal Residual Block
# -------------------------------------------------------------
# Deep stacks of convs can suffer from vanishing gradients.
# Residual temporal block with two causal convolutional layers skip connections + normalisation + dropout.
# This is the "ResNet-style" block for TCN.
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        # First causal conv
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        # Second causal conv
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)

        # LayerNorm normalises across channels → stabilises training      
        self.layernorm1 = nn.LayerNorm(out_ch)  # applied on (batch, seq_len, channels) after permute
        self.layernorm2 = nn.LayerNorm(out_ch)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # LayerNorm normalises across channels → stabilises training
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else: 
            self.downsample = None # match channels for residual

    def forward(self, x):
        # input (x): (B, C_in, L) (batch, channels, seq_len)
        residual = x    # store for skip connection

        # --- First conv → norm → ReLU → dropout ---
        out = self.conv1(x)                 # (B, out_ch, L)
        out = out.permute(0, 2, 1)          # (B, L, out_ch) → needed for LayerNorm
        out = self.layernorm1(out)
        out = out.permute(0, 2, 1)          # back to (B, out_ch, L)
        out = self.activation(out)
        out = self.dropout(out)

        # --- Second conv → norm → ReLU → dropout ---
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.layernorm2(out).permute(0, 2, 1)
        out = self.activation(out)
        out = self.dropout(out)

        # --- Residual add ---
        if self.downsample is not None:
            residual = self.downsample(residual[:, :, : out.shape[2]])  # handle any small length mismatch

        # Residual add
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