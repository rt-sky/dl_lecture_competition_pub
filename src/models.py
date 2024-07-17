#7 domain適応
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class GradientReversalLayer(nn.Module):
    def forward(self, x):
        return GradientReversalFunction.apply(x)

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        d_model: int = 256, #512
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024, #2048
        dropout: float = 0.15
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_channels, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)

        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, 100),
            nn.ReLU(),
            nn.Linear(100, 4)  # 4は被験者の数
        )
        
    def forward(self, X: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        X = X.permute(0, 2, 1)  # (b, c, t) -> (b, t, c)
        X = self.input_proj(X)  # (b, t, d_model)
        X = X + self.positional_encoding[:X.size(1), :]  # add positional encoding
        X = self.transformer_encoder(X)  # (b, t, d_model)
        
        # 分類タスク
        class_out = self.classifier(X[:, 0, :])  # (b, num_classes)
        
        # ドメイン識別タスク
        grl_out = self.grl(X[:, 0, :])
        domain_out = self.domain_classifier(grl_out)

        return class_out, domain_out
