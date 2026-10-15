"""
src/models/transformer_encoder.py
=================================
PyTorch Transformer Feature Extractor for 5G-V2X DoS Detection (Option B).

Architecture Specifications:
----------------------------
1. Input Shape: (batch_size, seq_len=10, in_features=9)
2. Linear Input Projection: Linear(9, 64)
3. Learned Positional Encoding: Parameter(1, 10, 64)
4. Transformer Encoder: 2 layers, 4 attention heads, d_ff=128, dropout=0.1, GELU activation
5. Temporal Pooling: Mean pooling over all 10 timesteps -> 64-dimensional embedding vector
6. Temporary Classification Head: Linear(64, 32) -> ReLU -> Dropout -> Linear(32, 1)
   Used strictly during pretraining with BCEWithLogitsLoss to shape discriminative representations.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class TransformerFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_features: int = 9,
        seq_len: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation = activation

        # 1. Linear Input Projection
        self.input_proj = nn.Linear(in_features, d_model)

        # 2. Learned Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.input_layer_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Final LayerNorm prior to pooling
        self.final_norm = nn.LayerNorm(d_model)

        # 5. Temporary Binary Classification Head (for pretraining only)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 64-dimensional temporal mean-pooled representation.

        Args:
            x: Input sequence tensor of shape (batch_size, seq_len, in_features).
        Returns:
            embeddings: Tensor of shape (batch_size, d_model=64).
        """
        # x: (B, 10, 9)
        h = self.input_proj(x) + self.pos_embedding  # (B, 10, 64)
        h = self.input_layer_norm(self.input_dropout(h))
        out = self.transformer_encoder(h)  # (B, 10, 64)
        out = self.final_norm(out)
        # Temporal mean pooling over 10 transmissions
        embeddings = out.mean(dim=1)  # (B, 64)
        return embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor + temporary classification head.

        Args:
            x: Input sequence tensor of shape (batch_size, seq_len, in_features).
        Returns:
            logits: Logits tensor of shape (batch_size,).
        """
        embeddings = self.forward_features(x)
        logits = self.classifier_head(embeddings).squeeze(-1)
        return logits

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference method to extract frozen embeddings.
        """
        was_training = self.training
        self.eval()
        emb = self.forward_features(x)
        if was_training:
            self.train()
        return emb

    def get_config(self) -> Dict:
        return {
            "in_features": self.in_features,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate,
            "activation": self.activation,
        }
