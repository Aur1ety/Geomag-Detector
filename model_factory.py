"""
model_factory.py
================
PatchTransformer model for MAG-based CME detection.

Only PatchTransformer is used — best performer from benchmark (F1=0.982).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class PatchTransformer(nn.Module):
    """
    ViT-style patch attention for time series (PatchTST approach).

    Best at detecting CME flux rope structure via patch-level attention.
    Splits 128-step sequence into 16-step patches → 8 patches total.
    Attention runs over patches, not timesteps — captures local structure
    while maintaining global context for 6-12hr patterns.
    """
    def __init__(
        self,
        input_dim: int = 8,
        patch_size: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        seq_len: int = 128,
    ):
        super().__init__()
        assert seq_len % patch_size == 0, \
            f"seq_len ({seq_len}) must be divisible by patch_size ({patch_size})"

        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        self.patch_embed = nn.Linear(patch_size * input_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape

        x = x.reshape(B, self.num_patches, self.patch_size * F)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        return self.head(cls_out).squeeze(-1)


class XGBoostModel:
    """XGBoost baseline for comparison."""
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: int = 20,
    ):
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            )
            self.fitted = False
        except ImportError:
            raise ImportError("pip install xgboost")

    @staticmethod
    def extract_features(X: np.ndarray) -> np.ndarray:
        from scipy.stats import skew, kurtosis
        N, T, F = X.shape
        stats = []
        for i in range(N):
            window = X[i]
            row = []
            for f in range(F):
                col = window[:, f]
                row.extend([
                    col.mean(), col.std(), col.min(), col.max(),
                    col.max() - col.min(), col[-1], col[0],
                    np.polyfit(np.arange(T), col, 1)[0],
                    float(skew(col)), float(kurtosis(col)),
                ])
            bz_col = window[:, 0]
            pers_col = window[:, 6]
            rot_col = window[:, 4]
            row.extend([
                bz_col.min(), pers_col.max(), rot_col.sum(),
                (bz_col < -10).sum(), (bz_col < -20).sum(),
            ])
            stats.append(row)
        return np.array(stats, dtype=np.float32)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        Xf = self.extract_features(X_train)
        eval_set = None
        if X_val is not None:
            eval_set = [(self.extract_features(X_val), y_val)]
        self.model.fit(Xf, y_train, eval_set=eval_set, verbose=False)
        self.fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        return self.model.predict_proba(self.extract_features(X))[:, 1]


class LightGBMModel:
    """LightGBM for SHAP interpretability."""
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: int = 20,
    ):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            self.fitted = False
            self._explainer = None
            self._feat_names = None
        except ImportError:
            raise ImportError("pip install lightgbm")

    @staticmethod
    def extract_features(X: np.ndarray) -> np.ndarray:
        return XGBoostModel.extract_features(X)

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        Xf = self.extract_features(X_train)
        self._feat_names = feature_names

        callbacks = []
        eval_set = None
        if X_val is not None:
            try:
                import lightgbm as lgb
                callbacks = [lgb.early_stopping(50, verbose=False)]
            except Exception:
                pass
            eval_set = [(self.extract_features(X_val), y_val)]

        self.model.fit(Xf, y_train, eval_set=eval_set, callbacks=callbacks if eval_set else None)
        self.fitted = True

        try:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
        except Exception:
            self._explainer = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        return self.model.predict_proba(self.extract_features(X))[:, 1]

    def shap_summary(self, X: np.ndarray, feature_names=None):
        import pandas as pd
        if self._explainer is None:
            raise RuntimeError("SHAP explainer not available")
        Xf = self.extract_features(X)
        sv = self._explainer.shap_values(Xf)
        sv = sv[1] if isinstance(sv, list) else sv
        names = feature_names or self._feat_names
        df = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        return df.sort_values("mean_abs_shap", ascending=False).head(15)