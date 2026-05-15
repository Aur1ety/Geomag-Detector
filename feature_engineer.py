"""
feature_engineer.py
===================
MAG-only feature engineering -- 9 physics-informed features
derived from Aditya-L1 MAG Level-2 data.

No SWIS dependency. Designed to complement the existing
SWIS-based plasma detector as a separate magnetic detector.

Input:  DataFrame with columns [bx, by, bz, b_mag] from mag_pipeline
Output: DataFrame with 9 feature columns on the same DatetimeIndex
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


MAG_FEATURE_NAMES = [
    "bz",             # North-south IMF -- primary CME indicator
    "b_mag",          # Total field magnitude -- sheath compression
    "clock_angle",    # arctan2(By, Bz) -- flux rope orientation
    "dbz_dt",         # Bz rate of change -- shock arrival detector
    "b_rotation",     # Rolling B vector rotation -- flux rope passage
    "bz_smoothed",    # Low-pass Bz -- sustained southward field
    "bz_persistence", # Consecutive southward steps -- rules out noise
    "b_elevation",    # arctan(Bz / sqrt(Bx^2+By^2)) -- 3D field angle
    "high_b_mag_rotation" # Sheath Detector -- catches Northward Bz CMEs
]


def build_mag_features(mag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 9 MAG feature columns from resampled MAG data.

    Parameters
    ----------
    mag_df : DataFrame with columns [bx, by, bz, b_mag]
             produced by mag_pipeline.resample_mag_to_1min()

    Returns
    -------
    DataFrame with 9 columns (MAG_FEATURE_NAMES), same DatetimeIndex.
    NaN rows filled by forward/backward fill then column median.
    """
    feat = pd.DataFrame(index=mag_df.index)

    bx = mag_df["bx"]
    by = mag_df["by"]
    bz = mag_df["bz"]
    bt = mag_df["b_mag"]

    # ------------------------------------------------------------------
    # 1. Bz [nT]
    # The north-south component of the interplanetary magnetic field.
    # ------------------------------------------------------------------
    feat["bz"] = bz

    # ------------------------------------------------------------------
    # 2. Total field magnitude |B| [nT]
    # Enhancement in |B| signals either sheath or flux rope arrival.
    # ------------------------------------------------------------------
    feat["b_mag"] = bt

    # ------------------------------------------------------------------
    # 3. Clock angle = arctan2(By, Bz) in degrees [-180, 180]
    # ------------------------------------------------------------------
    feat["clock_angle"] = np.degrees(np.arctan2(by, bz))

    # ------------------------------------------------------------------
    # 4. dBz/dt -- first difference of Bz
    # ------------------------------------------------------------------
    feat["dbz_dt"] = bz.diff().fillna(0.0)

    # ------------------------------------------------------------------
    # 5. B vector rotation rate (rolling 60-step sum)
    # High rolling rotation = flux rope actively passing the sensor.
    # ------------------------------------------------------------------
    bvec = np.column_stack([bx.values, by.values, bz.values])
    dot  = np.sum(bvec[1:] * bvec[:-1], axis=1)
    norm = (
        np.linalg.norm(bvec[1:], axis=1) *
        np.linalg.norm(bvec[:-1], axis=1)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.clip(dot / (norm + 1e-9), -1.0, 1.0)

    rotation_deg = np.degrees(np.arccos(cos_angle))
    rotation_deg = np.concatenate([[0.0], rotation_deg])

    feat["b_rotation"] = (
        pd.Series(rotation_deg, index=mag_df.index)
        .rolling(window=60, min_periods=1)
        .sum()
    )

    # ------------------------------------------------------------------
    # 6. Smoothed Bz (Savitzky-Golay, window=121 steps ~ 2 hours)
    # ------------------------------------------------------------------
    bz_arr = bz.interpolate(method="linear", limit_direction="both").values
    win = min(121, len(bz_arr) - 1)
    if win % 2 == 0:
        win -= 1

    if win >= 5:
        bz_smooth = savgol_filter(bz_arr, window_length=win, polyorder=3)
    else:
        bz_smooth = bz_arr.copy()

    bz_smooth[bz.isna().values] = np.nan
    feat["bz_smoothed"] = bz_smooth

    # ------------------------------------------------------------------
    # 7. Bz persistence (consecutive southward steps)
    # ------------------------------------------------------------------
    southward   = (bz.fillna(0) < 0).astype(int).values
    persistence = np.zeros(len(southward), dtype=np.float32)
    count = 0
    for i, s in enumerate(southward):
        count       = (count + 1) * int(s)
        persistence[i] = count
    feat["bz_persistence"] = persistence

    # ------------------------------------------------------------------
    # 8. B elevation angle = arctan2(Bz, sqrt(Bx^2 + By^2)) in degrees
    # ------------------------------------------------------------------
    bxy = np.sqrt(bx**2 + by**2)
    feat["b_elevation"] = np.degrees(np.arctan2(bz, bxy))

    # ------------------------------------------------------------------
    # 9. Sheath Detector (high_b_mag_rotation)
    # Catches Northward-Bz CMEs by flagging sudden spikes in magnitude
    # combined with chaotic field rotation.
    # ------------------------------------------------------------------
    # Rolling 2-hour background solar wind
    b_mag_mean = bt.rolling(window=120, min_periods=1).mean()
    b_mag_std = bt.rolling(window=120, min_periods=1).std().fillna(0)

    # Condition 1: Magnitude is 2+ standard deviations above background
    is_high_compression = bt > (b_mag_mean + 2 * b_mag_std)

    # Condition 2: High rotation (>30 degrees cumulative over an hour)
    is_chaotic_rotation = feat["b_rotation"] > 30.0

    # Combine: 1.0 if both happen, 0.0 otherwise
    feat["high_b_mag_rotation"] = (is_high_compression & is_chaotic_rotation).astype(float)

    # ------------------------------------------------------------------
    # Final NaN handling
    # 1. Forward/backward fill for short gaps (sensor dropouts <= 12 min)
    # 2. Column median for any remaining NaN (long gaps)
    # ------------------------------------------------------------------
    feat = feat.ffill(limit=12).bfill(limit=12)
    for col in feat.columns:
        if feat[col].isna().any():
            feat[col] = feat[col].fillna(feat[col].median())

    print(
        f"Feature matrix: {feat.shape} | "
        f"columns: {list(feat.columns)}"
    )
    return feat