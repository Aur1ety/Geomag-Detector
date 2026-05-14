"""
mag_pipeline.py
===============
Aditya-L1 MAG Level-2 NetCDF ingestion pipeline.

Reads L2_AL1_MAG_YYYYMMDD_V00.nc files from PRADAN,
extracts GSM magnetic field components, cleans fill values,
and resamples from 10-second cadence to 1-minute for modelling.

Variable map (verified against actual L2 files):
    time     : float64 Unix seconds since 1970-01-01
    Bx_gsm   : X component, GSM frame [nT]
    By_gsm   : Y component, GSM frame [nT]
    Bz_gsm   : Z component, GSM frame [nT]  <- key CME indicator
    b_mag    : computed as sqrt(Bx^2 + By^2 + Bz^2)

Why GSM?
    GSM Z-axis aligns with Earth's magnetic dipole. Bz_gsm is the
    standard space weather quantity -- sustained negative Bz_gsm
    drives geomagnetic storms. GSE is ecliptic-aligned and less
    directly relevant to geoeffectiveness.

Fill value: -9999.0 -> replaced with NaN.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("mag_pipeline")

FILL_THRESHOLD = -9000.0

MAG_VARS = {
    "bx": "Bx_gsm",
    "by": "By_gsm",
    "bz": "Bz_gsm",
}


def parse_mag_nc(filepath: Path) -> pd.DataFrame:
    """
    Parse a single MAG L2 NetCDF file into a DataFrame at 10s cadence.

    Parameters
    ----------
    filepath : Path to L2_AL1_MAG_YYYYMMDD_V00.nc

    Returns
    -------
    DataFrame with columns [bx, by, bz, b_mag], DatetimeIndex (UTC naive)
    """
    ds = xr.open_dataset(str(filepath), engine="netcdf4")

    raw_time   = ds["time"].values.astype(np.float64)
    timestamps = pd.to_datetime(raw_time, unit="s", utc=True).tz_localize(None)

    data = {"time": timestamps}
    for key, varname in MAG_VARS.items():
        try:
            arr = ds[varname].values.astype(np.float64)
            arr[arr < FILL_THRESHOLD] = np.nan
            data[key] = arr
        except KeyError:
            logger.warning(
                "Variable '%s' not found in %s -- filling NaN",
                varname, filepath.name
            )
            data[key] = np.full(len(timestamps), np.nan)

    ds.close()

    df = pd.DataFrame(data).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Compute total field magnitude from components
    df["b_mag"] = np.sqrt(df["bx"]**2 + df["by"]**2 + df["bz"]**2)

    logger.info(
        "Parsed: %s | %d rows | Bz [%.1f, %.1f] nT",
        filepath.name, len(df),
        float(np.nanmin(df["bz"])), float(np.nanmax(df["bz"]))
    )
    return df


def load_mag_directory(directory: str) -> pd.DataFrame:
    """
    Load and concatenate all MAG L2 NetCDF files found recursively
    in a directory. Automatically skips L1 files and corrupt files.

    Parameters
    ----------
    directory : path to folder (searched recursively for *.nc)

    Returns
    -------
    Concatenated DataFrame sorted by time, duplicates removed
    """
    all_files = sorted(Path(directory).rglob("*.nc"))
    l2_files  = [
        f for f in all_files
        if f.name.startswith("L2_") and "(" not in f.name
    ]
    skipped = len(all_files) - len(l2_files)

    if skipped:
        logger.warning("Skipped %d non-L2 or duplicate files", skipped)
    if not l2_files:
        raise FileNotFoundError(
            f"No L2 MAG .nc files found in: {directory}"
        )

    frames = []
    for f in l2_files:
        try:
            frames.append(parse_mag_nc(f))
        except Exception as e:
            logger.warning("Skipping corrupt file %s: %s", f.name, e)

    if not frames:
        raise RuntimeError(
            "All files failed to parse -- check your data"
        )

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    logger.info(
        "MAG loaded: %d rows from %d files | %s to %s",
        len(df), len(frames),
        df.index.min().date(), df.index.max().date()
    )
    return df


def resample_mag_to_1min(mag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample MAG from 10s cadence to a uniform 1-minute grid.

    Uses median aggregation -- more robust than mean against
    single-point spike artefacts common in raw magnetometer data.
    At 10s cadence there are 6 samples per minute, giving a
    reliable median estimate.

    Returns
    -------
    DataFrame on a uniform 1-minute DatetimeIndex
    """
    mag_df    = mag_df[~mag_df.index.duplicated(keep="first")].sort_index()
    resampled = mag_df.resample("1min").median()

    nan_frac = float(resampled.isna().mean().mean())
    logger.info(
        "Resampled to 1-min: %d steps | NaN fraction: %.1f%%",
        len(resampled), nan_frac * 100
    )
    if nan_frac > 0.3:
        logger.warning(
            "High NaN fraction (%.1f%%) -- check for data gaps",
            nan_frac * 100
        )
    return resampled