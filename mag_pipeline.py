from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("mag_pipeline")
FILL_THRESHOLD = -9000.0 

MAG_VARS = {"time": "time", "bx": "Bx_gsm", "by": "By_gsm", "bz": "Bz_gsm"}

def parse_mag_nc(filepath: Path) -> pd.DataFrame:
    # ONLY CHANGE: Added engine="netcdf4" to resolve the xarray ValueError
    ds = xr.open_dataset(str(filepath), engine="netcdf4")
    
    ts = pd.to_datetime(ds["time"].values.astype(np.float64), unit="s", utc=True).tz_localize(None)
    data = {"time": ts}
    for key, var in MAG_VARS.items():
        if key == "time": continue
        try:
            arr = ds[var].values.astype(np.float64)
            arr[arr < FILL_THRESHOLD] = np.nan
            data[key] = arr
        except: 
            data[key] = np.full(len(ts), np.nan)
            
    ds.close()
    df = pd.DataFrame(data).set_index("time").sort_index()
    df["b_mag"] = np.sqrt(df["bx"]**2 + df["by"]**2 + df["bz"]**2)
    return df

def load_mag_directory(directory: str) -> pd.DataFrame:
    files = sorted(Path(directory).rglob("*.nc"))
    l2 = [f for f in files if f.name.startswith("L2_")]
    if not l2: 
        raise FileNotFoundError(f"No L2 MAG files in: {directory}")
    return pd.concat([parse_mag_nc(f) for f in l2]).sort_index().pipe(lambda d: d[~d.index.duplicated(keep="first")])

def resample_mag_to_swis(mag_df: pd.DataFrame, swis_index: pd.DatetimeIndex, window_seconds: int = 60) -> pd.DataFrame:
    rolling = mag_df.rolling(f"{window_seconds}s", center=True).median()
    resampled = rolling.reindex(swis_index, method='nearest', tolerance=pd.Timedelta(seconds=window_seconds // 2))
    return resampled