import os
import numpy as np
import pandas as pd
import mne

# -------- settings --------
edf_path = "sleep-cassette/SC4002E0-PSG.edf"      # input EDF
out_csv = "SleepEDF_SC4002E0.csv" # output file
channels_keep = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
target_hz = 1.0                    # downsample to 1 Hz
start_time = pd.Timestamp("2000-01-01 00:00:00")
# --------------------------

raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

# Keep only requested channels that exist
available = [ch for ch in channels_keep if ch in raw.ch_names]
if not available:
    raise ValueError(f"No requested channels found. Available: {raw.ch_names}")
raw.pick(available)

# Resample for manageable sequence length
raw.resample(target_hz, npad="auto")

data = raw.get_data()  # shape: (n_channels, n_times)
n_channels, n_times = data.shape

# Per-channel z-score normalization
data_z = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

# Build datetime index
times = pd.date_range(start=start_time, periods=n_times, freq=f"{int(1/target_hz)}s")

# Build DUET long format: date, value, cols
rows = []
for i, ch in enumerate(available):
    df_ch = pd.DataFrame({
        "date": times,
        "value": data_z[i],
        "cols": ch
    })
    rows.append(df_ch)

out_df = pd.concat(rows, axis=0, ignore_index=True)
out_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}, shape={out_df.shape}, channels={available}, n_times={n_times}")