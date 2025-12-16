# Dataset Preprocessing (MobiAct/MobiFall -> LIMU-BERT)
# 1. Walks through all CSVs in the Annotated Data Folder
# 2. For each file
#       - Uses rel_time as the time axis
#       - Resamples to 20 Hz
#       - Low-pass iflters IMU channels
#       - Keeps 6 channels (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
#       - Though MobiAct dataset contains labelled folders, this uses the annotated `label` column
#
# 3. Compuates global mean/std across the entire dataset (pass 1)
# 4. Reprocesses all files, applies global normalization, windows into 6s chunks (120 samples, no overlap)
#   - Two methods of assigning label per window:
#       a. Assigns label per window by majority vote. Ideal.
#       b. Assigns label based on the first label of the window. Not ideal, but currently using.
# 5. Prefall is set as the window right before the fall. All other windows are set as NON.
# 6. Converts label strings to integer class IDs.
# 7. Optional: saves train/val/test splits and metadata (deprecated).

import os, glob, json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

BASE_DIR = r"./data"
OUT_DIR  = r"./output_data\MobiAct"
os.makedirs(OUT_DIR, exist_ok=True)

PRE_FALL_SECONDS = 6.0
FS_TARGET = 20.0
PRE_FALL_SAMPLES = int(PRE_FALL_SECONDS * FS_TARGET)
WINDOW_SIZE = 120
STRIDE = 60
CUTOFF = 8.0

FEATURE_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TIME_COL = "rel_time"
LABEL_COL = "label"

########################## Functions ############################

def lowpass_filter(signal, fs=FS_TARGET, cutoff=CUTOFF, order=4):
    nyq = 0.5 * fs
    norm_cut = cutoff / nyq
    b, a = butter(order, norm_cut, btype="low")
    return filtfilt(b, a, signal, axis=0)

# fix uneven timestamps into a new time axis that is x Hz
def resample_timeseries(df, tcol=TIME_COL, fs=FS_TARGET):
    ts = df[tcol].astype(float).values

    if np.median(ts) > 1e5:
        ts = ts / 1000.0  # convert ms -> sec

    t0, t1 = ts[0], ts[-1]
    n_samples = int(np.floor((t1 - t0) * fs)) + 1
    new_t = np.linspace(t0, t0 + (n_samples - 1) / fs, n_samples)

    out = {tcol: new_t}

    # interpolate sensors
    for col in FEATURE_COLS:
        out[col] = np.interp(new_t, ts, df[col].values)

    labels_df = pd.DataFrame({tcol: ts, LABEL_COL: df[LABEL_COL].astype(str).str.strip().str.upper().values})
    labels_df = labels_df.sort_values(tcol).drop_duplicates(subset=tcol, keep="first")
    labels_df = labels_df.set_index(tcol).reindex(new_t, method="ffill")

    out[LABEL_COL] = labels_df[LABEL_COL].values
    return pd.DataFrame(out)

# add prefall label when the first fall is detected
def add_prefall_after_resample(labels, pre_samples=PRE_FALL_SAMPLES):
    labels = np.array([str(x).strip().upper() for x in labels], dtype=object)
    for i in range(1, len(labels)):
        if labels[i] == "FALL" and labels[i-1] != "FALL":
            start = max(0, i - pre_samples)
            labels[start:i] = "PRE_FALL"

    return labels

# segment data into windows
def window_segments(data_array, window_size=WINDOW_SIZE, stride=STRIDE):
    N = data_array.shape[0]
    windows = []

    for start in range(0, N - window_size + 1, stride):
        windows.append(data_array[start:start+window_size])

    if not windows:
        return np.zeros((0, window_size, data_array.shape[1]))

    return np.stack(windows, axis=0)

# returns labelled windows. currently using option b. choose first label in each window.
def window_labels(label_series, window_size=WINDOW_SIZE, stride=STRIDE):
    labels_out = []
    N = len(label_series)

    # clean string
    label_series = np.array([str(x).strip().upper() for x in label_series], dtype=object)

    for start in range(0, N - window_size + 1, stride): # 0, stride, 2xstride, 3xstride, ..., until final window
        first = label_series[start]

        if first == "FALL":
            labels_out.append("FALL")
        elif first == "PRE_FALL":
            labels_out.append("PRE_FALL")
        else:
            labels_out.append("NON")

    return np.array(labels_out, dtype=object)




################### START ########################

csv_files = glob.glob(os.path.join(BASE_DIR, "*", "*.csv"))
print(f"Found {len(csv_files)} CSV files.")


all_filtered_signals = []

# load and resample files
for csv_path in csv_files:
    df = pd.read_csv(csv_path)

    df_resampled = resample_timeseries(df, TIME_COL, FS_TARGET)
    sig = df_resampled[FEATURE_COLS].values
    sig_filt = lowpass_filter(sig)

    all_filtered_signals.append(sig_filt)

# PASS 1: compute global normalization stats
total = np.vstack(all_filtered_signals)
global_mean = total.mean(axis=0)
global_std  = total.std(axis=0)

print("Global mean:", global_mean)
print("Global std:", global_std)

all_windows = []
all_labels  = []
# PASS 2: normalize, add prefall, window, label
for csv_path in csv_files:
    df = pd.read_csv(csv_path)

    df_resampled = resample_timeseries(df)
    labels = add_prefall_after_resample(df_resampled[LABEL_COL].values)

    # safety check: FALL must have PRE_FALL 
    fall_count = np.sum(labels == "FALL")
    prefall_count = np.sum(labels == "PRE_FALL")

    if fall_count > 0 and prefall_count == 0:
        raise RuntimeError(
            f"ERROR: File {os.path.basename(csv_path)} contains FALL but NO PRE_FALL.\n"
            "This means PRE_FALL insertion failed for this file."
        )


    # convert all other labels to NON
    labels_final = np.array(labels)
    labels_final[(labels_final != "FALL") & (labels_final != "PRE_FALL")] = "NON"
    df_resampled[LABEL_COL] = labels_final

    # filter + normalize
    sig = df_resampled[FEATURE_COLS].values
    sig_filt = lowpass_filter(sig)
    sig_norm = (sig_filt - global_mean) / (global_std + 1e-8)

    # window data into chunks and add labels to windows
    win = window_segments(sig_norm)
    wlab = window_labels(df_resampled[LABEL_COL].values)

    all_windows.append(win)
    all_labels.append(wlab)

# merge all windows. in NumPy format to be utilized by LIMU-BERT
X_all = np.concatenate(all_windows, axis=0)
y_all = np.concatenate(all_labels, axis=0)

print("Final merged labels:", np.unique(y_all, return_counts=True))

label_map = {"NON": 0, "PRE_FALL": 1, "FALL": 2}
y_all_int = np.array([label_map[x] for x in y_all])

# Save files
np.save(os.path.join(OUT_DIR, "windows.npy"), X_all)
np.save(os.path.join(OUT_DIR, "labels.npy"), y_all_int)

# Save normalization stats
stats = {
    "mean": global_mean.tolist(),
    "std": global_std.tolist(),
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "fs": FS_TARGET,
    "label_map": label_map
}
with open(os.path.join(OUT_DIR, "norm_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print("Files saved to:", OUT_DIR)
