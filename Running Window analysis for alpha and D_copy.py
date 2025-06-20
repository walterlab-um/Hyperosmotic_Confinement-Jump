import numpy as np
import pandas as pd
from scipy.stats import linregress
from math import ceil
import os

# Required constants
um_per_pixel = 0.117
s_per_frame = 2
window_size = 20

# Data type definition
dtype_dict = {
    "t": "float64",
    "x": "float64",
    "y": "float64",
    "trackID": "Int64",
}

csv_file_path = r"Z:\Bisal_Halder_turbo\PROCESSED_DATA\Impact_of_cytoskeleton_on_HOPS_condensates\no_drug\Analysed Data\2x\Trackmate analysis new 07052024\alpha, D and step sizes_w20\20240118_UGD-2x-2s-replicate1-FOV-2_processed_step_size_w20.csv"

df = pd.read_csv(csv_file_path, dtype=dtype_dict)

# Data cleaning and type conversion
df = df.dropna(subset=["t", "x", "y"])
df["trackID"] = pd.to_numeric(df["trackID"], errors="coerce").astype("Int64")
df = df.dropna(subset=["trackID"])

# Data scaling
df["t"] *= s_per_frame
df["x"] *= um_per_pixel
df["y"] *= um_per_pixel


def calc_MSD_NonPhysUnit(track_data, lags):
    Xs = track_data["x"].to_numpy()
    Ys = track_data["y"].to_numpy()
    MSDs = []
    for lag in lags:
        displacements = (Xs[:-lag] - Xs[lag:]) ** 2 + (Ys[:-lag] - Ys[lag:]) ** 2
        valid_displacements = displacements[~np.isnan(displacements)]
        MSD = np.nanmean(valid_displacements)
        MSDs.append(MSD)
    return np.array(MSDs, dtype=float)


def calc_alpha(MSDs, lags):
    valid_indices = ~np.isnan(MSDs)
    valid_MSDs = MSDs[valid_indices]
    valid_lags = lags[valid_indices]
    log_lags = np.log10(valid_lags)
    log_MSDs = np.log10(valid_MSDs)
    slope, intercept, r_value, _, _ = linregress(log_lags, log_MSDs)
    alpha = slope
    diffusion_coefficient = (1 / 4) * (10**intercept)
    r_squared = r_value**2
    return alpha, r_squared, diffusion_coefficient


def calculate_alpha_and_D_for_track(df_track, window_size):
    df_track["R2"] = np.nan
    df_track["alpha"] = np.nan
    df_track["D"] = np.nan
    step_size = 1
    for start in range(0, len(df_track) - window_size + 1, step_size):
        end = start + window_size
        df_window = df_track.iloc[start:end]
        number_lag = ceil(window_size / 2)
        if number_lag < 3:
            number_lag = 3
        window_msd = calc_MSD_NonPhysUnit(df_window, np.arange(1, number_lag + 1))
        if np.sum(window_msd <= 0) > 0:
            continue
        alpha, r_squared, D = calc_alpha(
            window_msd, np.arange(1, number_lag + 1) * s_per_frame
        )
        if not np.isnan(alpha):
            middle_frame_index = start + ceil((window_size - 1) / 2)
            df_track.at[df_track.index[middle_frame_index], "R2"] = r_squared
            df_track.at[df_track.index[middle_frame_index], "alpha"] = alpha
            df_track.at[df_track.index[middle_frame_index], "D"] = D
    return df_track


def process_trackID_195(csv_file_path, window_size, um_per_pixel, s_per_frame):
    df = pd.read_csv(csv_file_path, dtype=dtype_dict)
    df = df.dropna(subset=["t", "x", "y"])
    df["trackID"] = pd.to_numeric(df["trackID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["trackID"])

    df["t"] *= s_per_frame
    df["x"] *= um_per_pixel
    df["y"] *= um_per_pixel

    # Filter for trackID 195
    df_track_195 = df[df["trackID"] == 195].copy()

    processed_track = calculate_alpha_and_D_for_track(df_track_195, window_size)

    output_df = processed_track[["trackID", "x", "y", "t", "R2", "alpha", "D"]]

    save_path = os.path.dirname(csv_file_path)
    base_name = os.path.basename(csv_file_path)
    name, ext = os.path.splitext(base_name)

    output_file_name = f"{name}_trackID_195_alpha_D_results.csv"
    output_file_path = os.path.join(save_path, output_file_name)

    output_df.to_csv(output_file_path, index=False)
    print(f"Processed CSV file saved: {output_file_path}")

    return output_file_path


process_trackID_195(csv_file_path, window_size, um_per_pixel, s_per_frame)
