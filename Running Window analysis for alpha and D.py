import scipy.stats as stats
import numpy as np
import pandas as pd
from tkinter import filedialog
import os
from math import ceil
from rich.progress import track

# Disable chained assignment warning
pd.options.mode.chained_assignment = None

um_per_pixel = 0.117
s_per_frame = 2
window_size = 20
dtype_dict = {
    "t": "float64",
    "x": "float64",
    "y": "float64",
    "trackID": "Int64",
}

# File selection dialog (make sure to run this in an environment that supports dialogs, or replace with a direct file path)
csv_file_path = filedialog.askopenfilename(
    title="Select CSV File",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
)

# Read the CSV file with predefined data types
df = pd.read_csv(csv_file_path, dtype=dtype_dict)

# Data cleaning and type conversion
df = df.dropna(subset=["t", "x", "y"])
df["trackID"] = pd.to_numeric(df["trackID"], errors="coerce").astype("Int64")
df = df.dropna(subset=["trackID"])

# Data scaling
df["t"] *= s_per_frame
df["x"] *= um_per_pixel
df["y"] *= um_per_pixel

# df[["trackID", "x", "y", "t"]]


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


def calc_alpha_and_diffusion(MSDs, lags):
    valid_indices = ~np.isnan(MSDs)
    valid_MSDs = MSDs[valid_indices]
    valid_lags = lags[valid_indices]

    # Log-log calculation
    log_lags = np.log10(valid_lags)
    log_MSDs = np.log10(valid_MSDs)
    slope_loglog, intercept_loglog, r_value_loglog, _, _ = stats.linregress(
        log_lags, log_MSDs
    )

    alpha = slope_loglog
    diffusion_coefficient_loglog = (1 / 4) * (10**intercept_loglog)
    r_squared_loglog = r_value_loglog**2

    # Linear calculation
    slope_linear, _, r_value_linear, _, _ = stats.linregress(valid_lags, valid_MSDs)
    diffusion_coefficient_linear = slope_linear / (8 / 3)  # um^2/s
    r_squared_linear = r_value_linear**2

    return (
        alpha,
        r_squared_loglog,
        diffusion_coefficient_loglog,
        r_squared_linear,
        diffusion_coefficient_linear,
    )


def calculate_alpha_and_D_for_track(df_track, um_per_pixel, s_per_frame, window_size):
    df_track["R2_loglog"] = np.nan
    df_track["alpha"] = np.nan
    df_track["D_loglog"] = np.nan
    df_track["R2_linear"] = np.nan
    df_track["D_linear"] = np.nan

    step_size = 1
    for start in range(0, len(df_track) - window_size + 1, step_size):
        end = start + window_size
        df_window = df_track.iloc[start:end]

        number_lag = ceil(window_size / 2)
        if number_lag < 3:
            number_lag = 3
        window_msd = calc_MSD_NonPhysUnit(df_window, np.arange(1, number_lag + 1))
        if np.sum(window_msd <= 0) > 0:
            # Skip this window since it contains invalid MSD values
            continue

        alpha, r_squared_loglog, D_loglog, r_squared_linear, D_linear = (
            calc_alpha_and_diffusion(
                window_msd, np.arange(1, number_lag + 1) * s_per_frame
            )
        )

        if not np.isnan(alpha):
            middle_frame_index = start + ceil(window_size / 2)
            df_track.at[df_track.index[middle_frame_index], "R2_loglog"] = (
                r_squared_loglog
            )
            df_track.at[df_track.index[middle_frame_index], "alpha"] = alpha
            df_track.at[df_track.index[middle_frame_index], "D_loglog"] = D_loglog
            df_track.at[df_track.index[middle_frame_index], "R2_linear"] = (
                r_squared_linear
            )
            df_track.at[df_track.index[middle_frame_index], "D_linear"] = D_linear

    return df_track


def process_csv_and_add_alpha_and_D(
    csv_file_path, window_size, um_per_pixel, s_per_frame
):
    df = pd.read_csv(csv_file_path, dtype=dtype_dict)
    processed_track_list = []

    grouped_tracks = df.groupby("trackID")
    for track_id, df_track in track(grouped_tracks, description="Processing tracks"):
        processed_track = calculate_alpha_and_D_for_track(
            df_track, um_per_pixel, s_per_frame, window_size
        )
        processed_track_list.append(
            processed_track[
                [
                    "trackID",
                    "x",
                    "y",
                    "t",
                    "R2_loglog",
                    "alpha",
                    "D_loglog",
                    "R2_linear",
                    "D_linear",
                ]
            ]
        )

    processed_df = pd.concat(processed_track_list).reset_index(drop=True)

    save_path = os.path.dirname(csv_file_path)
    base_name = os.path.basename(csv_file_path)
    name, ext = os.path.splitext(base_name)

    output_file_name = f"{name}_processed_alpha_and_D_w{window_size}{ext}"
    output_file_path = os.path.join(save_path, output_file_name)

    processed_df.to_csv(output_file_path, index=False)
    print(f"Processed CSV file saved: {output_file_path}")

    return output_file_path


process_csv_and_add_alpha_and_D(csv_file_path, window_size, um_per_pixel, s_per_frame)
