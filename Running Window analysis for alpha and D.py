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


def calc_alpha(MSDs, lags):

    valid_indices = ~np.isnan(MSDs)
    valid_MSDs = MSDs[valid_indices]
    valid_lags = lags[valid_indices]

    log_lags = np.log10(valid_lags)
    log_MSDs = np.log10(valid_MSDs)

    slope, intercept, r_value, p_value, std_error = stats.linregress(log_lags, log_MSDs)

    alpha = slope
    diffusion_coefficient = (1 / 4) * (10**intercept)
    r_squared = r_value**2

    return alpha, r_squared, diffusion_coefficient


def calculate_alpha_and_D_for_track(df_track, um_per_pixel, s_per_frame, window_size):

    df_track["R2"] = np.nan
    df_track["alpha"] = np.nan
    df_track["D"] = np.nan

    # window_info = []

    # Iterate through windows and update alpha for the middle frame
    step_size = 1
    for start in range(0, len(df_track) - window_size + 1, step_size):
        end = start + window_size
        df_window = df_track.iloc[start:end]

        number_lag = ceil(window_size / 2)
        if number_lag < 3:
            number_lag = 3
        window_msd = calc_MSD_NonPhysUnit(
            df_track.iloc[start:end], np.arange(1, number_lag + 1)
        )
        if np.sum(window_msd <= 0) > 0:
            # Skip this window since it contains invalid MSD values
            continue

        alpha, r_squared, D = calc_alpha(
            window_msd, np.arange(1, number_lag + 1) * s_per_frame
        )
        # r_squared = r_value**2
        if not np.isnan(alpha):
            middle_frame_index = start + ceil(window_size / 2)
            df_track.at[df_track.index[middle_frame_index], "R2"] = r_squared
            df_track.at[df_track.index[middle_frame_index], "alpha"] = alpha
            df_track.at[df_track.index[middle_frame_index], "D"] = D

            # window_info.append(
            #     {
            #         "window_start_frame": df_window.iloc[0]["t"],
            #         "window_end_frame": df_window.iloc[-1]["t"],
            #         "alpha": alpha,
            #         "R2": r_squared,
            #         "D": D,
            #         "lags": np.arange(1, number_lag + 1).tolist(),
            #     }
            # )

    return df_track


def process_csv_and_add_alpha_and_D(
    csv_file_path, window_size, um_per_pixel, s_per_frame
):

    df = pd.read_csv(csv_file_path, dtype=dtype_dict)
    # df["t"] *= s_per_frame
    # df["x"] *= um_per_pixel
    # df["y"] *= um_per_pixel

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
                    "R2",
                    "alpha",
                    "D",
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
