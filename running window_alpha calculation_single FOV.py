import scipy.stats as stats
import numpy as np
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import ceil
from rich.progress import track

# Disable chained assignment warning
pd.options.mode.chained_assignment = None

# Scaling factors
um_per_pixel = 0.117
s_per_frame = 2
window_size = 5

# Specify column data types
dtype_dict = {
    "POSITION_T": "float64",
    "POSITION_X": "float64",
    "POSITION_Y": "float64",
    "TRACK_ID": "str",
}


# Define your functions here
def calc_MSD_NonPhysUnit(df_track_sorted, lags):
    # Assume df_track_sorted has already been preprocessed and scaled
    Xs = df_track_sorted["POSITION_X"].to_numpy()
    Ys = df_track_sorted["POSITION_Y"].to_numpy()

    MSDs = []
    for lag in lags:
        displacements = (Xs[:-lag] - Xs[lag:]) ** 2 + (Ys[:-lag] - Ys[lag:]) ** 2
        valid_displacements = displacements[
            ~np.isnan(displacements)
        ]  # Filter out NaN values
        MSD = np.nanmean(valid_displacements)
        MSDs.append(MSD)

    return np.array(MSDs, dtype=float)


def calc_alpha(MSDs, lags):
    # Filter out NaN values from MSDs and corresponding lags
    valid_indices = ~np.isnan(MSDs)
    valid_MSDs = MSDs[valid_indices]
    valid_lags = lags[valid_indices]  # lags should already be in real time units

    log_lags = np.log10(valid_lags)
    log_MSDs = np.log10(valid_MSDs)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_MSDs)

    diffusion_coefficient = (1 / 4) * (10**intercept)
    return slope, r_value, diffusion_coefficient


# Function to calculate alpha values for a given set of CSV files with sliding window
def calculate_alpha_for_track(df_track, um_per_pixel, s_per_frame, window_size):
    # Initialize the new columns for the track with NaN
    df_track["alpha"] = np.nan
    df_track["R2"] = np.nan
    df_track["D"] = np.nan
    df_track["POSITION_X"] *= um_per_pixel
    df_track["POSITION_Y"] *= um_per_pixel
    x = df_track["POSITION_X"].to_numpy()
    y = df_track["POSITION_Y"].to_numpy()

    # Iterate through windows and update alpha for the middle frame
    step_size = 1
    for start in range(0, len(x) - window_size + 1, step_size):
        end = start + window_size
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
        if not np.isnan(alpha):
            middle_frame_index = (
                start + window_size // 2
            )  # Integer division to get middle index
            df_track.at[df_track.index[middle_frame_index], "alpha"] = alpha
            df_track.at[df_track.index[middle_frame_index], "R2"] = r_squared
            df_track.at[df_track.index[middle_frame_index], "D"] = D

    return df_track


def process_csv_and_add_alpha(csv_file, window_size, um_per_pixel, s_per_frame):
    save_path = os.path.dirname(csv_file)
    base_name = os.path.basename(csv_file)
    name, ext = os.path.splitext(base_name)
    output_file_name = f"{name}_alpha_w{window_size}{ext}"
    output_file_path = os.path.join(save_path, output_file_name)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Preprocess position columns
    df["POSITION_X"] *= um_per_pixel
    df["POSITION_Y"] *= um_per_pixel

    # Process each track and calculate alphas
    tracks = df.groupby("TRACK_ID")
    alpha_df_list = []
    for track_id, track_df in tracks:
        alpha_df = calculate_alpha_for_track(
            track_df.copy(), window_size, s_per_frame, window_size
        )
        alpha_df_list.append(alpha_df)

    # Combine all the individual track DataFrames into one DataFrame
    result_df = pd.concat(alpha_df_list, ignore_index=True)

    # Save the result to a new CSV file
    result_df.to_csv(output_file_path, index=False)
    print(f"Processed file saved: {output_file_path}")
    return output_file_path


# Example usage:
# Select CSV files for each experiment
input_csv_file = filedialog.askopenfilename(
    title="Select CSV File",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
)

# Trigger the processing function
process_csv_and_add_alpha(input_csv_file, window_size, um_per_pixel, s_per_frame)
