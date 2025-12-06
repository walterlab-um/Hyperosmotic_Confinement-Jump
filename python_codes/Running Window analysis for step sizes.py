import numpy as np
import pandas as pd
from tkinter import filedialog
import os
from rich.progress import track
from math import ceil

pd.options.mode.chained_assignment = None

um_per_pixel = 0.117
s_per_frame = 2
window_size = 20
minimal_datapoints = 5
dtype_dict = {
    "t": "float64",
    "x": "float64",
    "y": "float64",
    "trackID": "Int64",
}

csv_file_path = filedialog.askopenfilename(
    title="Select CSV File",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
)

df = pd.read_csv(csv_file_path, dtype=dtype_dict)

df = df.dropna(subset=["t", "x", "y"])
df["trackID"] = pd.to_numeric(df["trackID"], errors="coerce").astype("Int64")
df = df.dropna(subset=["trackID"])

# Data scaling
df["t"] *= s_per_frame
df["x"] *= um_per_pixel
df["y"] *= um_per_pixel


def calculate_step_size_for_track(df_track, window_size):
    df_track["step_size_w20"] = np.nan

    step_size = 1
    # Iterate through windows to calculate step size for the middle frame
    for start in range(0, len(df_track) - window_size + 1, step_size):
        end = start + window_size
        middle_frame_index = start + ceil(window_size / 2)

        # Calculate mean step size for the segment
        segment = df_track.iloc[start:end]
        x_diffs = np.diff(segment["x"].to_numpy())
        y_diffs = np.diff(segment["y"].to_numpy())
        step_sizes = np.sqrt(x_diffs**2 + y_diffs**2)
        mean_step_size = np.mean(step_sizes)

        df_track.at[df_track.index[middle_frame_index], "step_size_w20"] = (
            mean_step_size
        )

    return df_track


def process_csv_and_add_step_size(csv_file_path, window_size):
    df = pd.read_csv(csv_file_path, dtype=dtype_dict)

    processed_track_list = []

    grouped_tracks = df.groupby("trackID")
    for track_id, df_track in track(grouped_tracks, description="Processing tracks"):
        processed_track = calculate_step_size_for_track(df_track, window_size)
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
                    "step_sizes",
                    "mean_step_sizes",
                    # "x_mean",
                    # "y_mean",
                    # "cellID",
                    "step_size_w20",
                ]
            ]
        )

    processed_df = pd.concat(processed_track_list).reset_index(drop=True)

    save_path = os.path.dirname(csv_file_path)
    base_name = os.path.basename(csv_file_path)
    name, ext = os.path.splitext(base_name)

    output_file_name = f"{name}_processed_step_size_w{window_size}{ext}"
    output_file_path = os.path.join(save_path, output_file_name)

    processed_df.to_csv(output_file_path, index=False)
    print(f"Processed CSV file saved: {output_file_path}")

    return output_file_path


process_csv_and_add_step_size(csv_file_path, window_size)
