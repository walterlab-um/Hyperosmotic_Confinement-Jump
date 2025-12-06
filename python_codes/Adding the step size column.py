import pandas as pd
import numpy as np
from rich.progress import track
from tkinter import filedialog
import os

pd.options.mode.chained_assignment = None

um_per_pixel = 0.117
s_per_frame = 2


def calculate_step_sizes_and_add_column(csv_file_path):
    dtype_dict = {"t": "float64", "x": "float64", "y": "float64", "trackID": "object"}

    df = pd.read_csv(csv_file_path, dtype=dtype_dict)

    df = df.dropna(subset=["t", "x", "y"])
    df["trackID"] = pd.to_numeric(df["trackID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["trackID"])

    df["step_sizes"] = np.nan
    df["mean_step_sizes"] = np.nan

    track_ids = df["trackID"].unique()

    for track_id in track(track_ids, description="Processing tracks..."):
        track_data = df[df["trackID"] == track_id]
        sorted_track_data = track_data.sort_values(by="t")

        x_diffs = np.diff(sorted_track_data["x"])
        y_diffs = np.diff(sorted_track_data["y"])
        step_sizes = np.sqrt(x_diffs**2 + y_diffs**2)

        step_sizes = np.insert(step_sizes, 0, np.nan)
        sorted_track_data["step_sizes"] = step_sizes

        mean_step_sizes = np.nanmean(step_sizes)
        sorted_track_data["mean_step_sizes"] = mean_step_sizes

        df.loc[sorted_track_data.index, "step_sizes"] = step_sizes
        df.loc[sorted_track_data.index, "mean_step_sizes"] = mean_step_sizes

    return df


csv_file_path = filedialog.askopenfilename(
    title="Select CSV Files for No drug_2x",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
)

modified_df = calculate_step_sizes_and_add_column(csv_file_path)

directory = os.path.dirname(csv_file_path)
base_file_name = os.path.splitext(os.path.basename(csv_file_path))[0]
new_file_name = f"{base_file_name}_with_step_sizes.csv"
new_csv_path = os.path.join(directory, new_file_name)

modified_df.to_csv(new_csv_path, index=False)
