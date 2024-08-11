import numpy as np
import pandas as pd
from tkinter import filedialog
import os
from copy import deepcopy
from rich.progress import track

# parameters
consecutive_window = 2
threshold_alpha = 1
threshold_D = 0.10  # um2/s
threshold_R2 = 0.5
threshold_step_size = 2  # pixels
disp_threshold = 5


def find_consecutive_true_ranges(bool_array):
    ranges = []
    n = len(bool_array)
    i = 0

    while i < n:
        if bool_array.iloc[i]:
            start = i
            while i < n and bool_array[i]:
                i += 1
            end = i - 1
            ranges.append((start, end))
        i += 1

    return ranges


def flag_tracks(df):
    track_flags = {}
    step_flags = []
    trackIDs = df["trackID"].unique()

    for track_id in track(trackIDs, description="Processing tracks..."):
        track_data = df[df["trackID"] == track_id].reset_index(drop=True)
        step_flags_pertrack = np.repeat(False, track_data.shape[0])
        high_alpha = track_data["alpha"] > threshold_alpha
        ranges = find_consecutive_true_ranges(high_alpha)
        for start, end in ranges:
            if end - start + 1 >= consecutive_window:
                mean_D = np.mean(track_data["D"][start : end + 1])
                mean_R2 = np.mean(track_data["R2"][start : end + 1])
                mean_stepsize = np.mean(track_data["step_sizes"][start : end + 1])
                x_start = track_data["x"][start]
                x_end = track_data["x"][end]
                y_start = track_data["y"][start]
                y_end = track_data["y"][end]
                disp = np.sqrt((x_start - x_end) ** 2 + (y_start - y_end) ** 2)
                if (
                    mean_D > threshold_D
                    and mean_R2 > threshold_R2
                    and mean_stepsize > threshold_step_size
                    and disp > disp_threshold
                ):
                    track_flags[track_id] = True
                    step_flags_pertrack[start : end + 1] = True

        step_flags.extend(step_flags_pertrack)

    df_out = deepcopy(df)
    df_out["step_flag"] = step_flags

    return df_out, track_flags


csv_file_path = filedialog.askopenfilename(
    title="Select CSV File",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
)

df = pd.read_csv(csv_file_path)
df_out, track_flags = flag_tracks(df)

df_out.to_csv(csv_file_path[:-4] + "-wstepflags.csv", index=False)
print(len(track_flags))
for track_id, flagged in track_flags.items():
    if flagged:
        print(track_id)
