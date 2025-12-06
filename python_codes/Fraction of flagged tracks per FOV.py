import pandas as pd
from tkinter import filedialog
import os
from rich.progress import track


def has_consecutive_trues(series):
    for i in range(len(series) - 1):
        if series.iloc[i] and series.iloc[i + 1]:
            return True
    return False


def calculate_fraction_of_flagged_tracks_per_file(df):
    track_groups = df.groupby("trackID")
    flagged_tracks = 0
    total_tracks = len(track_groups)

    for _, track_group in track_groups:
        if has_consecutive_trues(track_group["step_flag"]):
            flagged_tracks += 1

    fraction_flagged = flagged_tracks / total_tracks if total_tracks > 0 else 0
    return fraction_flagged, flagged_tracks


csv_files = filedialog.askopenfilenames(
    title="Select CSV Files",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
)

all_fractions = []
for file in track(csv_files, description="Processing CSV files..."):
    df = pd.read_csv(file)
    df["step_flag"] = df["step_flag"].astype(str).str.upper() == "TRUE"
    fraction, flagged_count = calculate_fraction_of_flagged_tracks_per_file(df)
    all_fractions.append({"file": os.path.basename(file), "fraction": fraction})

    print(f"File: {os.path.basename(file)}")
    print(f"Flagged tracks: {flagged_count}")

results_df = pd.DataFrame(all_fractions)

output_file = os.path.join(
    os.path.dirname(csv_files[0]), "fraction_of_flagged_tracks_per_file.csv"
)
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
