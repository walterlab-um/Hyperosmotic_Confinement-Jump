import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.progress import track
from tkinter import filedialog
import os
from statannot import add_stat_annotation

pd.options.mode.chained_assignment = None  # default='warn'


# scalling factors for physical units
um_per_pixel = 0.117
frame_duration = 2  # in seconds
minimum_frames = 20  # Minimum number of frames a track must be present


def calculate_step_sizes(csv_file_path):
    dtype_dict = {
        "POSITION_T": "float64",
        "POSITION_X": "float64",
        "POSITION_Y": "float64",
        "TRACK_ID": "object",
    }

    df = pd.read_csv(csv_file_path, dtype=dtype_dict)

    df = df.dropna(subset=["POSITION_T", "POSITION_X", "POSITION_Y"])
    df["TRACK_ID"] = pd.to_numeric(df["TRACK_ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["TRACK_ID"])

    df["POSITION_T"] *= frame_duration
    df["POSITION_X"] *= um_per_pixel
    df["POSITION_Y"] *= um_per_pixel

    # Filter out tracks with less than minimum_frames frames
    track_counts = df["TRACK_ID"].value_counts()
    valid_tracks = track_counts[
        (track_counts >= minimum_frames) & (track_counts <= 190)
    ].index
    df = df[df["TRACK_ID"].isin(valid_tracks)]

    all_step_sizes = []

    # Loop through all track IDs
    for track_id in df["TRACK_ID"].tolist():
        track_data = df[df["TRACK_ID"] == track_id]

        # Sort track data by time
        sorted_track_data = track_data.sort_values(by="POSITION_T")

        # Extract x and y coordinates
        x_coordinates = sorted_track_data["POSITION_X"].to_numpy()
        y_coordinates = sorted_track_data["POSITION_Y"].to_numpy()

        # Calculate step size for each adjacent pair of coordinates
        step_size = np.sqrt(
            (x_coordinates[1:] - x_coordinates[:-1]) ** 2
            + (y_coordinates[1:] - y_coordinates[:-1]) ** 2
        )
        all_step_sizes.extend(step_size.tolist())

    return all_step_sizes


def calculate_fraction_step_sizes_greater_than_0_5(all_step_sizes):
    fraction_gt_0_5 = (
        np.sum(np.array(all_step_sizes) > 0.5) / len(all_step_sizes)
        if all_step_sizes
        else 0
    )
    return fraction_gt_0_5


def process_csv_files_condition(csv_files, condition_label):
    condition_fractions = {}
    for csv_file in track(
        csv_files, description=f"Processing CSV files for {condition_label}"
    ):
        step_sizes = calculate_step_sizes(csv_file)
        fraction_gt_0_5 = calculate_fraction_step_sizes_greater_than_0_5(step_sizes)
        filename = os.path.basename(csv_file)
        condition_fractions[filename] = fraction_gt_0_5
    return condition_fractions


# Select the CSV files for the three experiments
csv_files_1 = filedialog.askopenfilenames(
    title="Select CSV Files for No drug_2x",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
)
csv_files_2 = filedialog.askopenfilenames(
    title="Select CSV Files for LatrunculinA_30 mins",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
)
csv_files_3 = filedialog.askopenfilenames(
    title="Select CSV Files for LatrunculinA_60 mins",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
)

# Calculate step sizes for the three experiments
step_sizes_1 = []
for csv_file in track(csv_files_1):
    step_sizes_1.extend(calculate_step_sizes(csv_file))

step_sizes_2 = []
for csv_file in track(csv_files_2):
    step_sizes_2.extend(calculate_step_sizes(csv_file))

step_sizes_3 = []
for csv_file in track(csv_files_3):
    step_sizes_3.extend(calculate_step_sizes(csv_file))


# Create the data DataFrame
label1 = "No drug_2x"
label2 = "LatrunculinA_30 mins"
label3 = "LatrunculinA_60 mins"

fractions_1 = process_csv_files_condition(csv_files_1, label1)
fractions_2 = process_csv_files_condition(csv_files_2, label2)
fractions_3 = process_csv_files_condition(csv_files_3, label3)

# Combine all the fractions into one dictionary keyed by condition label
all_fractions = {
    label1: fractions_1,
    label2: fractions_2,
    label3: fractions_3,
}

dataframes = []
for label, fractions in all_fractions.items():
    df = pd.DataFrame(list(fractions.items()), columns=["filename", "fraction_gt_0_5"])
    df["condition"] = label
    dataframes.append(df)
# Final DataFrame combining all conditions
final_dataframe = pd.concat(dataframes, ignore_index=True)

# Save the dataframe to CSV file
save_path = r"Z:\Bisal_Halder_turbo\PROCESSED_DATA\Trial_analysis"
csv_file_name = "fractions_step_size_gt_0_5_um.csv"
csv_file_path = os.path.join(save_path, csv_file_name)
final_dataframe.to_csv(csv_file_path, index=False)

plt.figure(figsize=(4, 3), dpi=300)
ax = sns.barplot(
    data=final_dataframe,
    x="condition",
    y="fraction_gt_0_5",
    order=final_dataframe["condition"].unique(),
    ci="sd",
    capsize=0.1,
)

sns.stripplot(
    data=final_dataframe,
    x="condition",
    y="fraction_gt_0_5",
    color="0.7",
    size=3,
    order=final_dataframe["condition"].unique(),
)

box_pairs = [
    ("No drug_2x", "LatrunculinA_30 mins"),
    ("No drug_2x", "LatrunculinA_60 mins"),
    ("LatrunculinA_30 mins", "LatrunculinA_60 mins"),
]

test_stats = add_stat_annotation(
    ax,
    data=final_dataframe,
    x="condition",
    y="fraction_gt_0_5",
    box_pairs=box_pairs,
    test="t-test_welch",
    comparisons_correction=None,
    text_format="star",
    loc="inside",
    verbose=2,
)

plt.xticks(rotation=45, ha="right", fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel("Fraction of step sizes > 0.5 um", fontsize=5)
plt.ylim(0, None)
plt.grid(False)
plt.tight_layout()
plt.savefig("fraction_bar_plot.png", dpi=300, format="png")
plt.show()
