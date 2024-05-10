import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.progress import track
from tkinter import filedialog

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


# Create a Tkinter root window
# root = tk.Tk()
# root.withdraw()  # Hide the root window

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
data = pd.concat(
    [
        pd.DataFrame({"step_sizes": step_sizes_1, "label": label1}),
        pd.DataFrame({"step_sizes": step_sizes_2, "label": label2}),
        pd.DataFrame({"step_sizes": step_sizes_3, "label": label3}),
    ],
    ignore_index=True,
)

# Generate CDF plot
plt.figure(figsize=(4, 3))
palette = ["skyblue", "orange", "green"]

ax = sns.histplot(
    data=data,
    x="step_sizes",
    hue="label",
    palette=palette,
    bins=17,
    binrange=(0, 0.8),
    stat="probability",
    common_norm=False,
    lw=2,
    element="step",
    fill=False,
    cumulative=False,
)

plt.xlim(0, 0.8)
sns.move_legend(
    ax,
    4,
    title=None,
    frameon=False,
    fontsize=18,
)
plt.xlabel("Step Size ($\mu$m)", fontsize=18)
plt.ylabel("Probabilty", fontsize=18)
plt.gca().spines[:].set_linewidth(1)
plt.gca().tick_params(
    axis="both",
    which="major",
    labelsize=18,
    direction="in",
    bottom=True,
    left=True,
    length=5,
    width=1,
)
plt.gca().set_axisbelow(False)
plt.gca().tick_params(axis="both", which="major", labelsize=18)
plt.savefig(
    "step-cdf.png",
    format="png",
    bbox_inches="tight",
    dpi=300,
)
plt.show()
