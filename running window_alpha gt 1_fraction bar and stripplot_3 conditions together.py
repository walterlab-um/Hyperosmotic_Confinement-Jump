import scipy.stats as stats
import numpy as np
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import ceil
from rich.progress import track
from statannot import add_stat_annotation

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
    return slope  # Alpha value


# Function to calculate alpha values for a given set of CSV files with sliding window
def calculate_fraction_alpha_gt_threshold(csv_file):
    try:
        df_current_file = pd.read_csv(csv_file, dtype=dtype_dict)

        # Additional preprocessing as necessary, including scaling
        df_current_file["POSITION_X"] *= um_per_pixel
        df_current_file["POSITION_Y"] *= um_per_pixel
        df_current_file.dropna(
            subset=["POSITION_X", "POSITION_Y", "TRACK_ID"], inplace=True
        )
        df_current_file.sort_values(by="POSITION_T", inplace=True)

        # Process tracks and calculate alphas
        alpha_values = []
        for trackID in df_current_file["TRACK_ID"].unique():
            df_track = df_current_file[df_current_file["TRACK_ID"] == trackID]
            x = df_track["POSITION_X"].to_numpy()
            y = df_track["POSITION_Y"].to_numpy()

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
                    alpha_values.append(alpha)
                    continue

                alpha = calc_alpha(
                    window_msd, np.arange(1, number_lag + 1) * s_per_frame
                )
                if not np.isnan(alpha):
                    alpha_values.append(alpha)

        fraction_alpha = (
            sum(a > 1.5 for a in alpha_values) / len(alpha_values)
            if alpha_values
            else None
        )
        return fraction_alpha
    except Exception as e:
        print(f"Error processing CSV file: {csv_file}")
        print(e)
        return None


def process_condition_csv_files(csv_files):
    fractions_dict = {}
    for csv_file in track(csv_files, description="Processing CSV files"):
        fraction_alpha = calculate_fraction_alpha_gt_threshold(csv_file)
        if fraction_alpha is not None:  # Ensure that a fraction was properly calculated
            # Key the fraction by the filename for easier identification
            filename = os.path.basename(csv_file)
            fractions_dict[filename] = fraction_alpha
    return fractions_dict


# Select CSV files for each experiment
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

# Calculate fractions for each experimental condition
fractions_1 = process_condition_csv_files(csv_files_1)
fractions_2 = process_condition_csv_files(csv_files_2)
fractions_3 = process_condition_csv_files(csv_files_3)

# Combine all the fractions into one dictionary keyed by condition label
all_fractions = {
    "No drug_2x": fractions_1,
    "LatrunculinA_30 mins": fractions_2,
    "LatrunculinA_60 mins": fractions_3,
}

label1 = "No drug_2x"
label2 = "LatrunculinA_30 mins"
label3 = "LatrunculinA_60 mins"

# Convert the dictionaries to dataframes
fractions_df1 = pd.DataFrame(
    list(fractions_1.items()), columns=["filename", "fraction_alpha"]
)
fractions_df1["label"] = label1

fractions_df2 = pd.DataFrame(
    list(fractions_2.items()), columns=["filename", "fraction_alpha"]
)
fractions_df2["label"] = label2

fractions_df3 = pd.DataFrame(
    list(fractions_3.items()), columns=["filename", "fraction_alpha"]
)
fractions_df3["label"] = label3

# Concatenate the dataframes into a single dataframe
data = pd.concat([fractions_df1, fractions_df2, fractions_df3], ignore_index=True)

# Save path (ensure this directory exists and you have write permissions)
save_path = r"Z:\Bisal_Halder_turbo\PROCESSED_DATA\Trial_analysis"
csv_file_path = os.path.join(save_path, "alpha_LatrunculinA_window5_alpha.csv")
data.to_csv(csv_file_path, index=False)

plt.figure(figsize=(4, 3), dpi=300)
ax = sns.barplot(
    data=data,
    x="label",
    y="fraction_alpha",
    order=data["label"].unique(),
    ci="sd",
    capsize=0.1,
)

sns.stripplot(
    data=data,
    x="label",
    y="fraction_alpha",
    color="0.7",
    size=3,
    order=data["label"].unique(),  # Ensure order matches barplot for strip plot as well
)

box_pairs = [
    ("No drug_2x", "LatrunculinA_30 mins"),
    ("No drug_2x", "LatrunculinA_60 mins"),
    ("LatrunculinA_30 mins", "LatrunculinA_60 mins"),
]

test_stats = add_stat_annotation(
    ax,
    data=data,
    x="label",
    y="fraction_alpha",
    box_pairs=box_pairs,
    test="t-test_welch",
    comparisons_correction=None,
    text_format="star",
    loc="inside",
    verbose=2,
)

plt.xticks(rotation=45, ha="right", fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel("Fraction of alpha values > 1.5", fontsize=5)
plt.xlabel("")
plt.ylim(0, None)
plt.grid(False)
plt.tight_layout()
# plt.savefig("fraction_bar_plot.png", dpi=300, format="png")
plt.show()
