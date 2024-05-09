import scipy.stats as stats
import numpy as np
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rich.progress import track

# Disable chained assignment warning
pd.options.mode.chained_assignment = None

# Scaling factors
um_per_pixel = 0.117
print("Please enter the seconds per frame for the video:")
s_per_frame = float(input())

# Initialization of window_size
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
    Xs = df_track_sorted['POSITION_X'].to_numpy()
    Ys = df_track_sorted['POSITION_Y'].to_numpy()

    MSDs = []
    for lag in lags:
        displacements = (Xs[:-lag] - Xs[lag:]) ** 2 + (Ys[:-lag] - Ys[lag:]) ** 2
        valid_displacements = displacements[~np.isnan(displacements)]  # Filter out NaN values
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
def calculate_alpha(csv_files):
    alpha_values = []
    for csv_file in track(csv_files, description="Processing CSV files"):
        try:
            # Load the data
            df_current_file = pd.read_csv(csv_file, dtype=dtype_dict)

            # Preprocessing steps, such as scaling and sorting, if not already done before
            df_current_file["POSITION_X"] *= um_per_pixel
            df_current_file["POSITION_Y"] *= um_per_pixel
            df_current_file = df_current_file.dropna(subset=["POSITION_X", "POSITION_Y", "TRACK_ID"])
            df_current_file.sort_values(by="POSITION_T", inplace=True, ignore_index=True)

            # Count the number of frames for each track
            track_counts = df_current_file.groupby("TRACK_ID")["POSITION_T"].count()
            
            # Filter out tracks that are not between 20 and 500 frames
            valid_tracks = track_counts[(track_counts >= 20) & (track_counts <= 500)].index

            # Process tracks
            for trackID in valid_tracks:
                df_track = df_current_file[df_current_file['TRACK_ID'] == trackID]
                x = df_track["POSITION_X"].to_numpy()
                y = df_track["POSITION_Y"].to_numpy()

                step_size = 3
                for start in range(0, len(x) - window_size + 1, step_size):
                    end = start + window_size
                    window_msd = calc_MSD_NonPhysUnit(df_track.iloc[start:end], np.arange(1, window_size + 1))

                    alpha = calc_alpha(window_msd, np.arange(1, window_size + 1) * s_per_frame)
                    if not np.isnan(alpha) and alpha > 1:
                        alpha_values.append(alpha)
                    
        except Exception as e:
            print(f"Error processing CSV file: {csv_file}")
            print(e)
    return alpha_values

# Select CSV files for each experiment
csv_files_1 = filedialog.askopenfilenames(
    title="Select CSV Files for No drug_2x", filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
)
csv_files_2 = filedialog.askopenfilenames(
    title="Select CSV Files for Nocodazole_30 mins", filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
)
csv_files_3 = filedialog.askopenfilenames(
    title="Select CSV Files for Nocodazole_60 mins", filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
)

# Calculate alpha values for each experiment
alpha_1 = calculate_alpha(csv_files_1)
alpha_2 = calculate_alpha(csv_files_2)
alpha_3 = calculate_alpha(csv_files_3)

# Concatenate the data with labels
label1 = "No drug_2x"
label2 = "Nocodazole_30 mins"
label3 = "Nocodazole_60 mins"
data = pd.concat(
    [
        pd.DataFrame({"alpha": alpha_1, "label": label1}),
        pd.DataFrame({"alpha": alpha_2, "label": label2}),
        pd.DataFrame({"alpha": alpha_3, "label": label3}),
    ],
    ignore_index=True,
)
save_path = r"Z:\Bisal_Halder_turbo\PROCESSED_DATA\Trial_analysis"
csv_file_path = os.path.join(save_path, "alpha_nocodazole_window8_trackcount500_alpha gt 1.5.csv")
data.to_csv(csv_file_path, index=False)


# Set axis limits
plt.figure(figsize=(4, 3))
palette = ["skyblue", "orange", "green"]

ax = sns.histplot(
    data=data,
    x="alpha",
    hue="label",
    palette=palette,
    bins=17,
    binrange=(1.5, 2),
    stat="probability",
    common_norm=False,
    lw=2,
    element="step",
    fill=False,
    cumulative=False,
)

plt.xlim(1.5, 2.0)
sns.move_legend(
    ax,
    loc ="upper right",
    title=None,
    frameon=False,
    fontsize=18,
)
plt.xlabel("Alpha", fontsize=18)
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