import pandas as pd
import numpy as np
from scipy.spatial import distance
from tkinter import filedialog as fd
from rich.progress import track
import os


def calculate_min_distances(lys_df, condensate_df):
    results = []  # Initialize an empty list to store result dictionaries

    # Iterate over each unique frame (time point)
    for frame in track(condensate_df["t"].unique(), description="Processing frames"):
        # Filter Lys and condensate data for the current frame
        lys_frame_data = lys_df[lys_df["frame"] == frame]
        condensate_frame_data = condensate_df[condensate_df["t"] == frame]

        # Iterate over each condensate in the current frame
        for _, cond_row in condensate_frame_data.iterrows():
            cond_id = cond_row["trackID"]  # Get track ID for the condensate
            cond_center_x = cond_row["x"]  # Get x coordinate of condensate center
            cond_center_y = cond_row["y"]  # Get y coordinate of condensate center

            min_distance = float("inf")  # Initialize minimum distance as infinity

            # Loop through Lys contours in the current frame
            for _, lys_row in lys_frame_data.iterrows():
                contour_coords = eval(
                    lys_row["contour_coord"]
                )  # Get contour coordinates

                # Check distance from condensate center to each point in the contour
                for coord in contour_coords:
                    lys_x, lys_y = coord  # Unpack contour coordinates
                    dist = np.sqrt(
                        (cond_center_x - lys_x) ** 2 + (cond_center_y - lys_y) ** 2
                    )  # Calculate Euclidean distance
                    if dist < min_distance:  # Update minimum distance if applicable
                        min_distance = dist

            # Record results for this condensate
            results.append(
                {
                    "frame": frame,
                    "trackID": cond_id,
                    "min_distance_to_Lys": min_distance,
                }
            )

    return pd.DataFrame(results)  # Return the results as a DataFrame


# Use file dialogs to select CSV files for Lys and Condensates
print("Select CSV file for Lys contours:")
lys_csv_path = fd.askopenfilename(
    title="Select CSV Files for Lys Segmentation", filetypes=[("CSV files", "*.csv")]
)
# Extract the base name of the Lys file to find its corresponding condensate file
lys_file_name = os.path.basename(lys_csv_path)
lys_file_base_name = os.path.splitext(lys_file_name)[0]  # Remove file extension

# Define the folder path where condensate files are stored
condensate_folder_path = r"Z:\Bisal_Halder_turbo\PROCESSED_DATA\Impact_of_cytoskeleton_on_HOPS_condensates\HOPS_Lys_dual imaging\colocalization_calculation\condensates"

# Construct full path to corresponding condensate file by matching names
condensate_csv_path = os.path.join(condensate_folder_path, lys_file_name)

# Check if the corresponding condensate file exists
if not os.path.exists(condensate_csv_path):
    raise FileNotFoundError(
        f"Condensate file {lys_file_name} not found in {condensate_folder_path}"
    )

# Load CSV files into DataFrames
lys_df = pd.read_csv(lys_csv_path)
condensate_df = pd.read_csv(condensate_csv_path)

# Calculate minimum distances and save results in the same directory as Lys file
results_df = calculate_min_distances(lys_df, condensate_df)
output_csv_name = f"{lys_file_base_name}_distance_to_lys.csv"
output_csv_path = os.path.join(os.path.dirname(lys_csv_path), output_csv_name)
results_df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
