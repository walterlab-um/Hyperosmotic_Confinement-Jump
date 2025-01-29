import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from tkinter import filedialog as fd
from rich.progress import track
import os


def calculate_min_distances(er_df, condensate_df):
    results = []  # Initialize an empty list to store result dictionaries

    # Iterate over each unique frame (time point)
    for frame in track(condensate_df["t"].unique(), description="Processing frames"):
        # Filter ER and condensate data for the current frame
        er_frame_data = er_df[er_df["frame"] == frame]
        condensate_frame_data = condensate_df[condensate_df["t"] == frame]

        # Iterate over each condensate in the current frame
        for _, cond_row in condensate_frame_data.iterrows():
            cond_id = cond_row["trackID"]  # Get track ID for the condensate
            cond_center_x = cond_row["x"]  # Get x coordinate of condensate center
            cond_center_y = cond_row["y"]  # Get y coordinate of condensate center

            min_distance = float("inf")  # Initialize minimum distance as infinity
            is_in_er = False  # Initialize in-ER status as False
            point = Point(
                cond_center_x, cond_center_y
            )  # Create a point for the condensate center

            # Loop through ER contours in the current frame
            for _, er_row in er_frame_data.iterrows():
                contour_coords = eval(
                    er_row["contour_coord"]
                )  # Get contour coordinates
                er_polygon = Polygon(
                    contour_coords
                )  # Create a polygon for the ER boundary

                # Check if the condensate center is within the ER boundary
                if er_polygon.contains(point):
                    is_in_er = True
                    min_distance = np.nan
                    break  # No need to check further if inside an ER

                # Calculate the minimum distance to the contour if outside
                else:
                    for coord in contour_coords:
                        er_x, er_y = coord  # Unpack contour coordinates
                        dist = np.sqrt(
                            (cond_center_x - er_x) ** 2 + (cond_center_y - er_y) ** 2
                        )  # Calculate Euclidean distance
                        if dist < min_distance:
                            min_distance = dist

            # Record results for this condensate
            results.append(
                {
                    "frame": frame,
                    "trackID": cond_id,
                    "In_ER": is_in_er,
                    "min_distance_to_ER": min_distance,
                }
            )

    return pd.DataFrame(results)  # Return the results as a DataFrame


# Use file dialogs to select CSV files for ER and Condensates
print("Select CSV file for ER contours:")
er_csv_path = fd.askopenfilename(
    title="Select CSV Files for ER Segmentation", filetypes=[("CSV files", "*.csv")]
)
# Extract the base name of the ER file to find its corresponding condensate file
er_file_name = os.path.basename(er_csv_path)
er_file_base_name = os.path.splitext(er_file_name)[0]  # Remove file extension

# Define the folder path where condensate files are stored
condensate_folder_path = r"Z:\Bisal_Halder_turbo\PROCESSED_DATA\Impact_of_cytoskeleton_on_HOPS_condensates\HOPS_ER_dual imaging\no_drug\Analysed data\Minimum distance HOPS_ER\refomatted"

# Construct full path to corresponding condensate file by matching names
condensate_csv_path = os.path.join(condensate_folder_path, er_file_name)

# Check if the corresponding condensate file exists
if not os.path.exists(condensate_csv_path):
    raise FileNotFoundError(
        f"Condensate file {er_file_name} not found in {condensate_folder_path}"
    )

# Load CSV files into DataFrames
er_df = pd.read_csv(er_csv_path)
condensate_df = pd.read_csv(condensate_csv_path)

# Calculate minimum distances and save results in the same directory as ER file
results_df = calculate_min_distances(er_df, condensate_df)
output_csv_name = f"{er_file_base_name}_distance_to_er_v2.csv"
output_csv_path = os.path.join(os.path.dirname(er_csv_path), output_csv_name)
results_df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
