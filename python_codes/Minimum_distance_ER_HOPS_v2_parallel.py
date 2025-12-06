import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from concurrent.futures import ProcessPoolExecutor
import os
from tkinter import filedialog as fd
from rich.progress import track


def calculate_min_distance_for_frame(frame_data):
    """
    Calculate minimum distances for given frame data and determine if condensates are inside ER.
    This function is meant to be parallelized.
    """
    frame, er_frame_data, condensate_frame_data = frame_data
    results = []

    # Iterate over each condensate in the current frame
    for _, cond_row in condensate_frame_data.iterrows():
        cond_id = cond_row["trackID"]
        cond_center_x = cond_row["x"]
        cond_center_y = cond_row["y"]

        min_distance = float("inf")
        is_in_er = False
        point = Point(cond_center_x, cond_center_y)

        for _, er_row in er_frame_data.iterrows():
            contour_coords = eval(er_row["contour_coord"])
            er_polygon = Polygon(contour_coords)

            if er_polygon.contains(point):
                is_in_er = True
                min_distance = np.nan
                break  # No need to check further if inside an ER

            for coord in contour_coords:
                er_x, er_y = coord
                dist = np.sqrt(
                    (cond_center_x - er_x) ** 2 + (cond_center_y - er_y) ** 2
                )
                if dist < min_distance:
                    min_distance = dist

        results.append(
            {
                "frame": frame,
                "trackID": cond_id,
                "In_ER": is_in_er,
                "min_distance_to_ER": min_distance,
            }
        )

    return results


def calculate_min_distances(er_df, condensate_df):
    unique_frames = condensate_df["t"].unique()
    frame_data = [
        (
            frame,
            er_df[er_df["frame"] == frame],
            condensate_df[condensate_df["t"] == frame],
        )
        for frame in unique_frames
    ]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(
            track(
                executor.map(calculate_min_distance_for_frame, frame_data),
                description="Calculating distances...",
            )
        )

    # Flatten list of results
    flat_results = [result for sublist in results for result in sublist]
    return pd.DataFrame(flat_results)


# Use file dialogs to select CSV files for ER and Condensates
print("Select CSV file for ER contours:")
er_csv_path = fd.askopenfilename(
    title="Select CSV Files for ER Segmentation", filetypes=[("CSV files", "*.csv")]
)
er_file_name = os.path.basename(er_csv_path)
er_file_base_name = os.path.splitext(er_file_name)[0]

# Updated the condensate folder path
condensate_folder_path = r"/home/bisal/Desktop/Bisal_Halder_turbo/PROCESSED_DATA/Impact_of_cytoskeleton_on_HOPS_condensates/HOPS_ER_dual imaging/no_drug/Analysed data/Minimum distance HOPS_ER/refomatted/"
condensate_csv_path = os.path.join(condensate_folder_path, er_file_name)

if not os.path.exists(condensate_csv_path):
    raise FileNotFoundError(
        f"Condensate file {er_file_name} not found in {condensate_folder_path}"
    )

er_df = pd.read_csv(er_csv_path)
condensate_df = pd.read_csv(condensate_csv_path)

results_df = calculate_min_distances(er_df, condensate_df)
output_csv_name = f"{er_file_base_name}_distance_to_er_v2.csv"
output_csv_path = os.path.join(os.path.dirname(er_csv_path), output_csv_name)
results_df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
