import os
from os.path import join
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from shapely.geometry import box
from rich.progress import track
from tkinter import filedialog as fd
import tkinter as tk

pd.options.mode.chained_assignment = None  # default='warn'

# Scaling factors for physical units
print("Type in the pixel size in nm:")
nm_per_pixel = float(input())
print("Scaling factors: nm_per_pixel = " + str(nm_per_pixel))

# Folder selection
print(
    "Please choose a folder containing subfolders: cell_body_manual, condensates, ER:"
)
root = tk.Tk()
root.withdraw()  # Hide the main window
folder_path = fd.askdirectory()
os.chdir(folder_path)

# Parameters
interaction_cutoff = 10  # pixels

# Output file columns
columns = [
    "fname_ER",
    "fname_condensate",
    "frame",
    "x",
    "y",
    "InCondensate",
    "condensateID",
    "R_nm",
    "distance_to_center_nm",
    "distance_to_edge_nm",
]


def list_like_string_to_polygon(list_like_string):
    list_of_xy_string = list_like_string[2:-2].split("], [")
    coords_roi = [
        (int(x), int(y))
        for x, y in (xy_string.split(", ") for xy_string in list_of_xy_string)
    ]
    return Polygon(coords_roi)


def find_common(condensate_files, ER_files):
    experiment_names1 = [file.rstrip(".csv") for file in condensate_files]
    experiment_names2 = [file.rstrip(".csv") for file in ER_files]
    return list(set(experiment_names1) & set(experiment_names2))


def fetch_nearby_condensates(
    df_condensate, t, mean_ER_x, mean_ER_y, interaction_cutoff
):
    df_condensate_current_t = df_condensate[df_condensate["t"] == t]
    df_condensate_current_t["distance_squared"] = (
        df_condensate_current_t["x"] - mean_ER_x
    ) ** 2 + (df_condensate_current_t["y"] - mean_ER_y) ** 2
    df_condensate_nearby = df_condensate_current_t[
        df_condensate_current_t["distance_squared"] <= interaction_cutoff**2
    ]

    dict_condensate_polygons_nearby = {}
    for _, row in df_condensate_nearby.iterrows():
        condensateID_nearby = row["trackID"]
        x, y, radius = row["x"], row["y"], row["R"]
        polygon = Point(x, y).buffer(radius)
        dict_condensate_polygons_nearby[condensateID_nearby] = polygon

    return dict_condensate_polygons_nearby


# Define the data directories
condensate_dir = join(folder_path, "condensates")
ER_dir = join(folder_path, "ER")
cell_boundaries_dir = join(folder_path, "cell_body_manual")

# List files in the directories
condensate_files = [
    file for file in os.listdir(condensate_dir) if file.endswith(".csv")
]
ER_files = [file for file in os.listdir(ER_dir) if file.endswith(".csv")]

# Find common experiment names
experiment_names = find_common(condensate_files, ER_files)

# Loop through each Field Of View (FOV)
for exp in track(experiment_names):
    ER_file = exp + ".csv"
    condensate_file = exp + ".csv"
    cell_roi_files = [
        file
        for file in os.listdir(cell_boundaries_dir)
        if file.startswith(exp) and file.endswith(".txt")
    ]

    df_ER = pd.read_csv(join(ER_dir, ER_file))
    df_condensate = pd.read_csv(join(condensate_dir, condensate_file))

    # Process ER tracks one by one
    lst_rows_of_df = []
    for frame_num in df_ER["frame"].unique():
        current_frame = df_ER[df_ER["frame"] == frame_num]
        mean_ER_x = current_frame["center_x_pxl"].mean()
        mean_ER_y = current_frame["center_y_pxl"].mean()

        for i, row in current_frame.iterrows():
            x = row["center_x_pxl"]
            y = row["center_y_pxl"]

            point_ER = Point(x, y)

            ## Perform colocalization
            # Fetch nearby condensates
            dict_condensate_polygons_nearby = fetch_nearby_condensates(
                df_condensate, frame_num, mean_ER_x, mean_ER_y, interaction_cutoff
            )
            # Search for which condensate it's in
            InCondensate = False
            for key, polygon in dict_condensate_polygons_nearby.items():
                if point_ER.within(polygon):
                    InCondensate = True
                    condensateID = key
                    R_nm = np.sqrt(polygon.area * nm_per_pixel**2 / np.pi)
                    distance_to_edge_nm = (
                        polygon.exterior.distance(point_ER) * nm_per_pixel
                    )
                    distance_to_center_nm = (
                        polygon.centroid.distance(point_ER) * nm_per_pixel
                    )
                    break
            if not InCondensate:
                condensateID = np.nan
                R_nm = np.nan
                distance_to_center_nm = np.nan
                distance_to_edge_nm = np.nan

            # Save
            new_row = [
                ER_file,
                condensate_file,
                frame_num,
                x,
                y,
                InCondensate,
                condensateID,
                R_nm,
                distance_to_center_nm,
                distance_to_edge_nm,
            ]
            lst_rows_of_df.append(new_row)

    df_save = pd.DataFrame.from_records(lst_rows_of_df, columns=columns)
    fname_save = join(folder_path, "colocalization_AIO-" + exp + ".csv")
    df_save.to_csv(fname_save, index=False)
