import os
from os.path import join
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from rich.progress import track
from tkinter import filedialog as fd
import tkinter as tk

pd.options.mode.chained_assignment = None  # default='warn'

# AIO: All in one format
# This script gives the collocalization information for every lysosome position over time.

# Scaling factors for physical units
print("Type in the pixel size in nm:")
nm_per_pixel = float(input())
print("Scaling factors: nm_per_pixel = " + str(nm_per_pixel))

print(
    "Please choose a folder containing subfolders: cell_body_manual, condensates, lysosomes:"
)
root = tk.Tk()
root.withdraw()  # Hide the main window
folder_path = fd.askdirectory()
os.chdir(folder_path)

# Parameters
interaction_cutoff = 10  # pixels

# Output file columns
columns = [
    "fname_lysosome",
    "fname_condensate",
    "lysosomeID",
    "t",
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


def list_like_string_to_xyt(list_like_string):
    list_of_xyt_string = list_like_string[1:-1].split(", ")
    return [float(xyt_string) for xyt_string in list_of_xyt_string]


def find_common(condensate_files, lysosome_files):
    experiment_names1 = [file.rstrip(".csv") for file in condensate_files]
    experiment_names2 = [file.rstrip(".csv") for file in lysosome_files]
    return list(set(experiment_names1) & set(experiment_names2))


def fetch_nearby_condensates(
    df_condensate, t, mean_lysosome_x, mean_lysosome_y, interaction_cutoff
):
    # Load condensates near the lysosome as dictionary of polygons
    df_condensate_current_t = df_condensate[df_condensate["t"] == t]

    # Calculate the squared distance between condensate centers and mean lysosome coordinates
    df_condensate_current_t["distance_squared"] = (
        df_condensate_current_t["x"] - mean_lysosome_x
    ) ** 2 + (df_condensate_current_t["y"] - mean_lysosome_y) ** 2

    # Filter condensates within the interaction cutoff
    df_condensate_nearby = df_condensate_current_t[
        df_condensate_current_t["distance_squared"] <= interaction_cutoff**2
    ]

    # Create a dictionary to store nearby condensate polygons
    dict_condensate_polygons_nearby = {}

    # Iterate over nearby condensates and create polygons
    for _, row in df_condensate_nearby.iterrows():
        condensateID_nearby = row["trackID"]
        x, y, radius = row["x"], row["y"], row["R"]
        polygon = Point(x, y).buffer(radius)
        dict_condensate_polygons_nearby[condensateID_nearby] = polygon

    return dict_condensate_polygons_nearby


# Define the data directories
condensate_dir = join(folder_path, "condensates")
lysosome_dir = join(folder_path, "lysosomes")
cell_boundaries_dir = join(folder_path, "cell_body_manual")

# List files in the directories
condensate_files = [
    file for file in os.listdir(condensate_dir) if file.endswith(".csv")
]
lysosome_files = [file for file in os.listdir(lysosome_dir) if file.endswith(".csv")]

# Find common experiment names
experiment_names = find_common(condensate_files, lysosome_files)

# Loop through each Field Of View (FOV)
for exp in track(experiment_names):
    lysosome_file = exp + ".csv"
    condensate_file = exp + ".csv"
    cell_roi_files = [
        file
        for file in os.listdir(cell_boundaries_dir)
        if file.startswith(exp) and file.endswith(".txt")
    ]

    df_lysosome = pd.read_csv(join(lysosome_dir, lysosome_file))
    df_condensate = pd.read_csv(join(condensate_dir, condensate_file))

    # Process lysosome tracks one by one
    lst_rows_of_df = []
    for lysosomeID in df_lysosome["condensateID"]:
        current_lysosome = df_lysosome[df_lysosome["condensateID"] == lysosomeID]
        lst_x = current_lysosome["center_x_pxl"].values
        lst_y = current_lysosome["center_y_pxl"].values
        lst_t = current_lysosome["frame"].values
        mean_lysosome_x = np.mean(lst_x)
        mean_lysosome_y = np.mean(lst_y)

        # Process each position in track one by one
        for i in range(len(lst_t)):
            t = lst_t[i]
            x = lst_x[i]
            y = lst_y[i]

            point_lysosome = Point(x, y)

            ## Perform colocalization
            # Fetch nearby condensates
            dict_condensate_polygons_nearby = fetch_nearby_condensates(
                df_condensate, t, mean_lysosome_x, mean_lysosome_y, interaction_cutoff
            )
            # Search for which condensate it's in
            InCondensate = False
            for key, polygon in dict_condensate_polygons_nearby.items():
                if point_lysosome.within(polygon):
                    InCondensate = True
                    condensateID = key
                    R_nm = np.sqrt(polygon.area * nm_per_pixel**2 / np.pi)
                    distance_to_edge_nm = (
                        polygon.exterior.distance(point_lysosome) * nm_per_pixel
                    )
                    distance_to_center_nm = (
                        polygon.centroid.distance(point_lysosome) * nm_per_pixel
                    )
                    break
            if not InCondensate:
                condensateID = np.nan
                R_nm = np.nan
                distance_to_center_nm = np.nan
                distance_to_edge_nm = np.nan

            # Save
            new_row = [
                lysosome_file,
                condensate_file,
                lysosomeID,
                t,
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
