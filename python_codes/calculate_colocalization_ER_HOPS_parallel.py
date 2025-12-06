import os
from os.path import join
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from rich.progress import track
from concurrent.futures import ProcessPoolExecutor, as_completed

pd.options.mode.chained_assignment = None  # default='warn'

# Scaling factors for physical units
nm_per_pixel = 117  # Set pixel size directly
print("Scaling factors: nm_per_pixel = " + str(nm_per_pixel))

# Set folder path directly
folder_path = "/home/bisal/Desktop/Bisal_Halder_turbo/PROCESSED_DATA/Impact_of_cytoskeleton_on_HOPS_condensates/HOPS_ER_dual imaging/no_drug/calculate_colocalization/"
os.chdir(folder_path)

# Parameters
interaction_cutoff = 10  # pixels

# Output file columns
columns = [
    "fname_condensate",
    "fname_ER",
    "condensateID",
    "t",
    "x",
    "y",
    "InER",
    "ER_ID",
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


def fetch_nearby_ER(df_ER, t, mean_condensate_x, mean_condensate_y, interaction_cutoff):
    df_ER_current_t = df_ER[df_ER["frame"] == t]
    df_ER_current_t["distance_squared"] = (
        df_ER_current_t["center_x_pxl"] - mean_condensate_x
    ) ** 2 + (df_ER_current_t["center_y_pxl"] - mean_condensate_y) ** 2
    df_ER_nearby = df_ER_current_t[
        df_ER_current_t["distance_squared"] <= interaction_cutoff**2
    ]

    dict_ER_polygons_nearby = {}
    for _, row in df_ER_nearby.iterrows():
        ER_ID_nearby = row["condensateID"]
        str_ER_coords = row["contour_coord"]
        ER_polygon = list_like_string_to_polygon(str_ER_coords)
        dict_ER_polygons_nearby[ER_ID_nearby] = ER_polygon

    return dict_ER_polygons_nearby


def process_FOV(exp):
    # Read files
    ER_file = exp + ".csv"
    condensate_file = exp + ".csv"

    df_ER = pd.read_csv(join(ER_dir, ER_file))
    df_condensate = pd.read_csv(join(condensate_dir, condensate_file))

    # Process condensate tracks one by one
    lst_rows_of_df = []
    for condensateID in df_condensate["trackID"]:
        current_condensate = df_condensate[df_condensate["trackID"] == condensateID]
        lst_x = current_condensate["x"].values
        lst_y = current_condensate["y"].values
        lst_t = current_condensate["t"].values
        radius = current_condensate["R"].values
        mean_condensate_x = np.mean(lst_x)
        mean_condensate_y = np.mean(lst_y)

        # Process each position in track one by one
        for i in range(len(lst_t)):
            t = lst_t[i]
            x = lst_x[i]
            y = lst_y[i]

            polygon_condensate = Point(x, y).buffer(radius[i])

            # Perform colocalization
            dict_ER_polygons_nearby = fetch_nearby_ER(
                df_ER, t, mean_condensate_x, mean_condensate_y, interaction_cutoff
            )

            InER = False
            for key, polygon in dict_ER_polygons_nearby.items():
                if polygon.contains(polygon_condensate):
                    InER = True
                    ER_ID = key
                    break
            if not InER:
                ER_ID = np.nan

            # Calculate distances
            if InER:
                distance_to_edge_nm = (
                    polygon.exterior.distance(polygon_condensate.centroid)
                    * nm_per_pixel
                )
                distance_to_center_nm = (
                    polygon.centroid.distance(polygon_condensate.centroid)
                    * nm_per_pixel
                )
            else:
                distance_to_edge_nm = np.nan
                distance_to_center_nm = np.nan

            # Save
            new_row = [
                condensate_file,
                ER_file,
                condensateID,
                t,
                x,
                y,
                InER,
                ER_ID,
                radius[i] * nm_per_pixel,
                distance_to_center_nm,
                distance_to_edge_nm,
            ]
            lst_rows_of_df.append(new_row)

    df_save = pd.DataFrame.from_records(lst_rows_of_df, columns=columns)
    fname_save = join(folder_path, "colocalization_AIO-" + exp + ".csv")
    df_save.to_csv(fname_save, index=False)


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

# Parallelize the processing using ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_FOV, exp) for exp in experiment_names]
    for future in track(
        as_completed(futures), total=len(futures), description="Processing FOVs"
    ):
        future.result()  # wait for each future to complete

print("All FOVs have been processed.")
