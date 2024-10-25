import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import os
from tkinter import filedialog


def load_cell_polygons(cell_boundary_folder, fov_name):
    cell_polygons = []
    for cell_file in os.listdir(cell_boundary_folder):
        if cell_file.startswith(f"{fov_name}_cell"):
            cell_num = cell_file.split("_cell")[1].replace(".txt", "")
            cell_outline_coordinates = pd.read_csv(
                os.path.join(cell_boundary_folder, cell_file), sep="\t", header=None
            )
            coords_roi = [tuple(row) for _, row in cell_outline_coordinates.iterrows()]
            cell_polygon = Polygon(coords_roi)
            cell_polygons.append((cell_polygon, cell_num))

    return cell_polygons


def assign_cellID_to_tracks(df, cell_polygons, trackID):
    df["x_mean"] = df.groupby(trackID)["x"].transform("mean")
    df["y_mean"] = df.groupby(trackID)["y"].transform("mean")

    track_to_cellID = {}

    for name, group in df.groupby(trackID):
        point = Point(group.iloc[0]["x_mean"], group.iloc[0]["y_mean"])
        cellID = "cell1"

        for polygon, cell_num in cell_polygons:
            if point.within(polygon):
                cellID = f"cell{cell_num}"
                break

        track_to_cellID[name] = cellID

    df["cellID"] = df[trackID].map(track_to_cellID)
    return df


csv_file_path = filedialog.askopenfilename(
    title="Select the track CSV File", filetypes=[("CSV files", "*.csv")]
)


cell_boundary_folder = r"Z:\Bisal_Halder_turbo\PROCESSED_DATA\Impact_of_cytoskeleton_on_HOPS_condensates\no_drug\Analysed Data\1x\cell boundary coordinates"


if csv_file_path and cell_boundary_folder:
    fov_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    cell_polygons = load_cell_polygons(cell_boundary_folder, fov_name)

    track_df = pd.read_csv(csv_file_path)

    track_df_with_cellID = assign_cellID_to_tracks(track_df, cell_polygons, "trackID")

    output_file_path = csv_file_path.replace(".csv", "_with_cellIDs.csv")
    track_df_with_cellID.to_csv(output_file_path, index=False)
    print(f"Updated track data with cell IDs has been saved to: {output_file_path}")
