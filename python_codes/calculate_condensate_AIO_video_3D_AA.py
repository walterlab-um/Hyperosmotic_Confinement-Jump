from tifffile import imread, imsave
import os
from os.path import join, dirname, basename
import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from tkinter import filedialog as fd

pd.options.mode.chained_assignment = None  # default='warn'

# Set your microscope calibration here (microns per pixel)
um_per_pixel = 0.117

print("Choose Simple Segmentation.h5 files for processing:")
lst_fpath = list(fd.askopenfilenames())
folder_save = dirname(lst_fpath[0])
os.chdir(folder_save)

columns = [
    "condensateID",
    "contour_coord",
    "center_x_pxl",
    "center_y_pxl",
    "area_um2",
    "R_nm",
    "mean_intensity",
    "max_intensity",
    "max_location",
    "aspect_ratio",
    "contour_solidity",
    "contour_extent",
]

switch_plot = True  # turn to False to skip contour figure
plow = 0.05  # imshow intensity percentile
phigh = 99


# ---------- Functions ----------
def pltcontours(img, contours, fsave):
    plt.figure(dpi=600)
    vmin, vmax = np.percentile(img, (plow, phigh))
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    for cnt in contours:
        x = cnt[:, 0][:, 0]
        y = cnt[:, 0][:, 1]
        plt.plot(x, y, "-", color="firebrick", linewidth=0.2)
        xlast = [x[-1], x[0]]
        ylast = [y[-1], y[0]]
        plt.plot(xlast, ylast, "-", color="firebrick", linewidth=0.2)
    plt.xlim(0, img.shape[1])
    plt.ylim(0, img.shape[0])
    plt.tight_layout()
    plt.axis("scaled")
    plt.axis("off")
    plt.savefig(fsave, format="png", bbox_inches="tight", dpi=600)
    plt.close()


def cnt_fill(imgshape, cnt):
    mask = np.zeros(imgshape, dtype=np.uint8)
    cv2.fillPoly(mask, [cnt], (255))
    return mask


def cnt_to_list(cnt):
    cnt_2d = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
    lst_cnt = [cnt_2d[i, :].tolist() for i in range(cnt_2d.shape[0])]
    return lst_cnt


# ---------- Main Processing ----------
for fpath in lst_fpath:
    print(f"Processing segmentation: {fpath}")

    # Load ilastik output (assumed single-channel)
    with h5py.File(fpath, "r") as h5f:
        ilastik_output = h5f["exported_data"][:, :, 0]  # For 2D image

    # Find corresponding .tif file
    tif_path = fpath[:-23] + ".tif"
    if not os.path.isfile(tif_path):
        raise FileNotFoundError(f"Could not find TIFF file: {tif_path}")

    print(f"Reading image from: {tif_path}")
    img = imread(tif_path)
    print(f"Loaded file shape: {img.shape}")

    # Handle RGB or multichannel TIFFs
    if img.ndim == 2:
        img_2d = img
        print("Image is 2D. Shape:", img_2d.shape)
    elif img.ndim == 3:
        # Assume last axis is channel; pick the first (red) channel
        img_2d = img[:, :, 0]
        print("Image is 3D. Shape (Y,X,C). Using channel 0; shape:", img_2d.shape)
    else:
        raise ValueError(f"Expected 2D or 3D image, got {img.ndim}D data")

    # Now use img_2d for processing
    mask_all_condensates = 2 - ilastik_output  # 1==condensate, 0==background
    lst_rows_of_df = []

    if mask_all_condensates.sum() > 0:
        contours, _ = cv2.findContours(
            mask_all_condensates.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        condensateID = 1
        for cnt in contours:
            if cv2.contourArea(cnt) > 0.8 * img_2d.shape[0] * img_2d.shape[1]:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            center_x_pxl = M["m10"] / M["m00"]
            center_y_pxl = M["m01"] / M["m00"]
            area_um2 = cv2.contourArea(cnt) * um_per_pixel**2
            R_nm = np.sqrt(area_um2 / np.pi) * 1000

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            rect_area = w * h
            contour_extent = float(cv2.contourArea(cnt)) / rect_area

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            contour_solidity = float(cv2.contourArea(cnt)) / hull_area

            mask_current_condensate = cnt_fill(img_2d.shape, cnt)
            # Ensure mask is np.uint8 and image is 2D
            mean_intensity = cv2.mean(img_2d, mask=mask_current_condensate)[0]
            _, max_intensity, _, max_location = cv2.minMaxLoc(
                img_2d, mask=mask_current_condensate
            )

            new_row = [
                condensateID,
                cnt_to_list(cnt),
                center_x_pxl,
                center_y_pxl,
                area_um2,
                R_nm,
                mean_intensity,
                max_intensity,
                max_location,
                aspect_ratio,
                contour_solidity,
                contour_extent,
            ]
            lst_rows_of_df.append(new_row)
            condensateID += 1

        if switch_plot and len(contours) > 0:
            pltcontours(img_2d, contours, fpath[:-4] + ".png")

        output_file_path = join(
            dirname(fpath), basename(fpath)[:-24] + "_segmentation.tif"
        )
        imsave(output_file_path, mask_all_condensates.astype(np.uint8), imagej=True)
        print(f"Saved segmentation mask to {output_file_path}")

    # Save to CSV
    df_save = pd.DataFrame.from_records(lst_rows_of_df, columns=columns)
    fname_save = join(
        dirname(fpath), "condensates_AIO-" + basename(fpath)[:-23] + ".csv"
    )
    df_save.to_csv(fname_save, index=False)
    print(f"Saved {len(lst_rows_of_df)} condensates to {fname_save}")
