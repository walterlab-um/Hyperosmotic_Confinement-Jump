from tifffile import imread, imwrite
from tkinter import filedialog as fd
from rich.progress import track

#########################################
# Load and organize files
print("Type 1 to keep left, 2 to keep right channel")
selector = input()
if (selector != "1") & (selector != "2"):
    print("Please type only 1 or 2")
    exit()
print("Choose the tif files for crop")
lst_files = list(fd.askopenfilenames())
#########################################
# Apply registration and Crop

# Previous version
# for fpath in track(lst_files):
#     # load the tiff file
#     video = imread(fpath)
#     print(f"File: {fpath}, Shape: {video.shape}")
#     halfwidth = int(video.shape[2] / 2)


for fpath in track(lst_files):
    # load the tiff file
    img = imread(fpath)
    print(f"File: {fpath}, Shape: {img.shape}")

    # Check if it's a video (3D) or image (2D)
    if len(img.shape) == 3:  # Video case (T, Y, X)
        halfwidth = int(img.shape[2] / 2)
        if selector == "1":
            output = img[:, :, 0:halfwidth]
        else:
            output = img[:, :, halfwidth:]
    elif len(img.shape) == 2:  # Image case (Y, X)
        halfwidth = int(img.shape[1] / 2)
        if selector == "1":
            output = img[:, 0:halfwidth]
        else:
            output = img[:, halfwidth:]
    else:
        print(f"Unsupported image shape: {img.shape}")
        continue

    # Previous version
    # fsave = fpath[:-4] + "-cropped.tif"
    # if selector == "1":
    #     imwrite(
    #         fsave,
    #         video[:, :, 0:halfwidth],
    #         imagej=True,
    #         metadata={"axes": "TYX"},
    #     )
    # elif selector == "2":
    #     imwrite(
    #         fsave,
    #         video[:, :, halfwidth:],
    #         imagej=True,
    #         metadata={"axes": "TYX"},
    #     )

    fsave = fpath[:-4] + "-cropped.tif"
    imwrite(
        fsave,
        output,
        imagej=True,
        metadata={"axes": "TYX"} if len(img.shape) == 3 else None,
    )
