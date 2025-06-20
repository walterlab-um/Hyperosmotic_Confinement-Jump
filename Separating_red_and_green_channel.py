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
for fpath in track(lst_files):
    # load the tiff file
    video = imread(fpath)
    print(f"File: {fpath}, Shape: {video.shape}")
    halfwidth = int(video.shape[2] / 2)
    # split left and right
    # video_left = video[:, :, 0:halfwidth]
    # video_right = video[:, :, halfwidth:]
    fsave = fpath[:-4] + "-cropped.tif"
    if selector == "1":
        imwrite(
            fsave,
            video[:, :, 0:halfwidth],
            imagej=True,
            metadata={"axes": "TYX"},
        )
    elif selector == "2":
        imwrite(
            fsave,
            video[:, :, halfwidth:],
            imagej=True,
            metadata={"axes": "TYX"},
        )
