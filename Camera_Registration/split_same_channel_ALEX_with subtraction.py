from tifffile import imread, imwrite
import numpy as np
from tkinter import filedialog as fd
from copy import deepcopy
from rich.progress import track

print("Choose the tif files for channel alignment:")
lst_files = list(fd.askopenfilenames())

#########################################
# Apply registration and Crop
for fname in track(lst_files):
    # load the tiff file
    video = imread(fname)

    # even and odd frame indexes
    frames = int(video.shape[0])
    frames_odd = np.arange(0, frames, 2)
    frames_even = np.arange(1, frames + 1, 2)
    frames_even = frames_even[frames_even < frames]
    video_odd = np.delete(deepcopy(video), frames_even, 0)
    video_even = np.delete(deepcopy(video), frames_odd, 0)

    video_odd_modified = video_odd - video_even

    imwrite(
        fname.strip(".tif") + "-odd_modified.tif",
        video_odd_modified,
        imagej=True,
        metadata={"axes": "TYX"},
    )

    imwrite(
        fname.strip(".tif") + "-even.tif",
        video_even,
        imagej=True,
        metadata={"axes": "TYX"},
    )
