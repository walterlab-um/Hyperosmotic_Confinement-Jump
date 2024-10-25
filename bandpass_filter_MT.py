from tifffile import imread, imwrite
from scipy.ndimage import white_tophat
from skimage.filters import difference_of_gaussians
from skimage.exposure import rescale_intensity
from skimage.util import img_as_uint
from tkinter import filedialog as fd
import numpy as np
from rich.progress import track

print("Choose all tif files to be batch proccessed:")
lst_files = list(fd.askopenfilenames())

DoG_sigma_low = 1
DoG_sigma_high = 2.5

###############################
for f in track(lst_files):
    video = imread(f)
    video_out = []
    for img in video:
        img_filtered = difference_of_gaussians(img, DoG_sigma_low, DoG_sigma_high)
        img_rescaled = rescale_intensity(img_filtered, out_range=(-1, 1))
        video_out.append(img_rescaled)
    video_out = np.stack(video_out)
    video_out_0_1 = rescale_intensity(video_out, in_range=(-1, 1), out_range=(0, 1))
    fsave = f[:-4] + "-bandpass.tif"
    if len(video.shape) < 3:  # image:
        imwrite(fsave, img_as_uint(video_out_0_1), imagej=True)
    elif len(video.shape) > 2:  # video:
        imwrite(
            fsave, img_as_uint(video_out_0_1), imagej=True, metadata={"axes": "TYX"}
        )
