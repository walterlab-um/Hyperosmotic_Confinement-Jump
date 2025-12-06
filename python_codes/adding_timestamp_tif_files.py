import os
from os.path import join
import matplotlib.pyplot as plt
from tifffile import imread
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Configuration parameters
s_per_frame = 2
folder_data = r"Z:\Bisal_Halder_turbo\Gateway Exam"
folder_save = folder_data

tif_filename = "20240118_UGD-2x-2s-replicate1-FOV-2.tif"

# Load the data
os.chdir(folder_data)
video = imread(tif_filename)

# Check that the video data is loaded correctly
print(f"Video shape: {video.shape}")
print("Frame 0 sample data (min, max):", video[0].min(), video[0].max())


def animate(frame):
    fig, ax = plt.subplots(1, 1, facecolor="white", edgecolor="white")

    # Display the image
    img = video[frame, :, :]  # Adjusted zero-based indexing
    print(f"Animating frame {frame} with min {img.min()} and max {img.max()}")
    ax.imshow(img, cmap="gray", vmin=img.min(), vmax=img.max())  # Ensure dynamic range

    # Add time stamp
    ax.text(
        img.shape[1] - 100,
        30,
        f"{round((frame + 1) * s_per_frame, 2)} s",  # Frame + 1 to mimic one-based count
        color="white",
        weight="bold",
        size=14,
    )

    # Set viewing area and display configurations
    ax.axis("off")

    # Close figure after rendering
    plt.close(fig)
    return (fig,)


# Create video
ani = FuncAnimation(plt.figure(), animate, frames=len(video), interval=20, repeat=False)
writer = animation.FFMpegWriter(fps=20)
ani.save(join(folder_save, "Video_Timestamped.mp4"), writer=writer, dpi=600)
