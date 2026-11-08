import os
import sys
from tkinter import filedialog as fd
from rich.progress import track
from tifffile import TiffFile, TiffWriter

print("Type 1 to keep left, 2 to keep right channel")
selector = input().strip()
if selector not in ["1", "2"]:
    print("Please type only 1 or 2")
    sys.exit(1)

print("Choose the tif files for crop...")
lst_files = list(fd.askopenfilenames(filetypes=[("TIFF files", "*.tif *.tiff")]))

if not lst_files:
    print("No files selected.")
    sys.exit(0)

for fpath in track(lst_files):
    print(f"\nProcessing: {os.path.basename(fpath)}")

    try:
        with TiffFile(fpath) as tif:
            total_frames = len(tif.pages)
            first_frame = tif.pages[0].asarray()

            # Inspect dimensions from the first page
            if first_frame.ndim == 2:
                height, width = first_frame.shape
            elif first_frame.ndim == 3:
                height, width = first_frame.shape[1], first_frame.shape[2]
            else:
                print(f"  ✗ Unsupported frame shape: {first_frame.shape}")
                continue

            halfwidth = width // 2

            if selector == "1":
                crop_slice = slice(0, halfwidth)
                channel_name = "left"
            else:
                crop_slice = slice(halfwidth, None)
                channel_name = "right"

            base_name = os.path.splitext(fpath)[0]
            fsave = f"{base_name}-{channel_name}-channel.tif"
            print(f"  Frames: {total_frames} | Dimensions: {height}x{width}")
            print(f"  Output file: {fsave}")

            # Stream frame-by-frame directly to disk
            with TiffWriter(fsave, bigtiff=True) as tif_writer:
                for idx, page in enumerate(tif.pages):
                    frame = page.asarray()

                    # Apply crop based on array dimensions
                    if frame.ndim == 2:
                        cropped = frame[:, crop_slice]
                    else:
                        cropped = frame[:, :, crop_slice]

                    # Write single frame to file
                    if idx == 0:
                        tif_writer.write(
                            cropped,
                            contiguous=False,
                            metadata={"axes": "TYX" if total_frames > 1 else "YX"},
                            compression="zlib",
                            compressionargs={"level": 1},
                        )
                    else:
                        tif_writer.write(
                            cropped,
                            compression="zlib",
                            compressionargs={"level": 1},
                        )

                    # Update progress every 50 frames
                    if (idx + 1) % 50 == 0 or (idx + 1) == total_frames:
                        progress = ((idx + 1) / total_frames) * 100
                        print(
                            f"    Progress: {progress:.1f}% ({idx + 1}/{total_frames} frames)",
                            end="\r",
                        )

            print(f"\n  ✓ Successfully saved: {os.path.basename(fsave)}")

    except Exception as e:
        print(f"  ✗ Error processing {os.path.basename(fpath)}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\nProcessing complete!")