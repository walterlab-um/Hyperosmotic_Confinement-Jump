import pandas as pd
from tkinter import filedialog as fd
import os

# Ask user to select the combined_results.csv file
print("Select your combined_results.csv file")
input_path = fd.askopenfilename(filetypes=(("CSV files", "*.csv"),))
if not input_path:
    print("No file selected. Exiting...")
    exit()

# Set working directory to file location
os.chdir(os.path.dirname(input_path))

# Read the combined results CSV
df = pd.read_csv(input_path)

# Calculate number of rows (punctas/condensates) per source_file
puncta_counts = df["source_file"].value_counts().sort_index().reset_index()
puncta_counts.columns = ["source_file", "puncta_count"]

# Save result to new CSV
output_name = "puncta_counts_per_frame.csv"
puncta_counts.to_csv(output_name, index=False)

print(f"\nSaved puncta counts per frame to {output_name}:")
print(puncta_counts)
