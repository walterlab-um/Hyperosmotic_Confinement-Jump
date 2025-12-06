import os
from os.path import dirname, basename
import pandas as pd
from tkinter import filedialog as fd

print("Choose all condensates_AIO-xxxxx.csv files to concatenate:")
lst_path_data = list(
    fd.askopenfilenames(filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
)
folder_data = dirname(lst_path_data[0])
os.chdir(folder_data)

lst_df = []
for f in lst_path_data:
    df = pd.read_csv(f)
    # Add filename column (without the full path)
    df.insert(0, "source_file", basename(f))
    lst_df.append(df)

# Concatenate all DataFrames
df_all = pd.concat(lst_df, ignore_index=True)

# Save combined results
output_filename = "combined_results.csv"
df_all.to_csv(output_filename, index=False)
print(f"\nSuccessfully combined {len(lst_df)} files into {output_filename}")
print(f"Total condensates: {len(df_all)}")
