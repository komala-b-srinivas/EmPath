import pandas as pd
import os

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"

subject_name = "120614_w_61"
sample_name  = "120614_w_61-PA2-034"

bio_path = os.path.join(BIOVID_ROOT, "biosignals_filtered",
                        subject_name, sample_name + "_bio.csv")

print(f"File exists: {os.path.exists(bio_path)}")

df = pd.read_csv(bio_path, sep="\t")

print(f"\nShape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nAny missing values:\n{df.isnull().sum()}")