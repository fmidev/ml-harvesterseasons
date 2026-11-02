# Convert lat long to csv for soilt temperature stations from soiltempdb metadata file.

from io import StringIO

import numpy as np
import pandas as pd

path_metadata = (
    "/home/ubuntu/data/ML/validation-data/soiltemp/norway_24hr/NO-st-stations.txt"
)
path_output = (
    "/home/ubuntu/data/ML/validation-data/soiltemp/norway_24hr/norway_latlong.csv"
)


# Function to clean column names
def clean_column_names(header):
    return [col.strip().replace(" ", "_") for col in header]


# Read the CSV file and clean the column names
with open(path_metadata, "r", encoding="utf-8") as f:
    header = f.readline().strip().split(",")
    cleaned_header = clean_column_names(header)


# Load the CSV file with cleaned header
metadata = pd.read_csv(
    path_metadata, skiprows=1, names=cleaned_header, on_bad_lines="skip"
)

# Rename the column 'Station_number_(id)' to 'Station'
metadata.rename(columns={"Station_number_(id)": "Station"}, inplace=True)


# Manipulate column values using lambda functions
metadata["Latitude"] = metadata["Latitude"].apply(
    lambda x: float(x.split("º")[0].strip())
)
metadata["Longitude"] = metadata["Longitude"].apply(
    lambda x: float(x.split("º")[0].strip())
)
print(metadata.head())

# Drop rows with missing values in Latitude and Longitude
metadata = metadata.dropna(subset=["Latitude", "Longitude"])

# Drop duplicate rows based on Latitude and Longitude
lat_long_unique = metadata.drop_duplicates(subset=["Latitude", "Longitude", "Station"])
lat_long_unique = lat_long_unique[["Latitude", "Longitude"]]
# Save the cleaned data to a new CSV file
lat_long_unique.to_csv(path_output, index=False)


# print((metadata.columns))
# # metadata = metadata[
#     [
#         "Latitude",
#         "Longitude",
#     ]
# ]  #'Sensor_height'

# metadata = metadata.dropna(
#     subset=[
#         "Latitude",
#         "Longitude",
#     ]
# )  #'Sensor_height'

# lat_long_unique = metadata.drop_duplicates(subset=["Latitude", "Longitude"])
# print(lat_long_unique.head())
# lat_long_unique.to_csv(path_output, index=False)


# def extract_float(value: str) -> float:
#     """
#     Extract float values from the string
#     """
#     return float(value.split("º")[0].strip())


# # extract float values from the string

# metadata["Latitude"] = metadata["Latitude"].apply(extract_float)
# metadata["Longitude"] = metadata["Longitude"].apply(extract_float)

# metadata = metadata.dropna()
# print(metadata.head())


# # Define a function to clean column names
# # def clean_column_names(header):
# #     return [
# #         col.strip().replace("(", "").replace(")", "").replace(" ", "_")
# #         for col in header
# #     ]


# # # Load the CSV file and clean the column names
# # with open(
# #     soitemp_path_1,
# #     "r",
# # ) as f:
# #     header = f.readline().strip().split(",")
# #     print(header)
# #     cleaned_header = clean_column_names(header)
# #     print(cleaned_header)
# #     # Write the cleaned header back to the file
# #     with open(soitemp_path_1, "r") as original_file:
# #         lines = original_file.readlines()

# #     lines[0] = ",".join(cleaned_header) + "\n"

# #     with open(soitemp_path_1, "w") as modified_file:
# #         modified_file.writelines(lines)

# # soiltemp1 = pd.read_csv(soitemp_path_1)
# # print(soiltemp1["Month"].value_counts(), "month")
# # print(soiltemp1["Year"].value_counts(), "year")
# # print(soiltemp1["Day"].value_counts(), "day")

# # Display the first few rows of the numpy array
# # print(data[:5])
