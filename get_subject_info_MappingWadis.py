# This code needs the subject CSV file pulled from the Zooniverse data 
# exports page. It separates the metadata into a readable format
# with the long/lat in separate columns. It hardcodes the pixel length 
# values of x and y to 512 pixels. Every image in this project is 512 pixels
# by 512 pixels.

import sys
import pandas as pd
import ast
import os


subjects_file = sys.argv[1]
df = pd.read_csv(subjects_file)


def extract_metadata_values(metadata):
    metadata_dict = ast.literal_eval(metadata)
    
    
    lat_0 = metadata_dict.get("#Lat_0_epsg4326", None)
    lat_1 = metadata_dict.get("#Lat_1_epsg4326", None)
    lon_0 = metadata_dict.get("#Lon_0_epsg4326", None)
    lon_1 = metadata_dict.get("#Lon_1_epsg4326", None)
    
    return lat_0, lat_1, lon_0, lon_1

df[['Lat_0_epsg4326', 'Lat_1_epsg4326', 'Lon_0_epsg4326', 'Lon_1_epsg4326']] = df['metadata'].apply(lambda x: pd.Series(extract_metadata_values(x)))

df = df.dropna(subset=['Lat_0_epsg4326', 'Lat_1_epsg4326', 'Lon_0_epsg4326', 'Lon_1_epsg4326'], how='all')

df['imsize_x_pix'] = 512
df['imsize_y_pix'] = 512

updated_subject_file = f"enhancedinfo_{os.path.basename(subjects_file)}"

df.to_csv(updated_subject_file, index=False)

print(f"Data saved to {updated_subject_file}")
