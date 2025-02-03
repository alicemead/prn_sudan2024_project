# This code is taking the reduced CSV file from the process_caesar_reducers_MappingWadis.py
# code and cropping the data based on a choosen AOI. List the area PCODES you want to focus on
# to mask the data.

# The reduced_coords.csv file from the process_caesar_reducers code = sys.argv[1] 
# Sudan Admin Boundaries Level 1 shapefile (can download from HDX) = sys.argv[2] 
# List of the Admin Level 1 PCODE(S) of your AOI (can be found in shapefile) = sys.argv[3]


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import csv
import sys
import os

def categorize_images_by_boundary(reduced_coords, admin_boundaries_shapefile, pcode_list):
    try:
        df = pd.read_csv(reduced_coords)
        print("CSV Columns:", df.columns)

        required_columns = ['Lon_0_epsg4326', 'Lat_0_epsg4326']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {', '.join([col for col in required_columns if col not in df.columns])}")

        geometry = [Point(xy) for xy in zip(df['Lon_0_epsg4326'], df['Lat_0_epsg4326'])]
        geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        polygons_gdf = gpd.read_file(admin_boundaries_shapefile)
        if 'ADM1_PCODE' not in polygons_gdf.columns or 'geometry' not in polygons_gdf.columns:
            raise ValueError("The required columns ('ADM1_PCODE', 'geometry') are missing in the admin boundaries shapefile.")

        points_in_polygons = gpd.sjoin(geo_df, polygons_gdf[['ADM1_PCODE', 'geometry']], how="left", predicate="within")
        df['Sudan_PCODE'] = points_in_polygons['ADM1_PCODE']

    
        df.to_csv(pcode_list, index=False)
        print(f"Images categorized and saved to {pcode_list}")
        return pcode_list
    except Exception as e:
        print(f"Error in categorize_images_by_boundary: {e}")
        return None



def target_pcode_classifications(PCODES, pcode_list, aoi_list):
    try:
        df = pd.read_csv(pcode_list)
        
        if 'Sudan_PCODE' not in df.columns:
            raise ValueError("The provided CSV does not contain the required 'Sudan_PCODE' column.")

        # Clean and standardize columns
        df['Sudan_PCODE'] = df['Sudan_PCODE'].astype(str).str.strip()  # Ensure it's a string
        
        # Filter for 'SD01'
        df_filtered = df[df['Sudan_PCODE'] == 'SD01']

        # Save filtered DataFrames to CSV
        if not df_filtered.empty:
            df_filtered.to_csv(aoi_list, index=False)
            print(f"Filtered data saved successfully to '{aoi_list}'.")
        else:
            print("No rows found for 'SD01'. Skipping CSV save.")

        return aoi_list

    except Exception as e:
        print(f"Error in target_pcode_classifications: {e}")
        return None




def create_shapefile_for_category(aoi_list, x_col_start, y_col_start, x_col_end, y_col_end, masked_line_data):
    if not all(col in aoi_list.columns for col in [x_col_start, y_col_start, x_col_end, y_col_end]):
        print(f"Error: Missing required columns in DataFrame.")
        return

    filtered_df = aoi_list.dropna(subset=[x_col_start, y_col_start, x_col_end, y_col_end])
    lines = []
    for _, row in filtered_df.iterrows():
        start_lon, start_lat = row[x_col_start], row[y_col_start]
        end_lon, end_lat = row[x_col_end], row[y_col_end]
        lines.append({'subject_id': row.get('subject_id', 'Unknown'), 'geometry': LineString([(start_lon, start_lat), (end_lon, end_lat)])})

    if lines:
        lines_gdf = gpd.GeoDataFrame(lines, crs="EPSG:4326")
        lines_gdf.to_file(masked_line_data)
        print(f"Shapefile '{masked_line_data}' created successfully.")
    else:
        print("No valid LineStrings could be created.")




if __name__ == "__main__":
    reduced_coords = sys.argv[1] 
    admin_boundaries_shapefile = sys.argv[2]
    PCODES = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3].strip() else []

    pcode_list = "subject_set_catogoised_by_PCODE.csv"
    aoi_list = f"length_wadi_{PCODES}_data.csv"
    masked_line_data = f'length_wadi_{PCODES}.shp'

    categorize_images_by_boundary(reduced_coords, admin_boundaries_shapefile, pcode_list)
    target_pcode_classifications(PCODES, pcode_list, aoi_list)
    create_shapefile_for_category(pd.read_csv(aoi_list),
                                  "Lng_Median_Wadi_Marking_clusters_point1", "Lat_Median_Wadi_Marking_clusters_point1",
                                  "Lng_Median_Wadi_Marking_clusters_point2", "Lat_Median_Wadi_Marking_clusters_point2",
                                  masked_line_data)
