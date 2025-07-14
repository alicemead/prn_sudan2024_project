# !Data Warning! The code produces a series some of the output files 
# can be at least 1GB each when merged, so 2GB storage is needed.

# Subjects CSV File (found in the Data Exports tab on the Project Page) = sys.argv[1]
# Caesar Reducer File (.csv) = sys.argv[2] 
# Caesar Extractors File (.csv) = sys.argv[3]
# Sudan Roads Shapefile (.shp needed) = sys.argv[4]
# Date of data accquistion (e.g. 02.02.2025) = sys.argv[5]

# Line Data (wadi marking): 
# Takes the reducers file, changes headers and filters out metadata
# into readable columns. Takes the clustered values for x/y, then finds 
# the median of these amounts. Uses the median values to calculate
# the length of wadi markings in lon/lat and produces a shapefile from 
# these amounts.

# Question Data (paved road):
# Minimum Consensus set to 70%
# Minimum Number of Votes set to 7 people
# Takes the reducer file, changes headers and filters the dataframe on paved roads
# AND data = 0 (yes) to create a reduced data frame, of only YES data. 
# Calculates the total number of votes per subject and filters on a total 
# number of votes >= 7 AND a minimum percent of agreement == 70%
# This data is exported to a shapefile.

# Point Data (bridge): 
# Takes the reducers file, changes headers and filters out metadata
# into readable columns. Takes the clustered values for x/y, then finds 
# the median of these amounts. Uses the median values to calculate the Lon/Lat
# of the birdge points and produces a shapefile of this data. 

import pandas as pd
import numpy as np
import sys
import json
import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import box
import os
import ast
from pyproj import Proj, transform
import hdbscan
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances



def extract_metadata_values(metadata):
    if pd.isna(metadata):  
        return None, None, None, None

    try:
        metadata_dict = ast.literal_eval(metadata)  

        lat_0 = metadata_dict.get("#Lat_0_epsg4326", None)
        lat_1 = metadata_dict.get("#Lat_1_epsg4326", None)
        lon_0 = metadata_dict.get("#Lon_0_epsg4326", None)
        lon_1 = metadata_dict.get("#Lon_1_epsg4326", None)

        return lat_0, lat_1, lon_0, lon_1

    except (ValueError, SyntaxError):
        print(f"Error parsing metadata: {metadata}")
        return None, None, None, None



########################
########################
########################
# Shape_line data: Length of Wadi Code



def process_wadi_data(reducer_file, split_x_y):
    df = pd.read_csv(reducer_file)
    keys_of_interest = ["T0_tool0_clusters_x1", "T0_tool0_clusters_x2", "T0_tool0_clusters_y1", "T0_tool0_clusters_y2"]

    for i, row in df.iterrows():
        try:
            if isinstance(row['data.frame0'], str):
                wadi_data = json.loads(row['data.frame0'].replace("'", '"'))
                for key in keys_of_interest:
                    df.at[i, key] = wadi_data.get(key, [np.nan])[0]
            else:
                for key in keys_of_interest:
                    df.at[i, key] = np.nan
        except json.JSONDecodeError:
            for key in keys_of_interest:
                df.at[i, key] = np.nan

    
    for key in keys_of_interest:
        df[key] = pd.to_numeric(df[key], errors='coerce')
    
    split_x_y = "split_x_y.csv"
    df.to_csv(split_x_y, index=False)
    print(f"Wadi data processed and saved to '{split_x_y}'")
    return split_x_y





def replace_headers(split_x_y, new_headers, replacements):
    """Replace headers in the input file based on a replacement dictionary."""
    with open(new_headers, "w") as fp_out:
        with open(split_x_y) as file:
            for i, line in enumerate(file):
                if i == 0:
                    for old, new in replacements.items():
                        line = line.replace(old, new)
                fp_out.write(line)
    print(f"Headers replaced and saved to {new_headers}")





def merge_csv_files(new_headers, updated_subject_file, merged_file):
    """Merge two CSV files based on a common 'subject_id' column."""
    try:
        df1 = pd.read_csv(new_headers)
        df2 = pd.read_csv(updated_subject_file)
        merged_df = pd.merge(df1, df2, on='subject_id', how='inner')
        merged_df.to_csv(merged_file, index=False)
        print(f"CSV files merged successfully. Merged data saved as {merged_file}")
    except FileNotFoundError:
        print("Error: One or both input files not found.")
        sys.exit(1)





def calculate_median(merged_file, median_file):
    try:
        df = pd.read_csv(merged_file)
        columns_to_median = {
            'Wadi_Marking_clusters_x1': 'Median_Wadi_Marking_clusters_x1',
            'Wadi_Marking_clusters_y1': 'Median_Wadi_Marking_clusters_y1',
            'Wadi_Marking_clusters_x2': 'Median_Wadi_Marking_clusters_x2',
            'Wadi_Marking_clusters_y2': 'Median_Wadi_Marking_clusters_y2',
        }
        
        for col in columns_to_median.keys():
            if col in df.columns:
                def parse_value(x):
                    if isinstance(x, str):
                        try:
                            parsed = ast.literal_eval(x)
                            if isinstance(parsed, (list, tuple)):
                                return parsed
                        except (ValueError, SyntaxError):
                            pass
                    return [x] if pd.notna(x) else []
                
                df[col] = df[col].apply(parse_value)
        
        # Calculate medians
        for col, median_col in columns_to_median.items():
            if col in df.columns:
                df[median_col] = df[col].apply(lambda x: np.median(x) if len(x) > 0 else np.nan)
        
        df.to_csv(median_file, index=False)
        print("Medians calculated and saved successfully.")
    except FileNotFoundError:
        print(f"The CSV file '{merged_file}' was not found.")





def get_long_lat_for_px(tl_px, br_px, tl_ll, br_ll, x_px, y_px):
    width_px = br_px[0] - tl_px[0]
    height_px = br_px[1] - tl_px[1]
    
    # Calculate degrees per pixel
    degs_per_px_x = (br_ll[0] - tl_ll[0]) / width_px
    degs_per_px_y = (tl_ll[1] - br_ll[1]) / height_px
    
    # Calculate longitude and latitude
    lng = tl_ll[0] + degs_per_px_x * x_px
    lat = tl_ll[1] - degs_per_px_y * y_px
    
    return lng, lat





def calculate_pixel_coordinates(median_file, reduced_coords_filename):
    """Convert pixel coordinates to latitude and longitude based on image metadata."""
    try:
        df = pd.read_csv(median_file)
        
        required_columns = ['Lon_0_epsg4326', 'Lat_0_epsg4326', 'Lon_1_epsg4326', 'Lat_1_epsg4326']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan  
        
        
        df_filtered = df.dropna(subset=required_columns)
        
        points_data = {
            'Median_Wadi_Marking_clusters_x1': 'Lng_Median_Wadi_Marking_clusters_point1',
            'Median_Wadi_Marking_clusters_y1': 'Lat_Median_Wadi_Marking_clusters_point1',
            'Median_Wadi_Marking_clusters_x2': 'Lng_Median_Wadi_Marking_clusters_point2',
            'Median_Wadi_Marking_clusters_y2': 'Lat_Median_Wadi_Marking_clusters_point2'
        }

        
        image_size = (512, 512)  # Width x Height in pixels

        def convert_to_coords(row, x_col, y_col):
            #Convert pixel coordinates to latitude and longitude
            
            top_left = (row['Lon_0_epsg4326'], row['Lat_0_epsg4326'])
            bottom_right = (row['Lon_1_epsg4326'], row['Lat_1_epsg4326'])
            
            return get_long_lat_for_px(
                (0, 0),  
                (512, 512), 
                top_left,
                bottom_right,
                row[x_col] if pd.notna(row[x_col]) else 0,
                row[y_col] if pd.notna(row[y_col]) else 0
            ) if pd.notna(row[x_col]) and pd.notna(row[y_col]) else (np.nan, np.nan)

        
        for x_col, output_lng_col in points_data.items():
            y_col = x_col.replace('_x', '_y')
            output_lat_col = output_lng_col.replace('Lng', 'Lat')
            df_filtered[[output_lng_col, output_lat_col]] = df_filtered.apply(
                lambda row: convert_to_coords(row, x_col, y_col), axis=1, result_type='expand'
            )

        
        df_filtered.replace({0: np.nan}, inplace=True)
        df_filtered.to_csv(reduced_coords_filename, index=False)
        print(f"Updated CSV file saved as '{reduced_coords_filename}'")
    except FileNotFoundError:
        print(f"The CSV file '{median_file}' was not found.")





def create_shapefile_for_category(df, x_col_start, y_col_start, x_col_end, y_col_end, output_filename):
    print(f"Columns in the DataFrame: {df.columns.tolist()}")
    if x_col_start in df.columns and y_col_start in df.columns and x_col_end in df.columns and y_col_end in df.columns:
        filtered_df = df[df[x_col_start].notna() & df[y_col_start].notna() & df[x_col_end].notna() & df[y_col_end].notna()]
        lines = []
        for index, row in filtered_df.iterrows():
            start_lon, start_lat = row[x_col_start], row[y_col_start]
            end_lon, end_lat = row[x_col_end], row[y_col_end]
            if pd.notna(start_lon) and pd.notna(start_lat) and pd.notna(end_lon) and pd.notna(end_lat):
                lines.append({'subject_id': row['subject_id'], 'geometry': LineString([(start_lon, start_lat), (end_lon, end_lat)])})
        if lines:
            lines_gdf = gpd.GeoDataFrame(lines, crs="EPSG:4326")
            lines_gdf.to_file(output_filename)
            print(f"Shapefile '{output_filename}' created successfully with {len(lines)} LineStrings.")
        else:
            print("No valid LineStrings could be created.")
    else:
        print(f"Error: One or more of the necessary columns '{x_col_start}', '{y_col_start}', '{x_col_end}', '{y_col_end}' not found in DataFrame.")





########################
########################
########################
# Question Data: Paved Road Code




def paved_replace_headers(reducer_file, paved_new_headers, replacements_dict):
    """
    Replace headers in the CSV file using a dictionary of replacements.
    """
    try:
        df = pd.read_csv(reducer_file, nrows=0)
        df.rename(columns=replacements_dict, inplace=True)
        df.to_csv(paved_new_headers, index=False)

        with open(reducer_file) as file_in, open(paved_new_headers, "a") as file_out:
            next(file_in)  # Skip the original header
            for line in file_in:
                file_out.write(line)

        print(f"Headers replaced and saved to {paved_new_headers}")
    except FileNotFoundError:
        print(f"Error: File {reducer_file} not found.")




def filter_csv(paved_new_headers, paved_filtered):
    """
    Filter the CSV file for specific criteria and save outputs for bridge and paved roads.
    """
    try:
        df = pd.read_csv(paved_new_headers)

        df['reducer_key'] = df['reducer_key'].str.strip()  

        df['data.agreement'] = pd.to_numeric(df['data.agreement'], errors='coerce')
        df['data.most_likely'] = pd.to_numeric(df['data.most_likely'], errors='coerce')

        df.dropna(subset=['data.agreement', 'data.most_likely'], inplace=True)

        print(f"Unique reducer_keys: {df['reducer_key'].unique()}")

        
        paved_df = df[
            (df['reducer_key'] == 'paved_road') &
            (df['data.most_likely'] == 0)
        ]

        print(f"Paved roads filtered rows: {len(paved_df)}")
        if not paved_df.empty:
            paved_df.to_csv(paved_filtered, index=False)
            print(f"Paved roads data saved successfully to '{paved_filtered}'.")
        else:
            print("No rows found for paved roads filter. Skipping CSV save.")

        return paved_filtered

    except Exception as e:
        print(f"Error during filtering: {e}")





def paved_compute_total_votes(paved_filtered, paved_total_class):
    try:
        df = pd.read_csv(paved_filtered)
        print(f"File loaded successfully. Columns found: {df.columns.tolist()}")
        
        df.columns = df.columns.str.strip()
        
        required_columns = ['data.agreement', 'data.num_votes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The file must contain the following columns: {', '.join(missing_columns)}")
        
        print(f"Rows with NaN in 'data.agreement' or 'data.num_votes' before calculation:")
        print(df[df[['data.agreement', 'data.num_votes']].isnull().any(axis=1)])

        df['data.agreement'].fillna(0, inplace=True)
        df['data.num_votes'].fillna(0, inplace=True)
        
        df['data.agreement'] = pd.to_numeric(df['data.agreement'], errors='coerce')
        df['data.num_votes'] = pd.to_numeric(df['data.num_votes'], errors='coerce')
        
        print(f"Rows with NaN in 'data.agreement' or 'data.num_votes' after conversion:")
        print(df[df[['data.agreement', 'data.num_votes']].isnull().any(axis=1)])
        
        df['total_classifications'] = df['data.num_votes'] / df['data.agreement']

        df.to_csv(paved_total_class, index=False)
        print(f"Processed file saved as: {paved_total_class}")
    
    except FileNotFoundError:
        print(f"Error: The file '{paved_filtered}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{paved_filtered}' is empty or improperly formatted.")
    except ValueError as e:
        print(f"ValueError: {e}")




# Function to reduce the agreement file based on `data.agreement` and `total_classifications`
def paved_reduce_by_min_value(paved_total_class, percentage, paved_min_value, paved_reduced):

    try:
        df = pd.read_csv(paved_total_class)
        reduced_df = df[(df['data.agreement'] >= percentage) & (df['total_classifications'] >= paved_min_value)]
        paved_reduced = paved_total_class.replace('_agreement.csv', '_reduced.csv')
        reduced_df.to_csv(paved_reduced, index=False)
        print(f"Reduced data saved as '{paved_reduced}' with {len(reduced_df)} rows.")
        return paved_reduced
    except (FileNotFoundError, KeyError) as e:
       print(f"Error: {e}")




def paved_merge_with_external_file(paved_total_class, updated_subject_file, paved_merged):
    try:
        processed_df = pd.read_csv(paved_total_class)
        external_df = pd.read_csv(updated_subject_file)
        merged_df = pd.merge(processed_df, external_df, on='subject_id', how='left')
        merged_df.to_csv(paved_merged, index=False)
        print(f"Merged data saved as '{paved_merged}'")
        return paved_merged
    except FileNotFoundError as e:
        print(f"Error: {e}")




def paved_match_and_reduce_shapefile(paved_merged, osm_roads_layer_shapefile, paved_shapefile):
    """Filter the shapefile based on bounding boxes defined in a CSV file and save the reduced shapefile."""

    df = pd.read_csv(paved_merged)
    required_columns = ['Lat_0_epsg4326', 'Lon_0_epsg4326']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in the CSV: {', '.join(missing_cols)}")
        return
    
    bounding_boxes = [box(lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)  # small box around the point
                      for lat, lon in zip(df['Lat_0_epsg4326'], df['Lon_0_epsg4326'])]
    bounding_boxes_gdf = gpd.GeoDataFrame(geometry=bounding_boxes, crs="EPSG:4326")

    try:
        polygons_gdf = gpd.read_file(osm_roads_layer_shapefile)
    except FileNotFoundError:
        print("Error: Administrative boundaries shapefile not found.")
        sys.exit(1)
    

    if polygons_gdf.geometry.name != 'geometry':
        polygons_gdf = polygons_gdf.set_geometry('geometry')

  
    if polygons_gdf.crs != bounding_boxes_gdf.crs:
        polygons_gdf = polygons_gdf.to_crs(bounding_boxes_gdf.crs)

  
    matched_polygons = gpd.sjoin(polygons_gdf, bounding_boxes_gdf, how="inner", op='intersects')

    matched_polygons.to_file(paved_shapefile)
    print(f"Reduced shapefile '{paved_shapefile}' created successfully, containing only polygons intersecting with CSV bounding boxes.")



###########
###########
###########
# Point data: Bridge code


def point_process_wadi_data(extractors_file, point_data_split_x_y):
    point_data = pd.read_csv(extractors_file)

    keys_of_interest_0 = ["T0_tool0_x", "T0_tool0_y"]
    keys_of_interest_1 = ["T0_tool1_x", "T0_tool1_y"]
    keys_of_interest_2 = ["T0_tool2_x", "T0_tool2_y"]

    all_keys = keys_of_interest_0 + keys_of_interest_1 + keys_of_interest_2


    for key in all_keys:
        point_data[key] = np.nan


    for i, row in point_data.iterrows():
        try:
            if isinstance(row['data.frame0'], str):
                parsed = json.loads(row['data.frame0'].replace("'", '"'))
                for key in all_keys:
                    point_data.at[i, key] = parsed.get(key, [np.nan])[0]
        except (json.JSONDecodeError, TypeError):
            continue


    for key in all_keys:
        point_data[key] = pd.to_numeric(point_data[key], errors='coerce')


    point_data = point_data.dropna(subset=all_keys, how='all')

    x_key = ["T0_tool0_x", "T0_tool1_x", "T0_tool2_x"]
    y_key = ["T0_tool0_y", "T0_tool1_y", "T0_tool2_y"]

    def collect_points_x(row):
        for key in x_key:
            x = row.get(key)
            if pd.notna(x):
                return x  
        return None

    def collect_points_y(row):
        for key in y_key:
            y = row.get(key)
            if pd.notna(y):
                return y  
        return None


    point_data["x"] = point_data.apply(collect_points_x, axis=1)
    point_data["y"] = point_data.apply(collect_points_y, axis=1)
    # Save to CSV
    point_data.to_csv(point_data_split_x_y, index=False)
    return point_data

    


def run_hdb_scan(point_data_split_x_y, point_data_clustered_file):
    df = pd.read_csv(point_data_split_x_y)

    required_cols = {"x", "y", "subject_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["subject_id"] = df["subject_id"].astype(str)
    df.dropna(subset=["x", "y"], inplace=True)

    def distance(points_per_image, metric):
        points_per_image = df.groupby(["user_id", "classification_id"])

        if points_per_image:
            metric = 'infinity'

        else:
            metric = 'euclidean'

    df["cluster"] = -1
    for b_value, group in df.groupby("subject_id"):
        print(f"Processing group: {b_value} with {len(group)} rows")
        if len(group) < 3:
            print(f"Skipping group '{b_value}' (too small for clustering)")
            continue

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2,
                                    cluster_selection_epsilon=10,
                                    metric=distance)
        labels = clusterer.fit_predict(group[["x", "y"]])

        if (labels == -1).all():
            print(f"Group '{b_value}': all points labeled as noise")

        df.loc[group.index, "cluster"] = labels

    df.to_csv(point_data_clustered_file, index=False)
    print(f"Clustered data saved to '{point_data_clustered_file}'")
    return point_data_clustered_file




def point_calculate_median(point_data_hdb_file, point_data_median_file):
    try:
        df = pd.read_csv(point_data_hdb_file)

        def parse_value(val):
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, (list, tuple)):
                        return parsed
                except (ValueError, SyntaxError):
                    pass
            return [val] if pd.notna(val) else []

        for col in ['x', 'y']:
            if col in df.columns:
                df[col] = df[col].apply(parse_value)

        filtered_df = df[df['cluster'] == 1]


        medians = filtered_df.groupby('subject_id').agg({
            'x': lambda x: np.median([item for sublist in x for item in sublist]) if x.any() else np.nan,
            'y': lambda y: np.median([item for sublist in y for item in sublist]) if y.any() else np.nan
        }).rename(columns={'x': 'Median_clusters_x', 'y': 'Median_clusters_y'}).reset_index()


        df = df.merge(medians, on='subject_id', how='left')


        df.to_csv(point_data_median_file, index=False)
        print("Subject-level medians (cluster=1) calculated and saved successfully.")

    except FileNotFoundError:
        print(f"The CSV file '{point_data_hdb_file}' was not found.")
    



def point_merge_csv_files(point_data_median_file, point_data_updated_subject_file, point_data_merged_file):
    """Merge two CSV files based on a common 'subject_id' column."""
    try:
        df1 = pd.read_csv(point_data_median_file)
        df2 = pd.read_csv(updated_subject_file)
        merged_df = pd.merge(df1, df2, on='subject_id', how='inner')
        merged_df.to_csv(point_data_merged_file, index=False)
        print(f"CSV files merged successfully. Merged data saved as {point_data_merged_file}")
    except FileNotFoundError:
        print("Error: One or both input files not found.")
        sys.exit(1)




def point_get_long_lat_for_px(tl_px, br_px, tl_ll, br_ll, x_px, y_px):
    
    width_px = br_px[0] - tl_px[0]
    height_px = br_px[1] - tl_px[1]
    
    # Calculate degrees per pixel
    degs_per_px_x = (br_ll[0] - tl_ll[0]) / width_px
    degs_per_px_y = (tl_ll[1] - br_ll[1]) / height_px
    
    # Calculate longitude and latitude
    lng = tl_ll[0] + degs_per_px_x * x_px
    lat = tl_ll[1] - degs_per_px_y * y_px
    
    return lng, lat




def point_calculate_pixel_coordinates(point_data_merged_file, point_data_lon_lat_file):
    
    try:
        df = pd.read_csv(point_data_merged_file)
        
        
        required_columns = ['Lon_0_epsg4326', 'Lat_0_epsg4326', 'Lon_1_epsg4326', 'Lat_1_epsg4326']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan  
        
       
        df_filtered = df.dropna(subset=required_columns)
        
       
        points_data = {
            'Median_clusters_x': 'Lng_Median_clusters',
            'Median_clusters_y': 'Lat_Median_clusters',

        }

       
        image_size = (512, 512)  # Width x Height in pixels

        def convert_to_coords(row, x_col, y_col):
            """Convert pixel coordinates to latitude and longitude."""
            
            top_left = (row['Lon_0_epsg4326'], row['Lat_0_epsg4326'])
            bottom_right = (row['Lon_1_epsg4326'], row['Lat_1_epsg4326'])
            
            return get_long_lat_for_px(
                (0, 0),  # Top-left corner in pixel coordinates
                (512, 512),  # Bottom-right corner in pixel coordinates
                top_left,
                bottom_right,
                row[x_col] if pd.notna(row[x_col]) else 0,
                row[y_col] if pd.notna(row[y_col]) else 0
            ) if pd.notna(row[x_col]) and pd.notna(row[y_col]) else (np.nan, np.nan)

        
        for x_col, output_lng_col in points_data.items():
            y_col = x_col.replace('_x', '_y')
            output_lat_col = output_lng_col.replace('Lng', 'Lat')
            df_filtered[[output_lng_col, output_lat_col]] = df_filtered.apply(
                lambda row: convert_to_coords(row, x_col, y_col), axis=1, result_type='expand'
            )

        
        df_filtered.replace({0: np.nan}, inplace=True)
        df_filtered.to_csv(point_data_lon_lat_file, index=False)
        print(f"Updated CSV file saved as '{point_data_lon_lat_file}'")
    except FileNotFoundError:
        print(f"The CSV file '{point_data_median_file}' was not found.")




def point_counts(point_data_lon_lat_file, point_data_bridge_count_file, point_data_ford_count_file, point_data_something_of_interest_count_file):

    df = pd.read_csv(point_data_lon_lat_file)

    bridge_count  = df.groupby('subject_id')['extractor_key'].apply(lambda x: (x=='bridge').sum()).reset_index(name='bridge_count')
    ford_count = df.groupby('subject_id')['extractor_key'].apply(lambda x: (x=='ford').sum()).reset_index(name='ford_count')
    something_of_interest_count = df.groupby('subject_id')['extractor_key'].apply(lambda x: (x=='something_of_interesting').sum()).reset_index(name='something_of_interest_count')

    bridge_count.to_csv(point_data_bridge_count_file, index=False)
    ford_count.to_csv(point_data_ford_count_file, index=False)
    something_of_interest_count.to_csv(point_data_something_of_interest_count_file, index=False)

    try:
        df1 = pd.read_csv(point_data_bridge_count_file)
        df2 = pd.read_csv(point_data_lon_lat_file)
        merged_df = pd.merge(df1, df2, on='subject_id', how='inner')
        merged_df.to_csv(point_data_bridge_count_file, index=False)
        print(f"CSV files merged successfully. Merged data saved as {point_data_merged_file}")
    except FileNotFoundError:
        print("Error: One or both input files not found.")
        sys.exit(1)

    try:
        df1 = pd.read_csv(point_data_ford_count_file)
        df2 = pd.read_csv(point_data_bridge_count_file)
        merged_df = pd.merge(df1, df2, on='subject_id', how='inner')
        merged_df.to_csv(point_data_ford_count_file, index=False)
        print(f"CSV files merged successfully. Merged data saved as {point_data_merged_file}")
    except FileNotFoundError:
        print("Error: One or both input files not found.")
        sys.exit(1)
    
    try:
        df1 = pd.read_csv(point_data_something_of_interest_count_file)
        df2 = pd.read_csv(point_data_ford_count_file)
        merged_df = pd.merge(df1, df2, on='subject_id', how='inner')
        merged_df.to_csv(point_data_all_counts_file, index=False)
        print(f"CSV files merged successfully. Merged data saved as {point_data_merged_file}")
    except FileNotFoundError:
        print("Error: One or both input files not found.")
        sys.exit(1)




def point_classification_to_type(point_data_all_counts_file, point_data_classification_type_file):
    df = pd.read_csv(point_data_all_counts_file)

    df.columns = df.columns.str.encode('ascii', 'ignore').str.decode('ascii').str.strip()

    required_cols = {'subject_id', 'extractor_key'}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"CSV is missing required columns: {required_cols - set(df.columns)}")

    df['extractor_key'] = df['extractor_key'].astype(str).str.strip()
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
   
    most_common = (
        df.groupby(['subject_id', 'extractor_key'])
        .size()
        .reset_index(name='count')
        .sort_values(['subject_id', 'count'], ascending=[True, False])
        .drop_duplicates('subject_id')
        .rename(columns={'extractor_key': 'data_type'})
    )

    merged = pd.merge(df, most_common[['subject_id', 'data_type']], on='subject_id', how='inner')

    final_df = merged[merged['extractor_key'] == merged['data_type']].drop_duplicates('subject_id')

    final_df.to_csv(point_data_classification_type_file, index=False)




def point_filter_csv_bridge(point_data_classification_type_file, point_data_bridge_filtered):
    try:
        df = pd.read_csv(point_data_classification_type_file)

        bridge_df = df[
            (df['data_type'] == 'bridge') 
        ]

        print(f"Bridges filtered rows: {len(bridge_df)}")
        if not bridge_df.empty:
            bridge_df.to_csv(point_data_bridge_filtered, index=False)
            print(f"Paved roads data saved successfully to '{point_data_bridge_filtered}'.")
        else:
            print("No rows found for paved roads filter. Skipping CSV save.")

        return point_data_bridge_filtered

    except Exception as e:
        print(f"Error during filtering: {e}")




def point_filter_csv_ford(point_data_classification_type_file, point_data_ford_filtered):
    try:
        df = pd.read_csv(point_data_classification_type_file)

        ford_df = df[
            (df['data_type'] == 'ford') 
        ]

        print(f"Fords filtered rows: {len(ford_df)}")
        if not ford_df.empty:
            ford_df.to_csv(point_data_ford_filtered, index=False)
            print(f"Paved roads data saved successfully to '{point_data_ford_filtered}'.")
        else:
            print("No rows found for paved roads filter. Skipping CSV save.")

        return point_data_ford_filtered

    except Exception as e:
        print(f"Error during filtering: {e}")




def point_filter_csv_something(point_data_classification_type_file, point_data_something_of_interest_filtered):
    try:
        df = pd.read_csv(point_data_classification_type_file)

        something_of_interest_df = df[
            (df['data_type'] == 'something_of_interesting') 
        ]

        print(f"something_of_interest filtered rows: {len(something_of_interest_df)}")
        if not something_of_interest_df.empty:
            something_of_interest_df.to_csv(point_data_something_of_interest_filtered, index=False)
            print(f"Paved roads data saved successfully to '{point_data_something_of_interest_filtered}'.")
        else:
            print("No rows found for paved roads filter. Skipping CSV save.")

        return point_data_something_of_interest_filtered

    except Exception as e:
        print(f"Error during filtering: {e}")




def point_create_shapefile_for_category(df, x_col, y_col, output_filename):
    if x_col in df.columns and y_col in df.columns:
        filtered_df = df[df[x_col].notna() & df[y_col].notna()]
        geometry = [Point(xy) for xy in zip(filtered_df[x_col], filtered_df[y_col])]
        geo_df = gpd.GeoDataFrame(filtered_df, geometry=geometry, crs="EPSG:4326")
        geo_df.to_file(output_filename)
        print(f"Shapefile '{output_filename}' created successfully.")
    else:
        print(f"Error: One or both columns '{x_col}', '{y_col}' not found in DataFrame.")








if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python process_caesar_reducers.py < zooniverse_project_subject.csv >   < Caesar_Reducer.csv >   < Caesar_Extractor.csv >   < osm_roads_layer.shp >   < date >")
        sys.exit(1)

    subjects_file = sys.argv[1]
    reducer_file = sys.argv[2]
    extractors_file = sys.argv[3]
    osm_roads_layer_shapefile = sys.argv[4]
    date = sys.argv[5]
    
    
    percentage = float(0.7)
    paved_min_value = float(7)
    updated_subject_file = f"enhancedinfo_{os.path.basename(subjects_file)}"
    split_x_y = "split_x_y.csv"
    new_headers = f"reducer_new_headers_.csv"
    merged_file = f"reducer_merged.csv"
    median_file = f"reducer_median.csv"
    reduced_coords_filename = f"reducer_coords.csv" 
    shapefile_output = f'Sudan_Wadi_Road_Markings_{date}.shp'

    paved_new_headers = "paved_new_headers.csv"
    paved_filtered = "paved_filtered.csv"
    paved_total_class = 'paved_total_class.csv'
    paved_reduced = 'paved_reduced.csv'
    paved_merged = 'paved_merged.csv'
    paved_shapefile = f'Sudan_Priority_Roads_PAVED_{date}.shp'

    point_data_split_x_y = "point_data_split_x_y.csv"
    point_data_clustered_file = "point_data_extractors_clustered.csv"
    point_data_hdb_file = "point_data_extractors_clustered.csv"
    point_data_median_file = "point_data_median.csv"
    point_data_merged_file = "point_data_merged.csv"
    point_data_lon_lat_file = "point_data_lon_lat.csv"
    point_data_bridge_count_file = "point_data_bridge_counts.csv"
    point_data_ford_count_file = "point_data_ford_counts.csv"
    point_data_something_of_interest_count_file = "point_data_something_of_interest_counts.csv"
    point_data_all_counts_file = "point_data_counts_merged.csv"
    point_data_classification_type_file = "point_data_final.csv"
    point_data_bridge_filtered = "point_data_bridge.csv"
    point_data_ford_filtered = "point_data_ford.csv"
    point_data_something_of_interest_filtered = "point_data_something_of_interest.csv"

    point_data_distance = "point_data_distance.csv"

    # Step 1: Replace headers
    replacements_dict = {
        "subject_ids": "subject_id",
        "T0_tool0": "Wadi_Marking",
        "data.0": "yes",
        "data.1": "no",
        "data.2": "both",
        
    }
    subjects_file = sys.argv[1]  
    df = pd.read_csv(subjects_file)  

    if 'metadata' not in df.columns:
        raise KeyError("The 'metadata' column is missing from the CSV file.")

    df[['Lat_0_epsg4326', 'Lat_1_epsg4326', 'Lon_0_epsg4326', 'Lon_1_epsg4326']] = df['metadata'].apply(
        lambda x: pd.Series(extract_metadata_values(x))
    )

    df = df.dropna(subset=['Lat_0_epsg4326', 'Lat_1_epsg4326', 'Lon_0_epsg4326', 'Lon_1_epsg4326'], how='all')

    
    df['imsize_x_pix'] = 512
    df['imsize_y_pix'] = 512
    updated_subject_file = f"enhancedinfo_{os.path.basename(subjects_file)}"
    df.to_csv(updated_subject_file, index=False)
    print(f"Data saved to {updated_subject_file}")

    # Line data function calls
    process_wadi_data(reducer_file, split_x_y)
    replace_headers(split_x_y, new_headers, replacements_dict)  
    merge_csv_files(new_headers, updated_subject_file, merged_file)
    calculate_median(merged_file, median_file)
    calculate_pixel_coordinates(median_file, reduced_coords_filename)
    create_shapefile_for_category(
        pd.read_csv(reduced_coords_filename),
        "Lng_Median_Wadi_Marking_clusters_point1", "Lat_Median_Wadi_Marking_clusters_point1",
        "Lng_Median_Wadi_Marking_clusters_point2", "Lat_Median_Wadi_Marking_clusters_point2",
        shapefile_output
    )

    # Question data function calls
    paved_replace_headers(reducer_file, paved_new_headers, replacements_dict)
    filter_csv(paved_new_headers, paved_filtered)
    paved_compute_total_votes(paved_filtered, paved_total_class)
    paved_reduce_by_min_value(paved_total_class, percentage, paved_min_value, paved_reduced) 
    paved_merge_with_external_file(paved_total_class, updated_subject_file, paved_merged)
    paved_match_and_reduce_shapefile(paved_merged, osm_roads_layer_shapefile, paved_shapefile)


    # Point data function calls
    point_process_wadi_data(extractors_file, point_data_split_x_y)
    run_hdb_scan(point_data_split_x_y, point_data_clustered_file)
    point_calculate_median(point_data_hdb_file, point_data_median_file)
    point_merge_csv_files(point_data_median_file, updated_subject_file, point_data_merged_file)
    point_calculate_pixel_coordinates(point_data_merged_file, point_data_lon_lat_file)
    point_counts(point_data_lon_lat_file, point_data_bridge_count_file, point_data_ford_count_file, point_data_something_of_interest_count_file)
    point_classification_to_type(point_data_lon_lat_file, point_data_classification_type_file)
    point_filter_csv_bridge(point_data_classification_type_file, point_data_bridge_filtered)
    point_filter_csv_ford(point_data_classification_type_file, point_data_ford_filtered)
    point_filter_csv_something(point_data_classification_type_file, point_data_something_of_interest_filtered)
    point_create_shapefile_for_category(
        pd.read_csv(point_data_bridge_filtered),
        "Lng_Median_clusters", "Lat_Median_clusters",
        f"Sudan_BRIDGE_data_{date}.shp"
    )
    point_create_shapefile_for_category(
        pd.read_csv(point_data_ford_filtered),
        "Lng_Median_clusters", "Lat_Median_clusters",
        f"Sudan_FORD_data_{date}.shp"
    )
    point_create_shapefile_for_category(
        pd.read_csv(point_data_something_of_interest_filtered),
        "Lng_Median_clusters", "Lat_Median_clusters",
        f"Sudan_something_of_interest_data_{date}.shp"
    )
   
