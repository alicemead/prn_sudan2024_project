# !Data Warning! The code produces a series some of the output files 
# can be at least 1GB each when merged, so 2GB sortage is needed.

# Caesar Reducer File = sys.argv[1] 
# Edited Subjects CSV File (output file from the get_subject_info code) = sys.argv[2]
# Sudan Roads Shapefile (.shp needed) = sys.argv[3]
# Question Data Minimum Consensus: RECOMMENDED 0.7 = sys.argv[4]
# Bridge Min value of Votes, RECOMMENDED value: 10 = sys.argv[5]
# Paved Road Min value of Votes, RECOMMENDED value: 7 = sys.argv[6]

# This code merges the Zooniverse reducers file with
# the enhanced subjects file. It hardcodes the headings and calculates
# the lon/lat of the x and y points of the line classifications from 
# Zooniverse in EPSG 4326. It then exports these values as an ESRI shapefile. 

# Shape_line data: Takes the reducers file, changes headers and filters out metadata
# into readable columns. Takes the clustered values for x/y, then finds 
# the median of these amounts. Uses the median values to calculate
# the length of wadi markings in lon/lat and produces a shapefile from 
# these amounts.

# Question data: Takes the reducer file, changes headers and filters the 
# dataframe on bridges and paved roads AND data = 0 (yes) to create two 
# reduced data frames, of only answer = YES data. 
# Then calculates the total number of votes per subject and filters on a total 
# number of votes >= 10 (current retirement limit) AND a minimum per cent
#, which is defined in the command line. This is exported to a shapefile.



import pandas as pd
import numpy as np
import sys
import json
import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import box
import os
import ast
from pyproj import Proj, transform


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

    # Ensure extracted values are numerical
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



def merge_csv_files(new_headers, enhanced_csv, merged_file):
    """Merge two CSV files based on a common 'subject_id' column."""
    try:
        df1 = pd.read_csv(new_headers)
        df2 = pd.read_csv(enhanced_csv)
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
        
        # Process columns
        for col in columns_to_median.keys():
            if col in df.columns:
                # Parse only if the value is a string and looks like a list
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
        
        # Save to output file
        df.to_csv(median_file, index=False)
        print("Medians calculated and saved successfully.")
    except FileNotFoundError:
        print(f"The CSV file '{merged_file}' was not found.")


def get_long_lat_for_px(tl_px, br_px, tl_ll, br_ll, x_px, y_px):
    """Convert pixel coordinates to latitude and longitude based on top-left and bottom-right points."""
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
        
        # Ensure required columns are present
        required_columns = ['Lon_0_epsg4326', 'Lat_0_epsg4326', 'Lon_1_epsg4326', 'Lat_1_epsg4326']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan  
        
        # Filter rows to ensure all required metadata is available
        df_filtered = df.dropna(subset=required_columns)
        
        # Define columns for the two points
        points_data = {
            'Median_Wadi_Marking_clusters_x1': 'Lng_Median_Wadi_Marking_clusters_point1',
            'Median_Wadi_Marking_clusters_y1': 'Lat_Median_Wadi_Marking_clusters_point1',
            'Median_Wadi_Marking_clusters_x2': 'Lng_Median_Wadi_Marking_clusters_point2',
            'Median_Wadi_Marking_clusters_y2': 'Lat_Median_Wadi_Marking_clusters_point2'
        }

        # Fixed pixel dimensions for the image
        image_size = (512, 512)  # Width x Height in pixels

        def convert_to_coords(row, x_col, y_col):
            """Convert pixel coordinates to latitude and longitude."""
            # Use the top-left and bottom-right coordinates for conversion
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

        # Loop through each point's x and y columns and apply conversion
        for x_col, output_lng_col in points_data.items():
            y_col = x_col.replace('_x', '_y')
            output_lat_col = output_lng_col.replace('Lng', 'Lat')
            df_filtered[[output_lng_col, output_lat_col]] = df_filtered.apply(
                lambda row: convert_to_coords(row, x_col, y_col), axis=1, result_type='expand'
            )

        # Replace default 0s with NaNs where applicable and save updated DataFrame
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
# Question Data: Bridge and Paved Road Code




def bridge_paved_replace_headers(reducer_file, bridge_paved_new_headers, replacements_dict):
    """
    Replace headers in the CSV file using a dictionary of replacements.
    """
    try:
        df = pd.read_csv(reducer_file, nrows=0)
        df.rename(columns=replacements_dict, inplace=True)
        df.to_csv(bridge_paved_new_headers, index=False)

        with open(reducer_file) as file_in, open(bridge_paved_new_headers, "a") as file_out:
            next(file_in)  # Skip the original header
            for line in file_in:
                file_out.write(line)

        print(f"Headers replaced and saved to {bridge_paved_new_headers}")
    except FileNotFoundError:
        print(f"Error: File {reducer_file} not found.")





def filter_csv(bridge_paved_new_headers, bridge_filtered, paved_filtered):
    """
    Filter the CSV file for specific criteria and save outputs for bridge and paved roads.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(bridge_paved_new_headers)

        # Clean and standardize columns
        df['reducer_key'] = df['reducer_key'].str.strip()  # Remove extra spaces

        # Ensure numeric columns are clean and valid
        df['data.agreement'] = pd.to_numeric(df['data.agreement'], errors='coerce')
        df['data.most_likely'] = pd.to_numeric(df['data.most_likely'], errors='coerce')

        # Drop rows where numeric columns contain NaN
        df.dropna(subset=['data.agreement', 'data.most_likely'], inplace=True)

        # Debug: Check for unique values in reducer_key and filtered rows
        print(f"Unique reducer_keys: {df['reducer_key'].unique()}")

        # Apply filters for bridge and paved roads
        bridge_df = df[
            (df['reducer_key'] == 'bridge') &
            (df['data.most_likely'] == 0)
        ]
        paved_df = df[
            (df['reducer_key'] == 'paved_road') &
            (df['data.most_likely'] == 0)
        ]

        # Debug: Check number of rows in each filtered DataFrame
        print(f"Bridge filtered rows: {len(bridge_df)}")
        print(f"Paved roads filtered rows: {len(paved_df)}")

        # Save filtered DataFrames to CSV
        if not bridge_df.empty:
            bridge_df.to_csv(bridge_filtered, index=False)
            print(f"Bridge data saved successfully to '{bridge_filtered}'.")
        else:
            print("No rows found for bridge filter. Skipping CSV save.")

        if not paved_df.empty:
            paved_df.to_csv(paved_filtered, index=False)
            print(f"Paved roads data saved successfully to '{paved_filtered}'.")
        else:
            print("No rows found for paved roads filter. Skipping CSV save.")

        return bridge_filtered, paved_filtered

    except Exception as e:
        print(f"Error during filtering: {e}")





def bridge_compute_total_votes(bridge_filtered, bridge_total_class):
    try:
        # Attempt to read the file
        df = pd.read_csv(bridge_filtered)
        print(f"File loaded successfully. Columns found: {df.columns.tolist()}")
        
        # Strip any leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        # Ensure the required columns exist
        required_columns = ['data.agreement', 'data.num_votes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The file must contain the following columns: {', '.join(missing_columns)}")
        
        
        # Check for NaN values in the required columns before the calculation
        print(f"Rows with NaN in 'data.agreement' or 'data.num_votes' before calculation:")
        print(df[df[['data.agreement', 'data.num_votes']].isnull().any(axis=1)])

        # Handle rows with NaN values: Optionally, you can drop or fill NaNs
        # For now, we will fill NaN values with 0 for the sake of calculation, or you could drop them
        df['data.agreement'].fillna(0, inplace=True)
        df['data.num_votes'].fillna(0, inplace=True)
        
        # Ensure the columns are numeric, coercing any errors to NaN
        df['data.agreement'] = pd.to_numeric(df['data.agreement'], errors='coerce')
        df['data.num_votes'] = pd.to_numeric(df['data.num_votes'], errors='coerce')
        
        # Check for NaN values after conversion and before calculation
        print(f"Rows with NaN in 'data.agreement' or 'data.num_votes' after conversion:")
        print(df[df[['data.agreement', 'data.num_votes']].isnull().any(axis=1)])
        
        # Calculate the product of the required columns (for each row)
        df['total_classifications'] = df['data.num_votes'] / df['data.agreement']
        
        # Save the updated DataFrame to a new CSV file
        df.to_csv(bridge_total_class, index=False)
        print(f"Processed file saved as: {bridge_total_class}")
    
    except FileNotFoundError:
        print(f"Error: The file '{bridge_filtered}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{bridge_filtered}' is empty or improperly formatted.")
    except ValueError as e:
        print(f"ValueError: {e}")





def paved_compute_total_votes(paved_filtered, paved_total_class):
    try:
        # Attempt to read the file
        df = pd.read_csv(paved_filtered)
        print(f"File loaded successfully. Columns found: {df.columns.tolist()}")
        
        # Strip any leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        # Ensure the required columns exist
        required_columns = ['data.agreement', 'data.num_votes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The file must contain the following columns: {', '.join(missing_columns)}")
        
        
        # Check for NaN values in the required columns before the calculation
        print(f"Rows with NaN in 'data.agreement' or 'data.num_votes' before calculation:")
        print(df[df[['data.agreement', 'data.num_votes']].isnull().any(axis=1)])

        # Handle rows with NaN values: Optionally, you can drop or fill NaNs
        # For now, we will fill NaN values with 0 for the sake of calculation, or you could drop them
        df['data.agreement'].fillna(0, inplace=True)
        df['data.num_votes'].fillna(0, inplace=True)
        
        # Ensure the columns are numeric, coercing any errors to NaN
        df['data.agreement'] = pd.to_numeric(df['data.agreement'], errors='coerce')
        df['data.num_votes'] = pd.to_numeric(df['data.num_votes'], errors='coerce')
        
        # Check for NaN values after conversion and before calculation
        print(f"Rows with NaN in 'data.agreement' or 'data.num_votes' after conversion:")
        print(df[df[['data.agreement', 'data.num_votes']].isnull().any(axis=1)])
        
        # Calculate the product of the required columns (for each row)
        df['total_classifications'] = df['data.num_votes'] / df['data.agreement']
        
        # Save the updated DataFrame to a new CSV file
        df.to_csv(paved_total_class, index=False)
        print(f"Processed file saved as: {paved_total_class}")
    
    except FileNotFoundError:
        print(f"Error: The file '{paved_filtered}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{paved_filtered}' is empty or improperly formatted.")
    except ValueError as e:
        print(f"ValueError: {e}")





# Function to reduce the agreement file based on `data.agreement` and `total_classifications`
def bridge_reduce_by_min_value(bridge_total_class, percentage, bridge_min_value, bridge_reduced):
    try:
        df = pd.read_csv(bridge_total_class)
        reduced_df = df[(df['data.agreement'] >= percentage) & (df['total_classifications'] >= bridge_min_value)]
        bridge_reduced = bridge_total_class.replace('_agreement.csv', '_reduced.csv')
        reduced_df.to_csv(bridge_reduced, index=False)
        print(f"Reduced data saved as '{bridge_reduced}' with {len(reduced_df)} rows.")
        return bridge_reduced
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")





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





# Function to merge processed CSV files with an external CSV on `subject_id`
def bridge_merge_with_external_file(bridge_total_class, enhanced_csv, bridge_merged):

    try:
        processed_df = pd.read_csv(bridge_total_class)
        external_df = pd.read_csv(enhanced_csv)
        merged_df = pd.merge(processed_df, external_df, on='subject_id', how='left')
        merged_df.to_csv(bridge_merged, index=False)
        print(f"Merged data saved as '{bridge_merged}'")
        return bridge_merged
    except FileNotFoundError as e:
        print(f"Error: {e}")





# Function to merge processed CSV files with an external CSV on `subject_id`
def paved_merge_with_external_file(paved_total_class, enhanced_csv, paved_merged):
    """
    Merge processed data with an external CSV file.
    """
    try:
        processed_df = pd.read_csv(paved_total_class)
        external_df = pd.read_csv(enhanced_csv)
        merged_df = pd.merge(processed_df, external_df, on='subject_id', how='left')
        merged_df.to_csv(paved_merged, index=False)
        print(f"Merged data saved as '{paved_merged}'")
        return paved_merged
    except FileNotFoundError as e:
        print(f"Error: {e}")





def bridge_match_and_reduce_shapefile(bridge_merged, admin_boundaries_shapefile, bridge_shapefile):
    """Filter the shapefile based on bounding boxes defined in a CSV file and save the reduced shapefile."""
    
    # Step 1: Load the CSV file
    df = pd.read_csv(bridge_merged)
    required_columns = ['Lat_0_epsg4326', 'Lon_0_epsg4326']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in the CSV: {', '.join(missing_cols)}")
        return

    # Step 2: Generate bounding boxes based on provided coordinates
    bounding_boxes = [box(lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)  # small box around the point
                      for lat, lon in zip(df['Lat_0_epsg4326'], df['Lon_0_epsg4326'])]
    bounding_boxes_gdf = gpd.GeoDataFrame(geometry=bounding_boxes, crs="EPSG:4326")

    # Step 3: Load the administrative boundaries shapefile
    try:
        polygons_gdf = gpd.read_file(admin_boundaries_shapefile)
    except FileNotFoundError:
        print("Error: Administrative boundaries shapefile not found.")
        sys.exit(1)
    
    # Ensure the geometry column is set correctly
    if polygons_gdf.geometry.name != 'geometry':
        polygons_gdf = polygons_gdf.set_geometry('geometry')

    # Step 4: Check if CRS matches, otherwise reproject
    if polygons_gdf.crs != bounding_boxes_gdf.crs:
        polygons_gdf = polygons_gdf.to_crs(bounding_boxes_gdf.crs)

    # Step 5: Filter polygons that intersect with any bounding box using spatial join for efficiency
    matched_polygons = gpd.sjoin(polygons_gdf, bounding_boxes_gdf, how="inner", predicate='intersects')
    
    # Step 6: Save the reduced shapefile
    matched_polygons.to_file(bridge_shapefile)
    print(f"Reduced shapefile '{bridge_shapefile}' created successfully, containing only polygons intersecting with CSV bounding boxes.")




def paved_match_and_reduce_shapefile(paved_merged, admin_boundaries_shapefile, paved_shapefile):
    """Filter the shapefile based on bounding boxes defined in a CSV file and save the reduced shapefile."""
    
    # Step 1: Load the CSV file
    df = pd.read_csv(paved_merged)
    required_columns = ['Lat_0_epsg4326', 'Lon_0_epsg4326']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in the CSV: {', '.join(missing_cols)}")
        return

    # Step 2: Generate bounding boxes based on provided coordinates
    bounding_boxes = [box(lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)  # small box around the point
                      for lat, lon in zip(df['Lat_0_epsg4326'], df['Lon_0_epsg4326'])]
    bounding_boxes_gdf = gpd.GeoDataFrame(geometry=bounding_boxes, crs="EPSG:4326")

    # Step 3: Load the administrative boundaries shapefile
    try:
        polygons_gdf = gpd.read_file(admin_boundaries_shapefile)
    except FileNotFoundError:
        print("Error: Administrative boundaries shapefile not found.")
        sys.exit(1)
    
    # Ensure the geometry column is set correctly
    if polygons_gdf.geometry.name != 'geometry':
        polygons_gdf = polygons_gdf.set_geometry('geometry')

    # Step 4: Check if CRS matches, otherwise reproject
    if polygons_gdf.crs != bounding_boxes_gdf.crs:
        polygons_gdf = polygons_gdf.to_crs(bounding_boxes_gdf.crs)

    # Step 5: Filter polygons that intersect with any bounding box using spatial join for efficiency
    matched_polygons = gpd.sjoin(polygons_gdf, bounding_boxes_gdf, how="inner", op='intersects')
    
    # Step 6: Save the reduced shapefile
    matched_polygons.to_file(paved_shapefile)
    print(f"Reduced shapefile '{paved_shapefile}' created successfully, containing only polygons intersecting with CSV bounding boxes.")






if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python process_caesar_reducers.py <Caesar Reducer File>   <Edited Subjects CSV File>   <Sudan Roads Shapefile>   <Question Data Minimum Consensus: RECOMMENDED 0.7>   <Bridge Min value of Votes: RECOMMENDED 10>  <Paved Road Min value of Votes: RECOMMENDED 7>")
        sys.exit(1)

    reducer_file = sys.argv[1]
    enhanced_csv = sys.argv[2]
    admin_boundaries_shapefile = sys.argv[3]
    percentage = float(sys.argv[4])
    bridge_min_value = float(sys.argv[5])
    paved_min_value = float(sys.argv[6])

    # Filenames for each step
    new_headers = f"reducer_with_new_headers_.csv"
    merged_file = f"reducer_merged.csv"
    median_file = f"reducer_median.csv"
    reduced_coords_filename = f"reducer_coords.csv" 
    shapefile_output = f'length_of_wadi.shp'
    bridge_paved_new_headers = "bridge_paved_new_headers.csv"
    bridge_filtered = "bridge.csv"
    paved_filtered = "paved.csv"
    bridge_total_class = 'bridge_total_class.csv'
    paved_total_class = 'paved_total_class.csv'
    bridge_reduced = 'bridge_reduced.csv'
    paved_reduced = 'paved_reduced.csv'
    bridge_merged = 'bridge_merged.csv'
    paved_merged = 'paved_merged.csv'
    bridge_shapefile = f'bridge_=>percent_{percentage}_min_value_{bridge_min_value}.shp'
    paved_shapefile = f'paved_road_=>percent_{percentage}_min_value_{paved_min_value}.shp'
    

    # Step 1: Replace headers
    replacements_dict = {
        "subject_ids": "subject_id",
        "T0_tool0": "Wadi_Marking",
        "data.0": "yes",
        "data.1": "no",
        "data.2": "both",
        
    }

    split_x_y = "split_x_y.csv"
    process_wadi_data(reducer_file, split_x_y)

    replace_headers(split_x_y, new_headers, replacements_dict)

    # Step 2: Merge CSV files
    merge_csv_files(new_headers, enhanced_csv, merged_file)

    # Step 3: Calculate medians on the merged file and save the result in `median_file`
    calculate_median(merged_file, median_file)

    # Step 4: Calculate pixel coordinates using the median file as input, saving to `reduced_coords_filename`
    calculate_pixel_coordinates(median_file, reduced_coords_filename)

    # Step 5: Create shapefile based on the calculated coordinates in `reduced_coords_filename`
    create_shapefile_for_category(
        pd.read_csv(reduced_coords_filename),
        "Lng_Median_Wadi_Marking_clusters_point1", "Lat_Median_Wadi_Marking_clusters_point1",
        "Lng_Median_Wadi_Marking_clusters_point2", "Lat_Median_Wadi_Marking_clusters_point2",
        shapefile_output
    )

    bridge_paved_replace_headers(reducer_file, bridge_paved_new_headers, replacements_dict)
        
    filter_csv(bridge_paved_new_headers, bridge_filtered, paved_filtered)

    bridge_compute_total_votes(bridge_filtered, bridge_total_class)
    
    paved_compute_total_votes(paved_filtered, paved_total_class)

    bridge_reduce_by_min_value(bridge_total_class, percentage, bridge_min_value, bridge_reduced)

    paved_reduce_by_min_value(paved_total_class, percentage, paved_min_value, paved_reduced)

    bridge_merge_with_external_file(bridge_total_class, enhanced_csv, bridge_merged)

    paved_merge_with_external_file(paved_total_class, enhanced_csv, paved_merged)

    bridge_match_and_reduce_shapefile(bridge_merged, admin_boundaries_shapefile, bridge_shapefile)

    paved_match_and_reduce_shapefile(paved_merged, admin_boundaries_shapefile, paved_shapefile)

