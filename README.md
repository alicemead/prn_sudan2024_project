# prn_sudan2024_project
## Code relating to the Zooniverse/ Logistics Cluster: Sudan Road Access project.

The **get_subject_info_MappingWadis.py** code will take the project subject CSV file from Zooniverse and split the metadata into columns in a readable format to be used in the next code. 

The **process_caesar_reducers.py** code is written to take the edited subjects file created above, and the reducers CSV file from [Zooniverse Caesar API](https://caesar.zooniverse.org). The code processes these files into three shapefiles (length of wadi, bridges, paved roads).

The **shapefile_for_area.py** code can reduce the data down by the specific area of focus you want. Please note to use the AOI descriptor which is included in the shapefile. 
