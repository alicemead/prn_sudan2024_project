# PRN: Sudan 2024 Road Map project
## Post-processing code for Zooniverse/ Logistics Cluster: Sudan Road Access project 
## Mapping Wadis Workflow (ONLY)

The **get_subject_info_MappingWadis.py** code will take the project subject CSV file from Zooniverse and split the metadata into columns in a readable format to be used in the next code. 

The **process_caesar_reducers.py** code is written to take the edited subjects file created above, and the reducers CSV file from [Zooniverse Caesar API](https://caesar.zooniverse.org). The code processes these files into three shapefiles (length of wadi, bridges, paved roads).

The **mask_data_to_AOI.py** code can reduce the data down to a specific area of focus. The code is programmed to Admin Level 1, and you must list the PCODE(S) identifier for your region of interest. The shapefile can be downloaded from HDX [here](https://data.humdata.org/dataset/sudan-administrive-boundaries-1).
