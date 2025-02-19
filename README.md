# PRN: Sudan 2024 Road Map project
## Post-processing code for Zooniverse/ Logistics Cluster: Sudan Road Access project 
## Mapping Wadis Workflow (ONLY)

The **process_caesar_reducers.py** code will take the project subject CSV file from Zooniverse 'Data Export' page and split the metadata into columns in a readable format. Then the code will take the edited subjects file, and the reducers CSV file from [Zooniverse Caesar API](https://caesar.zooniverse.org) . This is the download link to the OSM administartion boundaries shapefile also needed to run the code: [link](https://data.humdata.org/dataset/hotosm_sdn_roads?force_layout=desktop) .The code processes these files into three shapefiles (length of wadi, bridges, paved roads).

The **mask_data_to_AOI.py** code can reduce the data down to a specific area of focus. The code is programmed to Admin Level 1, and you must list the PCODE(S) identifier for your region of interest. The shapefile can be downloaded from HDX [here](https://data.humdata.org/dataset/sudan-administrive-boundaries-1).
