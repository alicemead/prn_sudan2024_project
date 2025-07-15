# PRN: Sudan Road Access project
## Post-processing code for Zooniverse/ Logistics Cluster: Sudan Road Access project 
## Mapping Wadis Workflow (ONLY)

The **process_caesar_reducers.py** processes the line, question and point data into shapefiles. 
  The code tskes the project subject CSV file from Zooniverse 'Data Export' page and split the metadata into columns in a readable format. 
  Then the code takes the edited subjects file, and the extractors AND reducers CSV files from [Zooniverse Caesar API](https://caesar.zooniverse.org) to process the data into shapefiles.    The code also needs the OSM roads layer shapefile  to run the code, you can download this here: [link](https://data.humdata.org/dataset/hotosm_sdn_roads?force_layout=desktop) .
  The code processes these files into three shapefiles (length of wadi (line), bridges (point) and paved roads (question data maksed to vector road file)).

The **mask_data_to_AOI.py** code can reduce the data down to a specific area of focus. The code is programmed to Admin Level 1, and you must list the PCODE(S) identifier for your region of interest. The shapefile can be downloaded from HDX [here](https://data.humdata.org/dataset/sudan-administrive-boundaries-1).
