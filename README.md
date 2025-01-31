# prn_sudan2024_project
This repository holds code relating to the Zooniverse/ Logistics Cluster: Sudan Road Access project.

The get_subject_info_MappingWadis.py code will take the project subjects CSV file from Zooniverse and split the metadata into columns
in a readable format to be used in the next code. 

The process_caesar_reducers.py code is written to take the edited subjects file created above, and the reducers CSV file from Zooniverse Caesar API. The code processes these files into three shapefiles (length of wadi, bridges, paved roads).
