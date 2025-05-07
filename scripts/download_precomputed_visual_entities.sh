#!/bin/bash

# Download and extract the scene probabilities for each dataset
wget https://zenodo.org/records/15348506/files/scenes_picalert.zip
wget https://zenodo.org/records/15348506/files/scenes_VISPR.zip
wget https://zenodo.org/records/15348506/files/scenes_privacyalert.zip

unzip scenes_picalert.zip -d resources/
unzip scenes_VISPR.zip -d resources/
unzip scenes_privacyalert.zip -d resources/

# Download and extract the detected objects for each dataset
wget https://zenodo.org/records/15348506/files/objects_picalert.zip
wget https://zenodo.org/records/15348506/files/objects_VISPR.zip
wget https://zenodo.org/records/15348506/files/objects_privacyalert.zip

unzip objects_picalert.zip -d resources/
unzip objects_VISPR.zip -d resources/
unzip objects_privacyalert.zip -d resources/

# Download and extract the pre-computed graph data for each dataset
wget https://zenodo.org/records/15348506/files/graphdata_picalert.zip
wget https://zenodo.org/records/15348506/files/graphdata_VISPR.zip
wget https://zenodo.org/records/15348506/files/graphdata_privacyalert.zip
wget https://zenodo.org/records/15348506/files/graphdata_IPD.zip

unzip graphdata_picalert.zip -d resources/
unzip graphdata_VISPR.zip -d resources/
unzip graphdata_privacyalert.zip -d resources/
unzip graphdata_IPD.zip -d resources/
