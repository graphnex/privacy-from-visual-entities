#!/bin/bash

# Download and extract the scene probabilities for each dataset
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/scenes_picalert.zip
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/scenes_VISPR.zip
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/scenes_privacyalert.zip

unzip scenes_picalert.zip -d resources/
unzip scenes_VISPR.zip -d resources/
unzip scenes_privacyalert.zip -d resources/

# Download and extract the detected objects for each dataset
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/objects_picalert.zip
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/objects_VISPR.zip
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/objects_privacyalert.zip

unzip objects_picalert.zip -d resources/
unzip objects_VISPR.zip -d resources/
unzip objects_privacyalert.zip -d resources/

# Download and extract the pre-computed graph data for each dataset
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/graphdata_picalert.zip
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/graphdata_VISPR.zip
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/graphdata_privacyalert.zip
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/graphdata_IPD.zip

unzip graphdata_picalert.zip -d resources/
unzip graphdata_VISPR.zip -d resources/
unzip graphdata_privacyalert.zip -d resources/
unzip graphdata_IPD.zip -d resources/
