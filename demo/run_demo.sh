#!/bin/bash

# INSTRUCTIONS
# Uncomment steps 1 and 2 only the first time you run this script
# There are two step 3: one for a folder with multiple images, and one for a single image

#################################################################
# STEP 1: DOWNLOAD IMAGES FOR DEMO
#
# Download selected images from PrivacyAlert (as shown in the paper, Sec.6.10) in the current directory

# wget https://live.staticflickr.com/65535/50039888078_88fb5d4722_c.jpg -O	50039888078.jpg
# wget https://live.staticflickr.com/65535/50252178257_4e69f47eac_c.jpg -O	50252178257.jpg
# wget https://live.staticflickr.com/65535/50789435118_4c012e37c1_c.jpg -O	50789435118.jpg
# wget https://live.staticflickr.com/65535/50583669162_77829f45af_c.jpg -O	50583669162.jpg
# wget https://live.staticflickr.com/65535/49196229738_3aa9ea88f8_c.jpg -O	49196229738.jpg
# wget https://live.staticflickr.com/65535/47705667841_00724f3a60_c.jpg -O	47705667841.jpg
# wget https://live.staticflickr.com/1887/43830339314_f317a8e1c9_c.jpg	-O	43830339314.jpg
# wget https://live.staticflickr.com/4305/35886895852_c4f26ed6f9_c.jpg	-O	35886895852.jpg
# wget https://live.staticflickr.com/65535/49834471323_2a4fd949b1_c.jpg	-O	49834471323.jpg

#################################################################
# STEP 2: DOWNLOAD AND EXTRACT MODEL IN CURRENT DEMO FOLDER
#
# wget https://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/s2p_PrivacyAlert_v1.0.0.zip

# unzip -j s2p_PrivacyAlert_v1.0.0.zip trained_models/privacyalert/2-class/s2p/best_acc_s2p-original.pth -d .
# mv best_acc_s2p-original.pth s2p.pth

# unzip -j s2p_PrivacyAlert_v1.0.0.zip import/smartcameras-002/alessio/GraphNEx/GNN-privacy/configs/s2p_v1.0.json -d .

#################################################################
# STEP 3: PREDICT IMAGES AS PRIVATE OR PUBLIC WITH MODEL
#
# Activate the conda environment. The command conda is the default one.
# Some OS might not work with the command conda, and the command source can be used as an alternative
source activate graphnex-gnn

# Set the GPU number to use in a multi-GPU machine. The variable 
# is not always needed, and depends on your machine/server
CUDA_DEVICE=0

CONFIG_FILE=s2p_v1.0.json

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python demo_s2p.py \
	s2p.pth \
	$CONFIG_FILE \
	"."

conda deactivate

#################################################################
# ALTERNATIVE STEP 3: PREDICT ONE IMAGE AS PRIVATE OR PUBLIC WITH THE MODEL
#
# Activate the conda environment. The command conda is the default one.
# Some OS might not work with the command conda, and the command source can be used as an alternative
# source activate graphnex-gnn

# # Set the GPU number to use in a multi-GPU machine. The variable 
# # is not always needed, and depends on your machine/server
# CUDA_DEVICE=0

# CONFIG_FILE=s2p_v1.0.json

# IMAGENAME=50039888078.jpg

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python demo_s2p.py \
# 	s2p.pth \
# 	$CONFIG_FILE \
# 	$IMAGENAME

# conda deactivate