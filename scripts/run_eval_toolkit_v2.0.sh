#!/bin/bash
#
# Script to automatically format python files according to PEP 8 format style.
# The script requires the pip package 'black' installed.
#   ```pip3 install black```
#
##############################################################################
# Authors: 
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/02/20
# Modified Date: 2023/06/29
#
##############################################################################
# PARAMETERS
#
# Name of the model to evaluate (passed as argument)
MODEL_NAME=llava
#
# Name of the dataset where the model was tested. Default: IPD. Options: IPD
DATASET=PrivacyAlert        # IPD, PrivacyAlert, PicAlert, VISPR, GIPS
# 
# Mode the model was trained:
#   - 'crossval': stratified K-fold cross-validation (training set split into 
#                  K folds, where in turn one fold is used as validation split)
#   - 'final': the training set is not split into subsets and the final model
#              is trained using all training data.
TRAINING_MODE='original' # crossval, final, original
#
# The data is split into 'train', 'val', 'test' if the training mode is 
# 'crossval', otherwise the data is split into 'train' and 'test' if the 
# training mode is 'final'.
SPLIT_MODE="test"
#
# Number of classes in the dataset. Default: 2 (binary classification). 
# Do not change
N_CLS=2
#
# This parameter is for changing the balance between precision and recall in 
# the F_beta score. Default: 2 (recall weighted twice precision). F_beta score
# is computed in the toolkit, but not reported (no need to change).
BETA=2.0
#
MODEL_MODE="best"
##############################################################################
# PATHS AND FILENAME
#
# Add here the directory of the repository in the current machine/server
# ROOT_DIR='/import/smartcameras-002/alessio/GraphNEx/GNN-architecture'
ROOT_DIR=$PWD

### Add here your directory to the image privacy datasets
PREFIX_DIR=''

#
if [ $DATASET == "IPD" ]
then 
    # Directory where the dataset is stored
    DATA_DIR=$PREFIX_DIR/curated_ipd/
    #
    # Output directory where the results information will be saved    
    RES_DIR=$ROOT_DIR/results/ipd/

elif [ $DATASET == "PrivacyAlert" ]
then
    # Directory where the dataset is stored
    DATA_DIR=$PREFIX_DIR/curated-privacy-alert-dataset/
    #
    # Output directory where the results information will be saved    
    RES_DIR=$ROOT_DIR/results/privacyalert/
elif [ $DATASET == "GIPS" ]
then
    # Directory where the dataset is stored
    DATA_DIR=$PREFIX_DIR/gips/
    #
    # Output directory where the results information will be saved    
    RES_DIR=$ROOT_DIR/results/gips
elif [ $DATASET == "PicAlert" ]
then
    # Directory where the dataset is stored
    DATA_DIR=$PREFIX_DIR/curated_picalert/
    #
    # Output directory where the results information will be saved    
    RES_DIR=$ROOT_DIR/results/picalert
elif [ $DATASET == "VISPR" ]
then
    # Directory where the dataset is stored
    DATA_DIR=$PREFIX_DIR/curated_vispr/
    #
    # Output directory where the results information will be saved    
    RES_DIR=$ROOT_DIR/results/vispr
fi
#
# RES_DIR=$ROOT_DIR/results/${N_CLS}-class
#
if [ $TRAINING_MODE == 'crossval' ]
then
    # In the case of cross-validation, the number of the fold used for training the
    # model is passed as argument
    FOLD_ID=0
    #
    # Fullpath of the CSV file with the prediction of the model in the testing set
    # MODEL_PRED_FILE=$RES_DIR/${MODEL_NAME}-${TRAINING_MODE}_${MODEL_MODE}.csv
    MODEL_PRED_FILE=$RES_DIR/${MODEL_NAME}-${FOLD_ID}_${MODEL_MODE}.csv
    #
    # Fullpath of the CSV file where to report the performance measures of the model
    # OUT_FILE=$RES_DIR/res_${DATASET}_fold${FOLD_ID}.csv
elif [ $TRAINING_MODE == 'final' ]
then
    # In the case of cross-validation, the number of the fold used for training the
    # model is passed as argument
    FOLD_ID=-1
    # Fullpath of the CSV file with the prediction of the model in the testing set
    MODEL_PRED_FILE=$RES_DIR/${MODEL_NAME}-${TRAINING_MODE}.csv
    #
    # Fullpath of the CSV file where to report the performance measures of the model
    # OUT_FILE=$RES_DIR/res_${DATASET}.csv
elif [ $TRAINING_MODE == 'original' ]
then
    # In the case of cross-validation, the number of the fold used for training the
    # model is passed as argument
    FOLD_ID=-1
    # Fullpath of the CSV file with the prediction of the model in the testing set
    MODEL_PRED_FILE=$RES_DIR/${MODEL_NAME}-${TRAINING_MODE}_${MODEL_MODE}.csv
    #
    # Fullpath of the CSV file where to report the performance measures of the model
    # OUT_FILE=$RES_DIR/res_${DATASET}_${TRAINING_MODE}.csv
fi

# Fullpath of the CSV file where to report the performance measures of the model
OUT_FILE=$RES_DIR/results_comparison.csv
#
##############################################################################
#
source activate graphnex-gnn

python srcs/eval/toolkit.py \
    --dataset           $DATASET            \
    --model_name        $MODEL_NAME         \
    --model_results_csv $MODEL_PRED_FILE    \
    --out_file          $OUT_FILE           \
    --root_dir          $ROOT_DIR           \
    --data_dir          $DATA_DIR           \
    --n_out_classes     $N_CLS              \
    --training_mode     $TRAINING_MODE      \
    --split_mode        $SPLIT_MODE         \
    --fold_id           $FOLD_ID            \
    --beta              $BETA               \
    --model_mode        $MODEL_MODE
    # --b_filter_imgs

conda deactivate

echo 'Finished!'