#!/bin/bash
#
##############################################################################
# Authors: 
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/08/31
# Modified Date: 2023/10/19
#
#
##############################################################################

### Add here your directory to the image privacy datasets
IMAGEPRIVACY_DIR=''

# PARAMETERS
today=`date +%Y%m%d`
#
CUDA_DEVICE=0

# PATHS
# Directory of the repository in the current machine/server
ROOT_DIR=$PWD

DATASET=IPD        # IPD, PrivacyAlert, PicAlert, VISPR, GIPS
DATASET_low="${DATASET,,}"    # ipd, privacyalert, picalert, vispr, gips

#
# Mode to train the model:
#   - 'crossval': stratified K-fold cross-validation (training set split into 
#                  K folds, where in turn one fold is used as validation split)
#   - 'final': the training set is not split into subsets and the final model
#              is trained using all training data.
TRAINING_MODE='crossval' # crossval, final, original

MODEL_NAME=gip       # ga_mlp, gpa-rev
MODEL_NAME_B=gip    # ga_mlp, gpa_rev

GRAPH_MODE=obj-only # obj-only, obj_scene

DSTDIR=backups/
mkdir -p $DSTDIR

for R in {0..0} # run/repetition
do

for M in {5..5} # mode
do

CONFIG_FILE=$ROOT_DIR/configs/${MODEL_NAME}_v2.$M.json

##############################################################################
#
# Activate the conda environment
source activate graphnex-gnn
# conda activate graphnex-gnn

# Training (Split mode test not used here)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python srcs/main.py                \
    --root_dir              $ROOT_DIR           \
    --dataset               $DATASET            \
    --config                $CONFIG_FILE        \
    --training_mode         $TRAINING_MODE      \
    --mode                  "training"
    # --weight_loss
    # --use_bce
    # --resume

# Testing
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python srcs/main.py                \
    --root_dir              $ROOT_DIR           \
    --dataset               $DATASET            \
    --config                $CONFIG_FILE        \
    --training_mode         $TRAINING_MODE      \
    --mode                  "testing"           \
    --split_mode            "test"              \
    --model_mode            "best"              
# #     --weight_loss
#     # --use_bce
#     # --resume

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python srcs/main.py                \
    --root_dir              $ROOT_DIR           \
    --dataset               $DATASET            \
    --config                $CONFIG_FILE             \
    --training_mode         $TRAINING_MODE      \
    --mode                  "testing"           \
    --split_mode            "test"              \
    --model_mode            "last"              
#     # --weight_loss
#     # --use_bce
#     # --resume
# #

conda deactivate

# Evaluation
source scripts/run_eval_toolkit.sh $MODEL_NAME $DATASET $TRAINING_MODE "test" 0 "best"
source scripts/run_eval_toolkit.sh $MODEL_NAME $DATASET $TRAINING_MODE "test" 0 "last"

   curr_dir=$PWD

   myzipfile=${MODEL_NAME}_${DATASET}_v${M}.0.${R}_${today}.zip

   zip -ur $DSTDIR/$myzipfile configs/${MODEL_NAME}_v2.$M.json
   zip -ur $DSTDIR/$myzipfile results/$DATASET_low/${MODEL_NAME}-*.csv 
   zip -ur $DSTDIR/$myzipfile results/$DATASET_low/results_comparison.csv
   zip -ur $DSTDIR/$myzipfile trained_models/$DATASET_low/2-class/$MODEL_NAME/*

   echo "GIP model and results backed up!"
done
done


