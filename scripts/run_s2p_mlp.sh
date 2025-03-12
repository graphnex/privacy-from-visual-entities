#!/bin/bash
#
##############################################################################
# Authors: 
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2025/02/06
# Modified Date: 2025/02/06
#
#
##############################################################################

### Add here your directory to the image privacy datasets
IMAGEPRIVACY_DIR=''

# PARAMETERS
today=`date +%Y%m%d`
#
CUDA_DEVICE=2

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

MODEL_NAME=s2p_mlp       # ga_mlp, gpa-rev
MODEL_NAME_B=s2pmlp    # ga_mlp, gpa_rev

GRAPH_MODE=obj-only # obj-only, obj_scene

DSTDIR=backups/
mkdir -p $DSTDIR

for R in {0..0} # run/repetition
do

   for M in {1..1} # mode
   do

      CONFIG_FILE=$ROOT_DIR/configs/${MODEL_NAME}_v1.$M.json

      myzipfile=${today}__${MODEL_NAME}_${DATASET}_v1.${M}.${R}.zip


      ##############################################################################
      #
      # Activate the conda environment
      source activate graphnex-gnn

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
         # --weight_loss
         # --use_bce
         # --resume

      # CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python srcs/main.py                \
      #      --root_dir              $ROOT_DIR           \
      #      --dataset               $DATASET            \
      #      --config                $CONFIG_FILE             \
      #      --training_mode         $TRAINING_MODE      \
      #      --mode                  "testing"           \
      #      --split_mode            "test"              \
      #      --model_mode            "last"              
      #      # --weight_loss
      #      # --use_bce
      #      # --resume


      conda deactivate

      # Evaluation
      source scripts/run_eval_toolkit.sh $MODEL_NAME_B $DATASET $TRAINING_MODE "test" 0 "best"
      # source scripts/run_eval_toolkit.sh $MODEL_NAME $DATASET $TRAINING_MODE "test" 0 "last"

      ##############################################################################
      # Backup model and results

      curr_dir=$PWD

      zip -r  $DSTDIR/$myzipfile trained_models/$DATASET_low/2-class/$MODEL_NAME_B/
      zip -ur $DSTDIR/$myzipfile $CONFIG_FILE 
      # zip -ur $DSTDIR/$myzipfile results/$DATASET_low/${MODEL_NAME}-*.csv 
      # zip -ur $DSTDIR/$myzipfile results/$DATASET_low/results_comparison.csv

   done
done

echo "S2PMLP model and results backed up!"
