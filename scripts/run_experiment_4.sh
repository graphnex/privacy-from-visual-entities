#!/bin/bash
#
##############################################################################
# Authors: 
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2025/03/10
# Modified Date: 2025/03/10
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

DATASET=PrivacyAlert        # IPD, PrivacyAlert
DATASET_low="${DATASET,,}"    # convert string to lowercase

#
# Mode to train the model:
#   - 'crossval': stratified K-fold cross-validation (training set split into 
#                  K folds, where in turn one fold is used as validation split)
#   - 'final': the training set is not split into subsets and the final model
#              is trained using all training data.
TRAINING_MODE='original' # crossval, final, original

OUTFILENAME=res_experiment4.csv

MODELS=(tagsvm rnp2svm rnp2svm rnp2ftp s2p s2p_mlp s2p_mlp)
MODES=(0 2 0 0 0 0 1)

for ITER in {0..6}
do
   MODEL_NAME=${MODELS[$ITER]}
   echo $MODEL_NAME

   if [ $MODEL_NAME == "s2p_mlp" ]
   then
      MODEL_NAME_b="s2pmlp"
   else
      MODEL_NAME_b=$MODEL_NAME
   fi

   M=${MODES[$ITER]}

   unzip https://zenodo.org/records/15349470/files/${MODEL_NAME}_${DATASET}_v1.$M.0.zip trained_models/$DATASET_low/2-class/${MODEL_NAME_b}/*

   CONFIG_FILE=$ROOT_DIR/configs/${MODEL_NAME}_v1.$M.json
   #
   # Activate the conda environment
   source activate graphnex-gnn

   # Testing
   CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python srcs/main.py                \
      --root_dir              $ROOT_DIR           \
      --dataset               $DATASET            \
      --config                $CONFIG_FILE        \
      --training_mode         $TRAINING_MODE      \
      --mode                  "testing"           \
      --split_mode            "test"              \
      --model_mode            "best"              

   conda deactivate

   # Evaluation
   source scripts/run_eval_toolkit.sh $MODEL_NAME_b $DATASET $TRAINING_MODE "test" 0 "best" $OUTFILENAME

   rm -r trained_models/$DATASET_low/2-class/${MODEL_NAME}

   echo $MODEL_NAME + " model evaluated!"
done


#####################################################################
DATASET=IPD        # IPD, PrivacyAlert
DATASET_low="${DATASET,,}"    # convert string to lowercase

#
# Mode to train the model:
#   - 'crossval': stratified K-fold cross-validation (training set split into 
#                  K folds, where in turn one fold is used as validation split)
#   - 'final': the training set is not split into subsets and the final model
#              is trained using all training data.
TRAINING_MODE='crossval' # crossval, final, original


for ITER in {0..6}
do
   MODEL_NAME=${MODELS[$ITER]}
   echo $MODEL_NAME

   if [ $MODEL_NAME == "s2p_mlp" ]
   then
      MODEL_NAME_b="s2pmlp"
   else
      MODEL_NAME_b=$MODEL_NAME
   fi

   M=${MODES[$ITER]}

   unzip https://zenodo.org/records/15349470/files/${MODEL_NAME}_${DATASET}_v1.$M.0.zip trained_models/$DATASET_low/2-class/${MODEL_NAME_b}/*

   CONFIG_FILE=$ROOT_DIR/configs/${MODEL_NAME}_v1.$M.json
   #
   # Activate the conda environment
   source activate graphnex-gnn

   # Testing
   CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python srcs/main.py                \
      --root_dir              $ROOT_DIR           \
      --dataset               $DATASET            \
      --config                $CONFIG_FILE        \
      --training_mode         $TRAINING_MODE      \
      --mode                  "testing"           \
      --split_mode            "test"              \
      --model_mode            "best"              

   conda deactivate

   # Evaluation
   source scripts/run_eval_toolkit.sh $MODEL_NAME_b $DATASET $TRAINING_MODE "test" 0 "best" $OUTFILENAME

   rm -r trained_models/$DATASET_low/2-class/${MODEL_NAME}

   echo $MODEL_NAME + " model evaluated!"
done