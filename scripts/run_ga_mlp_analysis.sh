#!/bin/bash
#
##############################################################################
# Authors: 
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2024/02/21
# Modified Date: 2024/02/21
#
#
##############################################################################
# 
# PARAMETERS

### Add here your directory to the image privacy datasets
IMAGEPRIVACY_DIR=''

today=`date +%Y%m%d`

CUDA_DEVICE=0

# Directory of the repository in the current machine/server
ROOT_DIR=$PWD

DATASET=IPD        # IPD, PrivacyAlert
DATASET_low=ipd    # ipd, privacyalert


MODEL_NAME=ga_mlp       # ga_mlp, gpa-rev
MODEL_NAME_B=ga_mlp    # ga_mlp, gpa_rev

DSTDIR=backups/analysis_ga_mlp
mkdir -p $DSTDIR


M=1 # reproducibility run

for R in {8..9}
do
    CONFIG=$ROOT_DIR/configs/analysis_ga_mlp/${MODEL_NAME}_v${R}.json

    myzipfile=${today}__${MODEL_NAME}_${DATASET}_v1.${R}.${M}.zip

    if [ $R -lt 3 ]
    then
        GRAPH_MODE=obj-only
    else
        GRAPH_MODE=obj_scene
    fi

    if [ $DATASET == 'IPD' ]
    then
        TRAINING_MODE='crossval'
    elif [ $DATASET == 'PrivacyAlert' ]
    then
        TRAINING_MODE='original'
    fi

    echo $GRAPH_MODE
    echo $CONFIG
    echo $TRAINING_MODE

##############################################################################
source activate image-privacy

# Training (Split mode test not used here)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python srcs/main.py                \
        --root_dir              $ROOT_DIR           \
        --dataset               $DATASET            \
        --config                $CONFIG             \
        --training_mode         $TRAINING_MODE      \
        --mode                  "training"
        # --weight_loss
        # --use_bce
        # --resume


# Testing
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python srcs/main.py                \
        --root_dir              $ROOT_DIR           \
        --dataset               $DATASET            \
        --config                $CONFIG             \
        --training_mode         $TRAINING_MODE      \
        --mode                  "testing"           \
        --split_mode            "test"              \
        --model_mode            "best"              
        # --weight_loss
        # --use_bce
        # --resume

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python srcs/main.py                \
        --root_dir              $ROOT_DIR           \
        --dataset               $DATASET            \
        --config                $CONFIG             \
        --training_mode         $TRAINING_MODE      \
        --mode                  "testing"           \
        --split_mode            "test"              \
        --model_mode            "last"              
        # --weight_loss
        # --use_bce
        # --resume

conda deactivate

# Evaluation
source scripts/run_eval_toolkit.sh $MODEL_NAME $DATASET $TRAINING_MODE "test" 0 "best"
source scripts/run_eval_toolkit.sh $MODEL_NAME $DATASET $TRAINING_MODE "test" 0 "last"


##############################################################################
# Backup model and results

    curr_dir=$PWD

    zip -r  $DSTDIR/$myzipfile trained_models/$DATASET_low/$MODEL_NAME/
    zip -ur $DSTDIR/$myzipfile configs/analysis_ga_mlp/${MODEL_NAME}_v${R}.json

    zip -ur $DSTDIR/$myzipfile results/$DATASET_low/${MODEL_NAME}-*.csv results/$DATASET_low/results_comparison.csv
    zip -ur $DSTDIR/$myzipfile srcs/ 

    cd $IMAGEPRIVACY_DIR

    echo $PWD

    zip -ur $curr_dir/$DSTDIR/$myzipfile curated_vispr/graph_data/$GRAPH_MODE/node_feats/
    zip -ur $curr_dir/$DSTDIR/$myzipfile curated_picalert/graph_data/$GRAPH_MODE/node_feats/

    cd $curr_dir

    echo "Graph agnostic model and results backed up! (run ${R})"
done

rm results/$DATASET_low/${MODEL_NAME}-*.csv

echo "Finished"