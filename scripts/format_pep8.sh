#!/bin/bash
#
# Script to automatically format python files according to PEP 8 format style.
# The script requires the pip package 'black' installed.
# 	```pip3 install black```
#
##############################################################################
# Authors: 
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/02/03
# Modified Date: 2025/03/10
#-----------------------------------------------------------------------------
#
##############################################################################
# PARAMETERS
LINE_LEN=79
#
##############################################################################
#
python -m black --line-length=$LINE_LEN srcs/load_net.py
python -m black --line-length=$LINE_LEN srcs/logging_gnex.py
python -m black --line-length=$LINE_LEN srcs/main.py
python -m black --line-length=$LINE_LEN srcs/mlp_analysis.py
python -m black --line-length=$LINE_LEN srcs/perfmeas_tracker.py
python -m black --line-length=$LINE_LEN srcs/tester_pipeline.py
python -m black --line-length=$LINE_LEN srcs/training_pipelines.py
python -m black --line-length=$LINE_LEN srcs/utils.py
#
python -m black --line-length=$LINE_LEN srcs/baselines/pers_rule.py
python -m black --line-length=$LINE_LEN srcs/baselines/rand_classifier.py
#
python -m black --line-length=$LINE_LEN srcs/datasets/graph_image_privacy.py
python -m black --line-length=$LINE_LEN srcs/datasets/imageprivacy.py
python -m black --line-length=$LINE_LEN srcs/datasets/img_to_gipfeats.py
python -m black --line-length=$LINE_LEN srcs/datasets/img_to_graph.py
python -m black --line-length=$LINE_LEN srcs/datasets/ipd.py
python -m black --line-length=$LINE_LEN srcs/datasets/ipd_graph.py
python -m black --line-length=$LINE_LEN srcs/datasets/normalization.py
python -m black --line-length=$LINE_LEN srcs/datasets/privacyalert.py
python -m black --line-length=$LINE_LEN srcs/datasets/privacyalert_graph.py
python -m black --line-length=$LINE_LEN srcs/datasets/splitter.py
python -m black --line-length=$LINE_LEN srcs/datasets/utils.py
python -m black --line-length=$LINE_LEN srcs/datasets/wrapper.py
python -m black --line-length=$LINE_LEN srcs/datasets/wrapper_imgs.py
#
python -m black --line-length=$LINE_LEN srcs/datasets/gip_feats/demo.py
python -m black --line-length=$LINE_LEN srcs/datasets/gip_feats/gip_features.py
python -m black --line-length=$LINE_LEN srcs/datasets/gip_feats/resnet_v1.py
python -m black --line-length=$LINE_LEN srcs/datasets/gip_feats/test.py
python -m black --line-length=$LINE_LEN srcs/datasets/gip_feats/train.py
python -m black --line-length=$LINE_LEN srcs/datasets/gip_feats/vgg_v1.py
#
python -m black --line-length=$LINE_LEN srcs/eval/toolkit.py
#
python -m black --line-length=$LINE_LEN srcs/graph/concepts2graph.py
python -m black --line-length=$LINE_LEN srcs/graph/img2graph.py
python -m black --line-length=$LINE_LEN srcs/graph/img2graph_base.py
python -m black --line-length=$LINE_LEN srcs/graph/img2obj.py
python -m black --line-length=$LINE_LEN srcs/graph/img2objscene.py
python -m black --line-length=$LINE_LEN srcs/graph/main.py
python -m black --line-length=$LINE_LEN srcs/graph/prior_graph.py
python -m black --line-length=$LINE_LEN srcs/graph/prior_graph_builder.py
#
python -m black --line-length=$LINE_LEN srcs/nets/ga_mlp.py
python -m black --line-length=$LINE_LEN srcs/nets/gip.py
python -m black --line-length=$LINE_LEN srcs/nets/gip_img.py
python -m black --line-length=$LINE_LEN srcs/nets/gpa.py
python -m black --line-length=$LINE_LEN srcs/nets/gpa_img.py
python -m black --line-length=$LINE_LEN srcs/nets/gpa_rev.py
python -m black --line-length=$LINE_LEN srcs/nets/gpa_rev_b.py
python -m black --line-length=$LINE_LEN srcs/nets/grm.py
python -m black --line-length=$LINE_LEN srcs/nets/MLP.py
python -m black --line-length=$LINE_LEN srcs/nets/MLPReadout.py
python -m black --line-length=$LINE_LEN srcs/nets/resnet_ft.py
python -m black --line-length=$LINE_LEN srcs/nets/resnet_svm.py
python -m black --line-length=$LINE_LEN srcs/nets/resnet_v1.py
python -m black --line-length=$LINE_LEN srcs/nets/s2p.py
python -m black --line-length=$LINE_LEN srcs/nets/vgg_v1.py
#
python -m black --line-length=$LINE_LEN srcs/gpa3/demo.py
python -m black --line-length=$LINE_LEN srcs/gpa3/gpa.py
python -m black --line-length=$LINE_LEN srcs/gpa3/test.py
python -m black --line-length=$LINE_LEN srcs/gpa3/train.py
#
python -m black --line-length=$LINE_LEN srcs/vision/concept_extractor.py
python -m black --line-length=$LINE_LEN srcs/vision/concept_models.py
python -m black --line-length=$LINE_LEN srcs/vision/yolo/demo.py
python -m black --line-length=$LINE_LEN srcs/vision/yolo/models.py
python -m black --line-length=$LINE_LEN srcs/vision/yolo/object_detection.py
#
python -m black --line-length=$LINE_LEN srcs/utilities/analysis_obj_card.py
python -m black --line-length=$LINE_LEN srcs/utilities/format_converter.py
