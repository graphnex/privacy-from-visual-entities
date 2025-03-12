#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero
#
# Email: a.xompero@qmul.ac.uk
#
#  Created Date: 2025/02/05
# Modified Date: 2025/02/05
#
# MIT License

# Copyright (c) 2023-2025 GraphNEx

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# ----------------------------------------------------------------------------


# PlacesCNN for scene classification
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision
# (upgrade your torchvision please if there is trn.Resize error)
# Code originally from https://github.com/CSAILVision/places365

import os
import sys

import numpy as np

np.set_printoptions(threshold=sys.maxsize)

import torch
import torch.nn as nn
import torchvision.models as models


from srcs.nets.s2p import get_pre_trained_model

from srcs.utils import (
    device,
    print_model_parameters,
)

from pdb import set_trace as bp

#####################################################################3


class ResNetPlacesFineTuningPrivacy(nn.Module):
    def __init__(self, config):
        super(ResNetPlacesFineTuningPrivacy, self).__init__()

        self.model_name = config["model_name"]

        self.root_dir = config["paths"]["root_dir"]

        self.n_scene_categories = config["net_params"]["num_scene_cat"]
        self.n_out_classes = config["net_params"]["num_out_classes"]
        self.backbone_arch = config["net_params"]["backbone_arch"]

        self.b_bce = config["params"]["use_bce"]

        self.backbone = get_pre_trained_model(
            self.backbone_arch, self.n_scene_categories, self.root_dir
        )

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, self.n_out_classes
        )

        self.initialise_fc_layer()

        for param in self.backbone.parameters():
            param.requires_grad = True

    def print_number_parameters(self):
        """ """
        # Print classifier parameters to screen
        print("\nPre-trained ResNet-50 (scene recognition) parameters: ")
        print_model_parameters(self.backbone)

    def get_model_name(self):
        return self.model_name

    def initialise_fc_layer(self):
        """Initialise the replaced fully connected layer with Xavier uniform."""
        nn.init.xavier_uniform_(self.backbone.fc.weight.data)
        if self.backbone.fc.bias is not None:
            self.backbone.fc.bias.data.zero_()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, img):
        """ """
        privacy_logits = self.backbone.forward(img)

        return privacy_logits
