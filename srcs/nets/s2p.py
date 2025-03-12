#!/usr/bin/env python
#
# Scene-to-privacy models (fully connected and MLP-based variants)
#
##################################################################################
# Authors:
# - Alessio Xompero
#
# Email: a.xompero@qmul.ac.uk
#
#  Created Date: 2023/01/17
# Modified Date: 2023/02/25
#
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

from srcs.utils import (
    device,
    print_model_parameters,
)


##############################################################################
# Utility function for the class
#
def get_pre_trained_model(arch, n_scene_categories, repo_dir):
    # model_file = './models/%s_places365.pth.tar' % arch
    model_file = os.path.join(
        repo_dir, "resources", "%s_places365.pth.tar" % arch
    )

    if not os.access(model_file, os.W_OK):
        weight_url = os.path.join(
            "http://places2.csail.mit.edu/models_places365/",
            "%s_places365.pth.tar" % arch,
        )
        print(
            "wget -P " + os.path.join(repo_dir, "resources") + " " + weight_url
        )
        os.system(
            "wget -P " + os.path.join(repo_dir, "resources") + " " + weight_url
        )

    backbone = models.__dict__[arch](num_classes=n_scene_categories)

    checkpoint = torch.load(
        model_file, map_location=lambda storage, loc: storage
    )

    state_dict = {
        str.replace(k, "module.", ""): v
        for k, v in checkpoint["state_dict"].items()
    }

    backbone.load_state_dict(state_dict)

    return backbone


#####################################################################


class SceneToPrivacyClassifier(nn.Module):
    def __init__(self, config):
        super(SceneToPrivacyClassifier, self).__init__()

        self.model_name = config["model_name"]

        self.root_dir = config["paths"]["root_dir"]

        self.n_scene_categories = config["net_params"]["num_scene_cat"]
        self.n_out_classes = config["net_params"]["num_out_classes"]
        self.backbone_arch = config["net_params"]["backbone_arch"]

        self.b_bce = config["params"]["use_bce"]

        self.backbone = get_pre_trained_model(
            self.backbone_arch, self.n_scene_categories, self.root_dir
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone = self.backbone.to(device)
        self.backbone.eval()

        self.scene_to_privacy_fc_layer()

        self.initialise_fc_layer()

    def print_number_parameters(self):
        """ """
        # Print classifier parameters to screen
        print("\nPre-trained ResNet-50 (scene recognition) parameters: ")
        print_model_parameters(self.backbone)

        print("\nFully connected layer parameters: ")
        print_model_parameters(self.scene_to_privacy_layer)

    def get_model_name(self):
        return self.model_name

    def scene_to_privacy_fc_layer(self):
        """ """
        fc_in = self.n_scene_categories

        if self.b_bce:
            assert self.n_out_classes == 2

            self.scene_to_privacy_layer = nn.Linear(fc_in, 1)

        else:
            assert self.n_out_classes >= 2

            self.scene_to_privacy_layer = nn.Linear(fc_in, self.n_out_classes)

    def initialise_fc_layer(self):
        """ """
        for m in self.scene_to_privacy_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, img):
        """ """
        scene_logits = self.backbone.forward(img)

        privacy_logits = self.scene_to_privacy_layer(scene_logits)

        return privacy_logits, scene_logits


#####################################################################
from srcs.nets.MLPReadout import MLPReadout


class SceneToPrivacyMLPClassifier(nn.Module):
    def __init__(self, config):
        super(SceneToPrivacyMLPClassifier, self).__init__()

        self.model_name = config["model_name"]

        self.net_params = config["net_params"]

        # Load scene classifier pre-trained on Places365
        self.backbone = get_pre_trained_model(
            self.net_params["backbone_arch"],
            self.net_params["num_scene_cat"],
            config["paths"]["root_dir"],
        )

        # Fixed the parameters (no fine-tuning during training)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone = self.backbone.to(device)
        self.backbone.eval()

        # Create MLP classifier
        self.scene_to_privacy_mlp(b_bce=config["params"]["use_bce"])
        self.initialise_fc_layer()

    def print_number_parameters(self):
        """ """
        # Print classifier parameters to screen
        print("\nPre-trained ResNet-50 (scene recognition) parameters: ")
        print_model_parameters(self.backbone)

        print("\nMLP parameters: ")
        print_model_parameters(self.mlp)

    def get_model_name(self):
        return self.model_name

    def scene_to_privacy_mlp(self, b_bce=False):
        """ """
        if b_bce:
            assert self.net_params["num_out_classes"] == 2

            self.mlp = MLPReadout(
                self.net_params["num_scene_cat"],
                1,
                L=self.net_params["num_hidden_layers"],
            )

        else:
            assert self.net_params["num_out_classes"] >= 2

            self.mlp = MLPReadout(
                self.net_params["num_scene_cat"],
                self.net_params["num_out_classes"],
                L=self.net_params["num_hidden_layers"],
            )

    def initialise_fc_layer(self):
        """ """
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, img):
        """ """
        scene_logits = self.backbone.forward(img)

        privacy_logits = self.mlp(scene_logits)

        return privacy_logits, scene_logits
