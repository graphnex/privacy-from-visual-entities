#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2025/02/06
# Modified Date: 2025/02/06
# ----------------------------------------------------------------------------

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

#####################################################################3


class ResNetPlacesAndTags(nn.Module):
    def __init__(self, config):
        super(ResNetPlacesAndTags, self).__init__()

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

    def print_number_parameters(self):
        """ """
        # Print classifier parameters to screen
        print("\nPre-trained ResNet-50 (scene recognition) parameters: ")
        print_model_parameters(self.backbone)

    def get_model_name(self):
        return self.model_name

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_scene_tags(self, scene_logits):
        """ """
        topk = self.net_params["topk"]

        probs = torch.sigmoid(scene_logits)

        ind_scene = torch.argsort(probs, dim=1)[:, :-topk]
        top_k_scene_probs = probs.clone()

        for b in range(scene_logits.shape[0]):
            top_k_scene_probs[b, ind_scene[b, :]] = 0

        scene_tags = top_k_scene_probs.clone()
        scene_tags[top_k_scene_probs > 0] = 1.0

        return scene_tags

    def forward(self, img):
        """ """
        scene_logits = self.backbone.forward(img)

        if self.net_params["b_scene_tags"]:
            scene_tags = self.get_scene_tags(scene_logits)

            return scene_tags

        return scene_logits


##########################################################################3

from torchvision.models import resnet101, ResNet101_Weights


class ResNetObjectsAndTags(nn.Module):
    def __init__(self, config):
        super(ResNetObjectsAndTags, self).__init__()

        self.model_name = config["model_name"]

        self.net_params = config["net_params"]

        # Load object classifier pre-trained on ImageNet

        weights = ResNet101_Weights.IMAGENET1K_V2
        self.backbone = resnet101(weights=weights)

        # self.preprocess = weights.transforms() # What is this doing?

        # Fixed the parameters (no fine-tuning during training)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone = self.backbone.to(device)
        self.backbone.eval()

    def print_number_parameters(self):
        """ """
        # Print classifier parameters to screen
        print("\nPre-trained ResNet-101 (object recognition) parameters: ")
        print_model_parameters(self.backbone)

    def get_model_name(self):
        return self.model_name

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_object_tags(self, object_logits):
        """ """
        topk = self.net_params["topk"]

        probs = torch.sigmoid(object_logits)

        ind_scene = torch.argsort(probs, dim=1)[:, :-topk]
        top_k_object_probs = probs.clone()

        for b in range(object_logits.shape[0]):
            top_k_object_probs[b, ind_scene[b, :]] = 0

        object_tags = top_k_object_probs.clone()
        object_tags[top_k_object_probs > 0] = 1.0

        return object_tags

    def forward(self, img):
        """ """
        object_logits = self.backbone.forward(img)

        if self.net_params["b_object_tags"]:
            object_tags = self.get_object_tags(object_logits)

            return object_tags

        return object_logits


##########################################################################3
class ConvNetAndTags(nn.Module):
    def __init__(self, config):
        super(ConvNetAndTags, self).__init__()

        self.model_name = config["model_name"]

        self.net_params = config["net_params"]

        assert (
            self.net_params["b_object_tags"] or self.net_params["b_scene_tags"]
        )

        if self.net_params["b_object_tags"]:
            # Load object classifier pre-trained on ImageNet
            print("Loading object classifier pre-trained on ImageNet...")
            self.objectscnn = ResNetObjectsAndTags(config)

        if self.net_params["b_scene_tags"]:
            # Load scene classifier pre-trained on Places 365
            print("Loading scene classifier pre-trained on Places365...")
            self.scenescnn = ResNetPlacesAndTags(config)

    def print_number_parameters(self):
        """Print classifier parameters to screen."""
        if self.net_params["b_object_tags"]:
            self.objectscnn.print_number_parameters()

        if self.net_params["b_scene_tags"]:
            self.scenescnn.print_number_parameters()

    def get_model_name(self):
        return self.model_name

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

        if self.net_params["b_object_tags"]:
            self.objectscnn.set_batch_size(batch_size)

        if self.net_params["b_scene_tags"]:
            self.scenescnn.set_batch_size(batch_size)

    def forward(self, img):
        """ """
        if (
            self.net_params["b_object_tags"]
            and self.net_params["b_scene_tags"]
        ):
            object_tags = self.objectscnn.forward(img)
            scene_tags = self.scenescnn.forward(img)

            return torch.cat((object_tags, scene_tags), 1)

        if self.net_params["b_object_tags"]:
            return self.objectscnn.forward(img)

        if self.net_params["b_scene_tags"]:
            return self.scenescnn.forward(img)
