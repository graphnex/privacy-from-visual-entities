#!/usr/bin/env python
#
# MLP
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/05/19
# Modified Date: 2023/09/08
# -----------------------------------------------------------------------------

# PyTorch libraries
import torch
import torch.nn as nn

from torchsummary import summary

from srcs.utils import (
    set_seed,
    print_model_parameters,
)


#############################################################################
class MLP(nn.Module):
    """Multi-layer perceptron

    Preparation and forward pass for the multi-layer perceptron.
    """

    def __init__(
        self,
        config,
    ):
        """Constructor of the class

        Arguments:
            - n_layers: number of layers
            - n_features: number of features
            - n_features_layer: number of features per layer
            - n_out_classes: number of output classes (2 for binary classification)
            - b_batch_norm: boolean to use the batch normalization. Deafult: false
            - b_softmax: boolean to use softmax. Default: false
            - dropout_prob: probability for the droput layer. Default: 0
        """
        super(MLP, self).__init__()

        self.model_name = config["model_name"]

        net_params = config["net_params"]

        self.node_feat_size = net_params["node_feat_size"]

        in_dim = net_params["n_graph_nodes"] * net_params["node_feat_size"]
        hidden_dim = net_params["hidden_dim"]
        n_classes = net_params["num_out_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout_prob = net_params["dropout"]
        n_layers = net_params["num_layers"]
        b_batch_norm = net_params["use_bn"]

        self.dropout = nn.Dropout(p=in_feat_dropout)

        # Hidden layers
        feat_list = [in_dim] + [hidden_dim] * n_layers

        layers = []
        for l_idx in range(n_layers):
            # Add batch normalisation layer (if activated)
            if b_batch_norm:
                # layers.append(nn.BatchNorm1d(feat_list[l_idx + 1]))
                layers.append(
                    nn.Linear(
                        feat_list[l_idx], feat_list[l_idx + 1], bias=True
                    ),
                )
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(
                    nn.Linear(
                        feat_list[l_idx], feat_list[l_idx + 1], bias=True
                    ),
                )

            layers.append(nn.ReLU())
            layers.append(
                nn.Dropout(dropout_prob)
            )  # This was not present before

        self.layers = nn.Sequential(*layers)

        # Fully connected layer for output
        self.fc = nn.Linear(feat_list[-1], n_classes)

    def print_number_parameters(self):
        """ """
        # Print classifier parameters to screen
        print("\nMLP Layers parameters: ")
        print_model_parameters(self.layers)

        print("\nFinal fully connected layer parameters: ")
        print_model_parameters(self.fc)

    def get_model_name(self):
        return self.model_name

    def set_batch_size(self, batch_size):
        """Dummy function based on other networks"""
        batch_sz = batch_size

    def forward(self, x):
        """Forward pass.

        Arguments:
            - node_feats_var: node features
            - adj_mat: adjacency matrix

        Return:
            - outputs:
        """

        if self.node_feat_size == 1:
            x = torch.squeeze(x, -1)
        else:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2])

        x_d = self.dropout(x)
        x_l = self.layers(x_d)
        logits = self.fc(x_l)

        return logits


#############################################################################
import numpy as np

from srcs.nets.MLPReadout import MLPReadout


class iMLP(nn.Module):
    """Multi-layer perceptron

    Preparation and forward pass for the multi-layer perceptron.
    """

    def __init__(
        self,
        config,
    ):
        """Constructor of the class

        Arguments:
            - config: number of layers
        """
        super(iMLP, self).__init__()

        set_seed(config["params"]["seed"])
        self.rng = np.random.default_rng()

        self.load_config(config)

        self.initialise_model()

        # Print model summary and number of parameters
        self.print_model_summary()
        self.print_number_parameters()

    def get_model_name(self):
        return self.model_name

    def load_config(self, config):
        """ """
        self.config = config

        self.model_name = config["model_name"]
        self.net_params = config["net_params"]

        self.img_size = self.net_params["img_size"]

    def initialise_model(self):
        """ """
        in_dim = self.img_size * self.img_size * 3

        n_classes = self.net_params["num_out_classes"]
        n_layers = self.net_params["num_layers"]

        self.net_mlp = MLPReadout(in_dim, n_classes, L=n_layers)

    def print_number_parameters(self):
        """ """
        # Print classifier parameters to screen
        print("\niMLP Layers parameters: ")
        print_model_parameters(self.net_mlp)

    def print_model_summary(self):
        """ """
        img_size = self.net_params["img_size"]

        print("\iMLP: Classifier")
        print(summary(self.net_mlp, input=((1, 3, img_size, img_size))))

    def set_batch_size(self, batch_size):
        """Dummy function based on other networks
        @param batch_size:
        """
        batch_sz = batch_size

    def forward(self, img, filename):
        """Forward pass.

        Arguments:
            - img: imgs
            - filename: adjacency matrix

        Return:
            - outputs:
        """
        num_imgs = img.size()[0]
        img_vec = img.view(num_imgs, -1)

        if img_vec.shape[1] == 1:
            x = torch.squeeze(img_vec, -1)
        else:
            x = img_vec

        logits = self.net_mlp(x)

        return logits, None
