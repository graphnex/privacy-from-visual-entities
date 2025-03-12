#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/08/30
# Modified Date: 2024/09/02
# ----------------------------------------------------------------------------

import inspect
import os
import sys

# setting path
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.insert(0, pp_dir)

import numpy as np

np.set_printoptions(threshold=sys.maxsize, precision=4)

# PyTorch
import torch.nn as nn

# Utilities
from srcs.nets.grm import GraphReasoningModel as GRM
from srcs.utils import (
    set_seed,
    check_if_symmetric,
    print_model_parameters,
)


#############################################################################
#
class GraphImagePrivacy(nn.Module):
    def __init__(self, config):
        super(GraphImagePrivacy, self).__init__()

        self.model_name = config["model_name"]

        self.config = config
        self.net_params = config["net_params"]

        self.node_feature_size = self.net_params["node_feat_size"]

        # Number of output classes (privacy levels)
        self.n_out_classes = self.net_params["num_out_classes"]

        # Number of object categories (for COCO=80, no background)
        self.n_obj_cats = self.net_params["num_obj_cat"]

        self.n_graph_nodes = self.n_obj_cats + self.n_out_classes

        self.b_bce = config["params"]["use_bce"]

        self.initialise_mode()

        assert self.net_params["class_feats"] in [
            "zeros",
            "zero_padding",
            "transform",
        ]

        self.gip_feat_size = self.net_params["node_feat_size"] - 2

        if self.net_params["class_feats"] == "transform":
            self.resnet_transform_layer()

        # Initialise the graph convolutional network
        self.gnn = GRM(
            # grm_hidden_channel=self.node_feature_size + self.one_hot,
            grm_hidden_channel=self.net_params["ggnn_hidden_channel"],
            grm_output_channel=self.net_params["ggnn_output_channel"],
            time_step=self.net_params["time_step"],
            n_out_class=self.net_params["num_out_classes"],
            n_obj_cls=self.net_params["num_obj_cat"],
            attention=self.net_params["use_attention"],
            b_bce=self.b_bce,
        )

        print("\nGRM parameters: ")
        print_model_parameters(self.gnn)

        self.initialise_classifier(config["params"]["use_bce"])

        set_seed(config["params"]["seed"])
        self.rng = np.random.default_rng()

    def get_model_name(self):
        return self.model_name

    def resnet_transform_layer(self):
        """Transform and align the 2048 ResNet-101 features to the VGG-16 features.

        GIP extracts a feature vector from the second last layer of the ResNet-101 backbone.
        This feature vector has dimensionality of 2,048. GIP also extracts feature vectors,
        whose dimensionality is 4,096, from the region of object of interests as defined by
        the bounding boxes predicted by an object detector (YOLO or Mask R-CNN). By aligning
        the features to the same and largest dimensionality, GIP can initialise the nodes in
        the graph with feature vectors of the same dimensionality for the subsequent processing
        via the Graph Neural Network.
        """
        self.resnet_feat_size = 2048

        self.fc_resnet = nn.Linear(
            self.resnet_feat_size,  # 2,048
            self.gip_feat_size,  # 4,096 (VGG-16)
        )

        # Initialise layer
        self.fc_resnet.weight.data.normal_(0, 0.01)
        self.fc_resnet.bias.data.zero_()

    def initialise_mode(self):
        """ """
        mode = self.net_params["mode"]

        self.b_self_edges = False

        assert self.net_params["use_flag"]
        if self.net_params["use_flag"]:
            self.one_hot = 2
        else:
            self.one_hot = 0

        str1 = "bipartite, "

        ### Bipartite graphs (inspired by GIP)
        if mode in [a for a in range(2, 5)]:
            if mode == 2:
                self.b_undirected = True
                self.b_unweighted = True

            if mode == 3:
                self.b_undirected = True
                self.b_unweighted = False

            if mode == 4:
                self.b_undirected = False
                self.b_unweighted = True

            if mode == 5:
                self.b_undirected = False
                self.b_unweighted = False

            if self.b_undirected:
                str2 = "undirected, "
            else:
                str2 = "directed, "
            if self.b_unweighted:
                str3 = "unweighted, "
            else:
                str3 = "weighted"

            print("Graph: " + str1 + str2 + str3)

    def initialise_classifier(self, use_bce):
        """ """

        if use_bce:
            assert self.n_out_classes == 2
            fc_out = nn.Linear(self.gip_feat_size, 1)

        else:
            assert self.n_out_classes >= 2
            # fc_out = nn.Linear(self.node_feature_size, self.n_out_classes)
            fc_out = nn.Linear(self.gip_feat_size, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                (self.net_params["num_obj_cat"] + 1)
                * self.net_params["ggnn_output_channel"],
                self.gip_feat_size,
            ),
            nn.ReLU(True),
            nn.Dropout(),
            fc_out,
        )

        # Initialise the weights of the final classifier using Xavier's uniform.
        for m in self.classifier.modules():
            cnt = 0
            if isinstance(m, nn.Linear):
                if cnt == 0:
                    m.weight.data.normal_(0, 0.001)
                else:
                    m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                cnt += 1

        # Print classifier parameters to screen
        print("\nClassifier parameters: ")
        print_model_parameters(self.classifier)

    def initialise_prior_graph(self, prior_graph):
        # This is to avoid computing multiple times
        self.prior_graph = prior_graph
        (
            self.adj_mat_obj_occ,
            self.adj_mat_bipartite,
        ) = self.prior_graph.split_adjacency_matrix()

        if not check_if_symmetric(self.adj_mat_bipartite):
            self.gnn.set_adjacency_matrix(
                self.adj_mat_bipartite + self.adj_mat_bipartite.transpose()
            )
        else:
            self.gnn.set_adjacency_matrix(self.adj_mat_bipartite)

    def get_filename(
        self, config, model_name, extension=".csv", prefix="", suffix=""
    ):
        """Create a filename based on the fold ID and model name.

        The function returns the filename with any additional prefix appended,
        and the extension based the argument passed (.csv as default).
        """
        if config["params"]["training_mode"] == "crossval":
            filename = "{:s}-{:d}".format(
                model_name, config["params"]["fold_id"]
            )

        if config["params"]["training_mode"] == "final":
            filename = "{:s}-final".format(model_name)

        if config["params"]["training_mode"] == "original":
            filename = "{:s}-original".format(model_name)

        filename = prefix + filename + suffix + extension

        return filename

    def set_batch_size(self, batch_size):
        """Dummy function based on other networks"""
        # batch_sz = batch_size
        self.gnn.set_batch_size(batch_size)

    def forward(self, node_features):
        """ """
        num_imgs = node_features.shape[0]

        if self.net_params["class_feats"] == "zeros":
            node_features[:, : self.n_out_classes, 2:] = 0

        elif self.net_params["class_feats"] == "transform":
            tmp_feats = node_features[
                :, : self.n_out_classes, 2 : self.resnet_feat_size + 2
            ]
            tmp_feats_updated = self.fc_resnet(tmp_feats)
            node_features[:, : self.n_out_classes, 2:] = tmp_feats_updated

        model_input = node_features.view(num_imgs, -1)
        # print(model_input.shape)
        grm_feature = self.gnn(model_input)

        nodes_unnormalized_scores = self.classifier(grm_feature).view(
            num_imgs, -1
        )

        if self.b_bce:
            nodes_unnormalized_scores = nodes_unnormalized_scores.squeeze()

        nodes_unnormalized_scores.float()

        return nodes_unnormalized_scores
