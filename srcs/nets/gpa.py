#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/08/30
# Modified Date: 2023/09/07
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
from srcs.utils import print_model_parameters, set_seed


#############################################################################
#
class GraphPrivacyAdvisor(nn.Module):
    def __init__(self, config):
        super(GraphPrivacyAdvisor, self).__init__()

        self.model_name = config["model_name"]

        self.net_params = config["net_params"]

        self.node_feature_size = self.net_params["node_feat_size"]

        # Number of output classes (privacy levels)
        self.n_out_classes = self.net_params["num_out_classes"]

        # Number of object categories (for COCO=80, no background)
        self.n_obj_cats = self.net_params["num_obj_cat"]

        self.n_graph_nodes = self.n_obj_cats + self.n_out_classes

        self.b_bce = config["params"]["use_bce"]

        self.initialise_mode()

        # Initialise the graph convolutional network
        self.gnn = GRM(
            # grm_hidden_channel=self.net_params["ggnn_hidden_channel"],
            grm_hidden_channel=self.node_feature_size,
            grm_output_channel=self.net_params["ggnn_output_channel"],
            time_step=self.net_params["time_step"],
            n_out_class=self.net_params["num_out_classes"],
            n_obj_cls=self.net_params["num_obj_cat"],
            attention=self.net_params["use_attention"],
            b_bce=self.b_bce,
        )

        print("\nGRM parameters: ")
        print_model_parameters(self.gnn)

        self.reshape_input = nn.Linear(
            (self.net_params["num_obj_cat"] + 1)
            * self.net_params["num_out_classes"],
            (self.net_params["num_obj_cat"] + 1),
        )

        self.initialise_classifier(config["params"]["use_bce"])

        set_seed(config["params"]["seed"])
        self.rng = np.random.default_rng()

    def get_model_name(self):
        return self.model_name

    def initialise_mode(self, args):
        """ """
        mode = self.net_params["mode"]

        self.b_self_edges = False

        # adjacency_filename = args.adjacency_filename
        # print(adjacency_filename)

        if mode == 0:  # IEEE BIG-MM
            # self.gnn_type = "grm"
            self.one_hot = 1
            self.corrected_adj_mat = False
            self.isolated_nodes = True

        if mode == 1:
            # self.gnn_type = "grm"
            self.one_hot = 2
            self.corrected_adj_mat = True
            self.isolated_nodes = True

        ### Bipartite graphs (inspired by GIP)
        if mode in [a for a in range(2, 5)]:
            self.one_hot = 0
            self.corrected_adj_mat = True
            self.isolated_nodes = True

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

            str1 = "bipartite, "
            if self.b_undirected:
                str2 = "undirected, "
            else:
                str2 = "directed, "
            if self.b_unweighted:
                str3 = "unweighted, "
            else:
                str3 = "weighted"

            print("Graph: " + str1 + str2 + str3)

        # Updated the node feature size to take into account the 1-hot vector
        self.node_feature_size += self.one_hot

        if self.gnn_type == "grm":
            if mode in [2, 3, 4, 5]:
                adjacency_filename += "_bipartite.csv"

                self.load_adjacency_matrix(
                    adjacency_filename,
                    "bipartite_ggnn",
                    self_edges=self.b_self_edges,
                )
            elif mode < 2:
                if self.corrected_adj_mat:
                    adjacency_filename += ".json"

                    self.load_adjacency_matrix(
                        adjacency_filename,
                        "ggnn_triangular",
                        self_edges=self.b_self_edges,
                    )
                else:
                    adj_fn_list = adjacency_filename.split("prior_graph_", 1)
                    adjacency_filename = (
                        adj_fn_list[0] + "prior_graph_star_co_occ_binary.npy"
                    )

                    self.load_adjacency_matrix(
                        adjacency_filename,
                        "ggnn_bigmm",
                        self_edges=self.b_self_edges,
                    )

    # def initialise_classifier(self, use_bce):
    #     """ """

    #     if use_bce:
    #         assert self.n_out_classes == 2
    #         fc_out = nn.Linear(self.net_params["node_feat_size"], 1)

    #     else:
    #         assert self.n_out_classes >= 2
    #         # fc_out = nn.Linear(self.node_feature_size, self.n_out_classes)
    #         fc_out = nn.Linear(self.net_params["node_feat_size"], 1)

    #     self.classifier = nn.Sequential(
    #         nn.Dropout(),
    #         nn.Linear(
    #             (self.net_params["num_obj_cat"] + 1) * self.net_params["ggnn_output_channel"],
    #             self.net_params["node_feat_size"],
    #         ),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         fc_out,
    #     )

    #     # Initialise the weights of the final classifier using Xavier's uniform.
    #     for m in self.classifier.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight.data)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()

    #     # Print classifier parameters to screen
    #     print("\nClassifier parameters: ")
    #     print_model_parameters(self.classifier)

    def initialise_classifier(self, use_bce):
        """ """

        if use_bce:
            assert self.n_out_classes == 2
            fc_out = nn.Linear(self.net_params["num_out_classes"], 1)

        else:
            assert self.n_out_classes >= 2
            # fc_out = nn.Linear(self.node_feature_size, self.n_out_classes)
            fc_out = nn.Linear(self.net_params["num_out_classes"], 1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                self.net_params["num_obj_cat"] + 1,
                self.net_params["num_out_classes"],
            ),
            nn.ReLU(True),
            nn.Dropout(),
            fc_out,
        )

        # Initialise the weights of the final classifier using Xavier's uniform.
        for m in self.classifier.modules():
            cnt = 0
            # if isinstance(m, nn.Linear):
            #     nn.init.xavier_uniform_(m.weight.data)
            #     if m.bias is not None:
            #         m.bias.data.zero_()
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

        self.adj_mat_obj_occ[self.adj_mat_obj_occ > 0] = 1

        self.gnn.set_adjacency_matrix(self.adj_mat_obj_occ)

    def set_batch_size(self, batch_size):
        """Dummy function based on other networks"""
        # batch_sz = batch_size
        self.gnn.set_batch_size(batch_size)

    def forward(self, node_features):
        """ """
        num_imgs = node_features.shape[0]

        model_input = node_features.view(num_imgs, -1)
        # print(model_input.shape)
        grm_feature = self.gnn(model_input)

        # if self.mode in [a for a in range(0,6)]:
        grm_feature = self.reshape_input(grm_feature)

        nodes_unnormalized_scores = self.classifier(grm_feature).view(
            num_imgs, -1
        )

        if self.b_bce:
            nodes_unnormalized_scores = nodes_unnormalized_scores.squeeze()

        nodes_unnormalized_scores.float()

        return nodes_unnormalized_scores
