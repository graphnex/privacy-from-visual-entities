#!/usr/bin/env python
#
# Model and forward pass for the graph-agnostic baseline with MLPs.
#
# Partially taken from:
# https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/superpixels_graph_classification/mlp_net.py
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/07
# Modified Date: 2024/04/28
#
# ----------------------------------------------------------------------------

import sys

import numpy as np

np.set_printoptions(threshold=sys.maxsize, precision=2)

# PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

# PyTorch Geometric libraries
from torch_geometric.nn import global_mean_pool, global_add_pool

from srcs.nets.MLPReadout import MLPReadout

from srcs.utils import (
    set_seed,
    device,
    print_model_parameters,
)


#############################################################################


class GraphAgnosticMLP(nn.Module):
    """Graph-agnostic baseline with multi-layer perceptrons.

    Preparation and forward pass for the multi-layer perceptron.

    Simple graph-agnostic baseline that parallelly applies an MLP on each
    node’s feature vector, independent of other nodes. For graph-level
    classification, the node features are pooled together to form a global
    feature that is then passed to a classifer (3-layers MLP, or MLP Readout).

    Partially taken from https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/superpixels_graph_classification/mlp_net.py
    """

    def __init__(
        self,
        config,
    ):
        """Constructor of the class"""
        super(GraphAgnosticMLP, self).__init__()

        set_seed(config["params"]["seed"])
        self.rng = np.random.default_rng()

        self.load_config(config)

        self.initialise_model()
        self.initialise_embeddings()

        # Print model summary and number of parameters
        self.print_model_summary()

    def load_config(self, config):
        """ """
        self.config = config

        self.model_name = config["model_name"]

        net_params = config["net_params"]

        self.node_feat_size = net_params["node_feat_size"]

        self.n_classes = net_params["num_out_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]

        self.gated = net_params["gated"]
        self.readout_mode = net_params["readout"]

        self.max_num_roi = net_params["max_num_roi"]

        self.n_graph_nodes = net_params["n_graph_nodes"]

        n_obj_cat = net_params["num_obj_cat"]
        n_scene_cat = net_params["num_scene_cat"]

        b_class_nodes = net_params["use_class_nodes"]
        if b_class_nodes:
            assert (
                self.n_graph_nodes == n_scene_cat + n_obj_cat + self.n_classes
            )
        else:
            assert self.n_graph_nodes == n_scene_cat + n_obj_cat

        # self.dropout = nn.Dropout(p=dropout_prob)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.b_use_embedding = False

    def initialise_model(self):
        """ """
        net_params = self.config["net_params"]

        in_dim = net_params["node_feat_size"]
        hidden_dim = net_params["hidden_dim"]
        dropout = net_params["dropout"]
        n_layers = net_params["num_layers"]
        b_batch_norm = net_params["use_bn"]

        if "use_embedding" in net_params:
            self.b_use_embedding = net_params["use_embedding"]

            if self.b_use_embedding:
                try:
                    self.embedding_size = net_params["embedding_size"]
                except:
                    raise ValueError(
                        "embdedding_size parameter missing in the configuration file!"
                    )

                print("Using projection layer to higher-dimension")

                if self.node_feat_size == 2:
                    # The next 2 embedding layers are independent for projecting
                    # cardinality and confidence separetely, when both are used
                    self.embedding1 = nn.Linear(
                        # net_params["node_feat_size"],
                        1,
                        self.embedding_size,
                        bias=True,
                    )
                    self.embedding2 = nn.Linear(
                        # net_params["node_feat_size"],
                        1,
                        self.embedding_size,
                        bias=True,
                    )

                    in_dim = self.embedding_size * self.node_feat_size
                else:
                    set_seed(self.config["params"]["seed"])

                    self.embedding = nn.Linear(
                        # net_params["node_feat_size"],
                        1,
                        self.embedding_size,
                        bias=True,
                    )

                    in_dim = self.embedding_size

        # Hidden layers
        feat_list = [in_dim] + [hidden_dim] * n_layers

        feat_mlp_modules = []

        for l_idx in range(n_layers):
            # Add batch normalisation layer (if activated)
            # - currently not working
            if b_batch_norm:
                feat_mlp_modules.append(
                    nn.Linear(
                        feat_list[l_idx], feat_list[l_idx + 1], bias=True
                    )
                )
                feat_mlp_modules.append(nn.BatchNorm1d(self.n_graph_nodes))
            else:
                feat_mlp_modules.append(
                    nn.Linear(
                        feat_list[l_idx], feat_list[l_idx + 1], bias=True
                    )
                )

            feat_mlp_modules.append(nn.ReLU())
            feat_mlp_modules.append(nn.Dropout(dropout))

        self.feat_mlp = nn.Sequential(*feat_mlp_modules)

        if self.gated:
            self.gates = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # MLP classifier of 3 fully connected layers with ReLu functions
        self.readout_mlp = MLPReadout(
            feat_list[-1],
            self.n_classes,
            seed_val=self.config["params"]["seed"],
        )

    def initialise_embeddings(self):
        """Initialise the weights of the embedding layers using Xavier's uniform."""
        for m in self.feat_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        if self.b_use_embedding:
            if self.node_feat_size == 2:
                for m in self.embedding1.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                        if m.bias is not None:
                            m.bias.data.zero_()

                for m in self.embedding2.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                        if m.bias is not None:
                            m.bias.data.zero_()

            elif self.node_feat_size == 1:
                for m in self.embedding.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                        if m.bias is not None:
                            m.bias.data.zero_()

    def print_model_summary(self):
        """ """
        print(summary(self.feat_mlp, input=(1, 1, self.n_graph_nodes)))

        print("Classifier")
        print(summary(self.readout_mlp, input=(self.n_graph_nodes)))

    def print_number_parameters(self):
        """ """
        if self.node_feat_size == 2:
            print("\nEmbeddings: ")
            print_model_parameters(self.embedding1)
            print_model_parameters(self.embedding2)

        elif self.node_feat_size == 1:
            print("\nEmbedding: ")
            print_model_parameters(self.embedding)

        print("\nDropout: ")
        print_model_parameters(self.in_feat_dropout)

        # Print classifier parameters to screen
        print("\nGraph-agnostic MLP parameters: ")
        print_model_parameters(self.feat_mlp)

        print("\nMLP Readout parameters: ")
        print_model_parameters(self.readout_mlp)

    def print_embedding_param(self):
        """ """
        print("Parameters of embedding 1")

        print(torch.t(self.embedding1.weight.data))
        print(self.embedding1.bias.data)

    def get_model_name(self):
        return self.model_name

    def set_batch_size(self, batch_size):
        """Dummy function based on other networks"""
        batch_sz = batch_size

    def forward(self, x):
        """Forward pass.

        Arguments:
            - x: node features

        Return:
            - outputs:
        """
        batch_sz, n_feats, _ = x.shape
        batch = (
            torch.Tensor(np.array(range(batch_sz)))
            .unsqueeze(1)
            .repeat(1, n_feats)
            .view(-1)
        )
        batch = batch.to(dtype=torch.int64).to(device)

        if self.b_use_embedding:
            if self.node_feat_size == 2:
                # embeddings = self.embedding(torch.unsqueeze(x.to(device),3))
                # embeddings = embeddings.view(x.shape[0],x.shape[1],-1)
                embeddings1 = self.embedding1(torch.unsqueeze((x[:, :, 0]), 2))
                embeddings2 = self.embedding2(torch.unsqueeze((x[:, :, 1]), 2))
                embeddings = torch.cat((embeddings1, embeddings2), 2)
            else:
                embeddings = self.embedding(x)
            x_d = self.in_feat_dropout(F.relu(embeddings))  # Initial dropout
        else:
            x_d = self.in_feat_dropout(x.to(device))  # Initial dropout

        x_mlp = self.feat_mlp(x_d)  # Run through the individual MLPs

        # Pooling
        if self.gated:
            # Add sum of nodes (replace the DGL-based one)
            x_mlp = torch.sigmoid(self.gates(x_mlp)) * x_mlp
            x_mlp_global = global_add_pool(
                x_mlp.view(batch_sz * n_feats, -1), batch
            )
        else:
            if self.readout_mode == "mean":
                x_mlp_global = global_mean_pool(
                    x_mlp.view(batch_sz * n_feats, -1), batch
                )
            elif self.readout_mode == "sum":
                x_mlp_global = global_add_pool(
                    x_mlp.view(batch_sz * n_feats, -1), batch
                )

        # Final classifier (based on an MLP)
        logits = self.readout_mlp(x_mlp_global)

        return logits
