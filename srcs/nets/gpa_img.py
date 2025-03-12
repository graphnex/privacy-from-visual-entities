#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/08/30
# Modified Date: 2024/08/14
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

import copy
import json

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

from srcs.nets.s2p import SceneToPrivacyClassifier as S2P

# Utilities
from srcs.nets.grm import GraphReasoningModel as GRM
from srcs.utils import (
    print_model_parameters,
    set_seed,
    device,
    check_if_symmetric,
)

from pdb import set_trace as bp  # This is only for debugging


#############################################################################
#
class GraphPrivacyAdvisor(nn.Module):
    def __init__(self, config):
        super(GraphPrivacyAdvisor, self).__init__()

        self.model_name = config["model_name"]

        self.config = config
        self.net_params = config["net_params"]

        assert self.net_params["s2p_mode"] in [
            "rand",
            "pretrained",
            "finetune",
            "zeros",
        ]

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
            grm_hidden_channel=self.node_feature_size + self.one_hot,
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

        if self.net_params["s2p_mode"] in ["pretrained", "finetune"]:
            self.load_scene_to_privacy_classifier(config)

        set_seed(config["params"]["seed"])
        self.rng = np.random.default_rng()

    def get_model_name(self):
        return self.model_name

    def initialise_mode(self):
        """ """
        mode = self.net_params["mode"]

        self.b_self_edges = False

        # adjacency_filename = args.adjacency_filename
        # print(adjacency_filename)

        # mode:
        #   - 0: IEEE BIG-MM, binary flag
        #   - 1: Corrected loading of adjacency matrix (leading to no training)
        #   - 2: Weighted, undirected bipartite graph (as GIP)
        if mode == 0 or mode == 1:  # IEEE BIG-MM
            assert self.net_params["use_flag"]
            self.one_hot = 1

        # if mode == 1:
        #     self.one_hot = 2

        ### Bipartite graphs (inspired by GIP)
        if mode in [a for a in range(2, 5)]:
            if self.net_params["use_flag"]:
                self.one_hot = 1
            else:
                self.one_hot = 0

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
        # self.node_feature_size += self.one_hot

        # This should be with the prior graph and not here
        # if mode in [2, 3, 4, 5]:
        #     adjacency_filename += "_bipartite.csv"

        #     self.load_adjacency_matrix(
        #         adjacency_filename,
        #         "bipartite_ggnn",
        #         self_edges=self.b_self_edges,
        #     )
        # elif mode < 2:
        #     if self.corrected_adj_mat:
        #         adjacency_filename += ".json"

        #         self.load_adjacency_matrix(
        #             adjacency_filename,
        #             "ggnn_triangular",
        #             self_edges=self.b_self_edges,
        #         )
        #     else:
        #         adj_fn_list = adjacency_filename.split("prior_graph_", 1)
        #         adjacency_filename = (
        #                 adj_fn_list[0] + "prior_graph_star_co_occ_binary.npy"
        #         )

        #         self.load_adjacency_matrix(
        #             adjacency_filename,
        #             "ggnn_bigmm",
        #             self_edges=self.b_self_edges,
        #         )

    def initialise_classifier(self, use_bce):
        """ """

        # if use_bce:
        #     assert self.n_out_classes == 2
        #     fc_out = nn.Linear(self.net_params["num_out_classes"], 1)

        # else:
        #     assert self.n_out_classes >= 2
        #     # fc_out = nn.Linear(self.node_feature_size, self.n_out_classes)
        #     fc_out = nn.Linear(self.net_params["node_feat_size"], 1)
        #     # fc_out = nn.Linear(self.net_params["num_out_classes"], 1)

        if self.net_params["reshape_layer"]:
            reshape_input = nn.Linear(
                (self.net_params["num_obj_cat"] + 1)
                * self.net_params["ggnn_output_channel"],
                (self.net_params["num_obj_cat"] + 1),
            )

            self.classifier = nn.Sequential(
                reshape_input,
                nn.Dropout(),
                nn.Linear(
                    self.net_params["num_obj_cat"] + 1,
                    self.net_params["num_out_classes"],
                ),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(self.net_params["num_out_classes"], 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(
                    (self.net_params["num_obj_cat"] + 1)
                    * self.net_params["ggnn_output_channel"],
                    self.net_params["num_out_classes"],
                ),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(self.net_params["num_out_classes"], 1),
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
        mode = self.net_params["mode"]

        assert mode >= 0 and mode < 6

        self.prior_graph = prior_graph
        (
            self.adj_mat_obj_occ,
            self.adj_mat_bipartite,
        ) = self.prior_graph.split_adjacency_matrix()

        if mode in [2, 3, 4, 5]:
            self.gnn.set_adjacency_matrix(self.adj_mat_bipartite)
        else:
            # Binarise the matrix as defined by GPA paper (IEEE BigMM 2022)
            self.adj_mat_obj_occ[self.adj_mat_obj_occ > 0] = 1

            # This matrix should be symmetric as it is unweighted and undirected
            assert check_if_symmetric(self.adj_mat_obj_occ)
            # if not check_if_symmetric(self.adj_mat_obj_occ):
            #     bp()

            if mode == 1:
                # Correct way to initialise the matrix
                self.gnn.set_adjacency_matrix(self.adj_mat_obj_occ)
            else:
                # Model the same adjacency matrix as loaded by GPA in IEEE BigMM 2022
                # Wrong way to initialise a matrix
                n_rows, n_cols = self.adj_mat_obj_occ.shape
                n_out_cls = self.n_out_classes

                adj_mat_tmp = np.zeros((n_rows, n_cols))
                adj_mat_tmp[:-n_out_cls, n_out_cls:] = self.adj_mat_obj_occ[
                    n_out_cls:, n_out_cls:
                ]

                self.gnn.set_adjacency_matrix(adj_mat_tmp)

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

    def load_scene_to_privacy_classifier(self, config):
        """
        Load the trained S2P (scene-to-privacy) classifier based on the
        ResNet-50 pre-trained oon Places365 for scene categorisation. S2P maps
        the 365 logits of the scene classifier to the defined number of privacy
        classes with a fully connected layer trained from scratch on the
        training set(s) of the image privacy dataset.
        """
        s2p_config = dict2 = copy.deepcopy(config)

        s2p_config["net_params"]["num_scene_cat"] = 365
        s2p_config["net_params"]["backbone_arch"] = "resnet50"
        s2p_config["model_name"] = "s2p"

        self.s2p_net = S2P(s2p_config)

        if config["params"]["training_mode"] == "final":
            prefix_net = "last_acc_"
        else:
            prefix_net = "best_acc_"

        self.checkpoint_dir = os.path.join(
            os.path.join(
                config["paths"]["root_dir"],  # directory of the repository,
                "trained_models",
                # use_case_dir,
                config["dataset"].lower(),
            ),  # directory where the model is saved,
            "{:d}-class".format(self.n_out_classes),
            self.s2p_net.get_model_name(),
        )

        fullpathname = os.path.join(
            self.checkpoint_dir,
            self.get_filename(
                config,
                self.s2p_net.get_model_name(),
                ".pth",  # extension of the models
                prefix=prefix_net,
            ),
        )

        print("Loading S2P ... ({:s})".format(fullpathname))
        checkpoint = torch.load(fullpathname)

        self.s2p_net.load_state_dict(checkpoint["net"])

        if self.net_params["s2p_mode"] == "pretrained":
            for param in self.s2p_net.parameters():
                param.requires_grad = False

            self.s2p_net = self.s2p_net.to(device)
            self.s2p_net.eval()

        elif self.net_params["s2p_mode"] == "finetune":
            self.s2p_net.initialise_fc_layer()
            self.s2p_net = self.s2p_net.to(device)

        print("\nS2P parameters: ")
        print_model_parameters(self.s2p_net)

    def set_batch_size(self, batch_size):
        """Dummy function based on other networks"""
        # batch_sz = batch_size
        self.gnn.set_batch_size(batch_size)

    def load_node_features(self, img_fn):
        """ """
        assert self.node_feature_size > 0

        b_use_card = self.net_params["use_card"]
        b_use_conf = self.net_params["use_conf"]
        b_use_cls_nodes = self.net_params["use_class_nodes"]

        if b_use_card != b_use_conf:
            assert self.node_feature_size == 1
        elif (b_use_card == b_use_conf) and (b_use_card):
            assert self.node_feature_size == 2

        node_feats_var = Variable(
            torch.zeros(self.n_graph_nodes, self.node_feature_size),
            requires_grad=False,
        )

        node_feats = json.load(open(img_fn))

        for node in node_feats:
            n_id = node["node_id"]

            if not b_use_cls_nodes:
                if n_id < self.n_out_classes:
                    continue
                else:
                    n_id -= self.n_out_classes

            n_feat = node["node_feature"]
            # assert(len(n_feat) > 0 and len(n_feat) < 3)

            if self.node_feature_size == 1:
                if len(n_feat) > 1:
                    if b_use_card:
                        n_feat = [n_feat[0]]
                    elif b_use_conf:
                        n_feat = [n_feat[1]]

            elif self.node_feature_size == 2:
                n_feat = n_feat[:2]

            node_feats_var[n_id, :] = torch.Tensor(n_feat)

        # Thresholding - hardcoded value (not used for the moment)
        # if (self.node_feat_size == 1) and (self.b_use_conf):
        #     node_feats_var = (node_feats_var > 0.5).type(node_feats_var.type())

        return node_feats_var

    def prepare_node_features(self, full_im, filename):
        """ """
        num_imgs = full_im.size()[0]

        # initialise contextual matrix
        contextual = Variable(
            torch.zeros(
                num_imgs,
                self.n_graph_nodes,
                self.node_feature_size + self.one_hot,
            ),
            requires_grad=False,
        ).to(device)

        if self.one_hot == 0:
            feat_start_idx = 0

        elif self.one_hot == 1:
            # contextual[:, self.n_out_classes :, 0] = 0.0  # size: ([bs,2])
            contextual[:, : self.n_out_classes, 0] = 1.0  # size: ([bs,2])

            feat_start_idx = 1

        elif self.one_hot == 2:
            contextual[:, : self.n_out_classes, 0] = 1.0  # size: ([bs,2])
            contextual[:, self.n_out_classes :, 1] = 1.0  # size: ([bs,2])

            feat_start_idx = 2

        for b in range(num_imgs):
            contextual[b, :, feat_start_idx] = self.load_node_features(
                filename[b]
            ).squeeze()

        if self.net_params["s2p_mode"] == "rand":
            tmp_arr = self.rng.random(self.n_out_classes)
            # tmp_arr /= tmp_arr.sum()[:,None]
            contextual[
                :, : self.n_out_classes, feat_start_idx
            ] = torch.from_numpy(tmp_arr)
        elif self.net_params["s2p_mode"] in ["finetune", "pretrained"]:
            out_logits, scene_logits = self.s2p_net(full_im)
            contextual[:, : self.n_out_classes, feat_start_idx] = out_logits
        elif self.net_params["s2p_mode"] == "zeros":
            contextual[:, : self.n_out_classes, feat_start_idx] = 0.0

        # contextual[:, : self.n_out_classes, feat_start_idx] = 1.0

        # contextual[:, : self.n_out_classes, scene_idx] = 1.0
        # contextual[:, : self.n_out_classes, scene_idx+1] = torch.sigmoid(out_logits)
        # contextual[:, : self.n_out_classes, scene_idx+1] = 1. - torch.sigmoid(out_logits)

        return contextual

    # def forward(self, full_im, categories, card, scene):
    def forward(self, img, filename):
        """ """
        num_imgs = img.size()[0]

        node_features = self.prepare_node_features(img, filename)

        model_input = node_features.view(num_imgs, -1)
        # print(model_input.shape)
        grm_feature = self.gnn(model_input)

        # if self.mode in [a for a in range(0,6)]:
        nodes_unnormalized_scores = self.classifier(grm_feature).view(
            num_imgs, -1
        )

        if self.b_bce:
            nodes_unnormalized_scores = nodes_unnormalized_scores.squeeze()

        return nodes_unnormalized_scores.float(), None
