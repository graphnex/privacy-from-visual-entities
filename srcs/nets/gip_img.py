#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/08/30
# Modified Date: 2024/09/04
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

import json
import math

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

from torchsummary import summary


from srcs.nets.grm import GraphReasoningModel as GRM
from srcs.nets.resnet_v1 import ResNet, Bottleneck
from srcs.nets.vgg_v1 import vgg16_rois_v1

# Utilities
from srcs.utils import (
    set_seed,
    device,
    check_if_symmetric,
    print_model_parameters,
)


#############################################################################
# Code taken from https://github.com/guang-yanng/Image_Privacy/blob/master/networks/full_image_score.py
#
class full_image_score(nn.Module):
    def __init__(self, num_classes=2, b_pretrained=True, b_fixed=False):
        """Constructor to initialise the model to extract deep features from the whole image.

        Arguments:
            - num_classes: number of output classes depending on the task. Default: 2.
            - b_pretrained: boolean variable to load a ResNet model pretrained on ImageNet. Default: True.
            - b_fixed: boolean variable to keep the parameters of the loaded ResNet model fixed during training of the full pipeline. Default: False.
        """
        super(full_image_score, self).__init__()

        self.resnet101 = ResNet(Bottleneck, [3, 4, 23, 3])

        if b_pretrained:
            pretrained_model = model_zoo.load_url(
                "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
            )

            del pretrained_model["fc.weight"]
            del pretrained_model["fc.bias"]

            self.resnet101.load_state_dict(pretrained_model)

            if b_fixed:
                for param in self.resnet101.parameters():
                    param.requires_grad = False

                self.resnet101.to(device).eval()

        # If num_classes = 2, the total number of trainable parameters is
        # 8,400,898
        self.fc6 = nn.Linear(2048, 4096)
        self.fc7 = nn.Linear(4096, num_classes)
        self.ReLU = nn.ReLU(False)
        self.Dropout = nn.Dropout()

        self.initialize_weights()

    # x1 = union, x2 = object1, x3 = object2, x4 = bbox geometric info
    def forward(self, input_image):
        rn_feats = self.resnet101(input_image)

        x = self.Dropout(rn_feats)
        fc6 = self.fc6(x)
        x = self.ReLU(fc6)
        x = self.Dropout(x)
        x = self.fc7(x)

        return x, fc6, rn_feats

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


#############################################################################
#
class GraphImagePrivacy(nn.Module):
    def __init__(self, config):
        super(GraphImagePrivacy, self).__init__()

        set_seed(config["params"]["seed"])
        self.rng = np.random.default_rng()

        self.load_config(config)

        self.initialise_model()
        self.init_weights()

        # Print model summary and number of parameters
        self.print_model_summary()
        self.print_model_parameters()

    def get_model_name(self):
        return self.model_name

    def load_config(self, config):
        """ """
        self.config = config

        self.model_name = config["model_name"]
        self.net_params = config["net_params"]
        self.b_bce = config["params"]["use_bce"]

        self.node_feature_size = self.net_params["node_feat_size"]

        # Number of output classes (privacy levels)
        self.n_out_classes = self.net_params["num_out_classes"]

        # Number of object categories (for COCO=80, no background)
        self.n_obj_cats = self.net_params["num_obj_cat"]

        self.n_graph_nodes = self.n_obj_cats + self.n_out_classes

        self.b_vgg_fixed = self.net_params["vgg_fixed"]
        self.b_rn_fixed = self.net_params["rn_fixed"]
        self.b_self_edges = self.net_params["self_loop"]

        print("Is VGG-16 fixed? {:b}".format(self.b_vgg_fixed))
        print("Is ResNet-101 fixed? {:b}".format(self.b_rn_fixed))

        # assert(self.net_params["use_flag"])
        if self.net_params["use_flag"]:
            self.one_hot = 2
        else:
            self.one_hot = 0

        assert self.net_params["class_feats"] in [
            "zeros",
            "zero_padding",
            "transform",
        ]

        print("Mode: {:s}".format(self.net_params["class_feats"]))

        if self.net_params["class_feats"] == "zeros":
            assert self.b_rn_fixed

        ## Prior graph information (fixed for GIP)
        self.b_undirected = True

        if self.b_undirected:
            str2 = "undirected, "
        else:
            str2 = "directed, "

        self.b_unweighted = False

        if self.b_unweighted:
            str3 = "unweighted, "
        else:
            str3 = "weighted"

        str1 = "bipartite, "

        print("Graph: " + str1 + str2 + str3)

    def initialise_model(self):
        """ """
        # Load the pre-trained VGG-16 for detected ROIs to extract their deep features
        self.full_im_net = vgg16_rois_v1(pretrained=True)

        if self.b_vgg_fixed:
            for param in self.full_im_net.parameters():
                param.requires_grad = False

            self.full_im_net = self.full_im_net.to(device)
            self.full_im_net.eval()

        if self.net_params["class_feats"] in ["transform", "padding"]:
            self.fis = full_image_score(
                num_classes=self.n_out_classes,
                b_pretrained=True,
                b_fixed=self.b_rn_fixed,
            )

        # if self.net_params["class_feats"] == "transform":
        # self.resnet_transform_layer()

        # Initialise the graph convolutional network
        self.gnn = GRM(
            grm_hidden_channel=self.node_feature_size + self.one_hot,
            # grm_hidden_channel=self.net_params["ggnn_hidden_channel"],
            grm_output_channel=self.net_params["ggnn_output_channel"],
            time_step=self.net_params["time_step"],
            n_out_class=self.net_params["num_out_classes"],
            n_obj_cls=self.net_params["num_obj_cat"],
            attention=self.net_params["use_attention"],
            b_bce=self.b_bce,
        )

        if self.b_bce:
            assert self.n_out_classes == 2
            fc_out = nn.Linear(self.node_feature_size, 1)

        else:
            assert self.n_out_classes >= 2
            # fc_out = nn.Linear(self.node_feature_size, self.n_out_classes)
            fc_out = nn.Linear(self.node_feature_size, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                (self.net_params["num_obj_cat"] + 1)
                * self.net_params["ggnn_output_channel"],
                self.node_feature_size,
            ),
            nn.ReLU(True),
            nn.Dropout(),
            fc_out,
        )

    def init_weights(self):
        """ """
        for m in self.classifier.modules():
            cnt = 0
            if isinstance(m, nn.Linear):
                if cnt == 0:
                    m.weight.data.normal_(0, 0.001)
                else:
                    m.weight.data.normal_(0, 0.01)

                m.bias.data.zero_()

                cnt += 1

    def print_model_parameters(self):
        """ """
        print("\n### Model parameters ###")

        if self.net_params["class_feats"] in ["transform", "padding"]:
            print("GIP: ResNet-101")
            print_model_parameters(self.fis)

        print("\nGIP: VGG-16")
        print_model_parameters(self.full_im_net)

        print("\nGIP: Graph Reasoning Module (GGNN + GAT)")
        print_model_parameters(self.gnn)

        print("\nGIP: Classifier")
        print_model_parameters(self.classifier)

    def print_model_summary(self):
        """ """
        img_size = self.net_params["img_size"]

        if self.net_params["class_feats"] in ["transform", "padding"]:
            print("\nGIP: ResNet-101")
            print(summary(self.fis, input=(1, 3, img_size, img_size)))

        print("\nGIP: VGG-16 for ROIs")
        print(
            summary(
                self.full_im_net,
                input=((1, 3, img_size, img_size), (1, 5, 4), (1, 5)),
            )
        )

        print("\nGIP: Graph Reasoning Module (GGNN + GAT)")
        print(summary(self.gnn, input=(1, 82, 4098)))

        print("\nGIP: Classifier")
        print(
            summary(
                self.classifier,
                input=(
                    (self.net_params["num_obj_cat"] + 1)
                    * self.net_params["ggnn_output_channel"]
                ),
            )
        )

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

    def convert_bbox_format(self, bboxes, im_width, im_height):
        """Convert the bounding box format from the YOLOv8 output
        to the format required for VGG-16 with image crops in GIP.

        Format of the input bounding box: [x1, y1, w, h]
            - (x1, y1): top-left corner of the bounding box
            - w: width of the bounding box
            - h: height of the bounding box

        Format of the output bounding box: [x1, y1, x2, y2]
            - (x1, y1): top-left corner of the bounding box
            - (x2, y2): bottom-right corner of the bounding box
        """
        bboxes_converted = []
        for bbox in bboxes:
            x1 = self.net_params["img_size"] / im_width * bbox[0]
            y1 = self.net_params["img_size"] / im_height * bbox[1]
            x2 = self.net_params["img_size"] / im_width * (bbox[0] + bbox[2])
            y2 = self.net_params["img_size"] / im_height * (bbox[1] + bbox[3])

            bbox_new = [x1, y1, x2, y2]
            bboxes_converted.append(bbox_new)

        return bboxes_converted

    def load_rois_categories(self, img_fn, o_im_size, b_sort_objs=False):
        """ """
        assert self.node_feature_size > 0

        num_imgs = len(img_fn)
        max_num_rois = self.net_params["max_num_rois"]

        # Initialise outputs as torch variables
        rois = Variable(
            torch.zeros(num_imgs, max_num_rois, 4),
            requires_grad=False,
        )

        categories = Variable(
            torch.zeros(num_imgs, max_num_rois + 1),
            requires_grad=False,
        )

        confs = Variable(
            torch.zeros(num_imgs, max_num_rois),
            requires_grad=False,
        )

        # Process all detections for each image
        for b in range(num_imgs):
            path_elems = img_fn[b].split("/")

            dataset_name = self.config["dataset"]

            if self.config["dataset"] == "IPD":
                if "VISPR" in img_fn[b]:
                    dataset_name = "VISPR"
                elif "PicAlert" in img_fn[b]:
                    dataset_name = "PicAlert"

            fn = os.path.join(
                self.config["paths"]["root_dir"],
                "resources",
                dataset_name,
                "dets",
                path_elems[-2],
                path_elems[-1].split(".")[0] + ".json",
            )

            assert os.path.exists(os.path.dirname(fn))

            bboxes_objects = json.load(open(fn, "r"))

            n_objs = len(bboxes_objects["categories"])
            # Handle cases of images without any detected object
            if n_objs == 0:
                continue

            if b_sort_objs:
                # Sort confidence and categories // This is new (not part of the original GIP)
                idx = np.argsort(bboxes_objects["confidence"])[::-1]

                tmp0 = [bboxes_objects["bboxes"][x] for x in idx]
                tmp1 = [bboxes_objects["categories"][x] for x in idx]
                tmp2 = [bboxes_objects["confidence"][x] for x in idx]
            else:
                tmp0 = bboxes_objects["bboxes"]
                tmp1 = bboxes_objects["categories"]
                tmp2 = bboxes_objects["confidence"]

            im_width = o_im_size[b, 0]
            im_height = o_im_size[b, 1]
            # print("Image size (Width x Height): ({:d}, {:d})".format(im_width, im_height))
            tmp0 = self.convert_bbox_format(tmp0, im_width, im_height)

            # Store the outputs as tensors
            if n_objs > max_num_rois:
                rois[b, :, :] = torch.Tensor(tmp0[:max_num_rois])
                categories[b, 1:] = torch.IntTensor(tmp1[:max_num_rois])
                confs[b, :] = torch.Tensor(tmp2[:max_num_rois])

                n_objs = max_num_rois
            else:
                rois[b, :n_objs, :] = torch.Tensor(tmp0)
                categories[b, 1 : n_objs + 1] = torch.IntTensor(tmp1)
                confs[b, :n_objs] = torch.Tensor(tmp2)

            categories[b, 0] = torch.IntTensor([n_objs])

        return rois, categories, confs

    def prepare_node_features(self, full_im, filename, o_im_size):
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

        assert self.one_hot in [0, 2]
        if self.one_hot == 0:
            feat_start_idx = 0

        elif self.one_hot == 2:
            contextual[:, : self.n_out_classes, 0] = 1.0  # size: ([bs,2])
            contextual[:, self.n_out_classes :, 1] = 1.0  # size: ([bs,2])

            feat_start_idx = 2

        rois, categories, conf = self.load_rois_categories(filename, o_im_size)

        # VGG-16 returns all the rois of all images as a single matrix
        rois_feature = self.full_im_net(full_im, rois.view(-1, 4), categories)
        start_idx = 0
        end_idx = 0

        for b in range(num_imgs):
            cur_rois_num = int(categories[b, 0].item())
            if cur_rois_num == 0:
                continue

            end_idx += cur_rois_num

            idxs = categories[b, 1 : (cur_rois_num + 1)].data.tolist()

            # rois_feature = self.full_im_net(full_im[b], rois[b], categories[b])

            # ATTENTION!!!!
            # This assignment overwrites the feature of an object
            # If sorting is activated, the feature vector of the least confident
            # instance is used.
            for idx in range(cur_rois_num):
                cat_id = int(
                    idxs[idx] + self.n_out_classes
                )  # exclude the private nodes
                curr_roi_id = start_idx + idx

                contextual[b, cat_id, feat_start_idx:] = rois_feature[
                    curr_roi_id, :
                ]

            start_idx = end_idx

        tmp_feats = contextual[:, : self.n_out_classes, feat_start_idx:]

        # The following mode is not needed - default one
        if self.net_params["class_feats"] == "zeros":
            tmp_feats_updated = tmp_feats

        else:
            # Feature for privacy nodes
            scores, fc7_feature, rn_feats = self.fis(
                full_im
            )  # full image privacy scores

            if self.net_params["class_feats"] == "transform":
                # tmp_feats_updated = self.fc_resnet(tmp_feats)
                tmp_feats_updated = fc7_feature.view(num_imgs, 1, -1).repeat(
                    1, self.n_out_classes, 1
                )

            elif self.net_params["class_feats"] == "zero_padding":
                tmp_feats_updated[:, :, : rn_feats.shape[1]] = rn_feats.view(
                    num_imgs, 1, -1
                ).repeat(1, self.n_out_classes, 1)

        contextual[
            :, : self.n_out_classes, feat_start_idx:
        ] = tmp_feats_updated

        # print(torch.sum(contextual[0, 0, feat_start_idx:]))
        # print(torch.sum(contextual[0,self.n_out_classes:, feat_start_idx:],1).t())

        return contextual

    def forward(self, img, filename, o_im_size):
        """ """
        num_imgs = img.size()[0]

        node_features = self.prepare_node_features(img, filename, o_im_size)

        model_input = node_features.view(num_imgs, -1)
        # print(model_input.shape)
        grm_feature = self.gnn(model_input)

        nodes_unnormalized_scores = self.classifier(grm_feature).view(
            num_imgs, -1
        )

        if self.b_bce:
            nodes_unnormalized_scores = nodes_unnormalized_scores.squeeze()

        return nodes_unnormalized_scores.float(), None

    # def resnet_transform_layer(self):
    #     """ Transform and align the 2048 ResNet-101 features to the VGG-16 features.

    #     GIP extracts a feature vector from the second last layer of the ResNet-101 backbone.
    #     This feature vector has dimensionality of 2,048. GIP also extracts feature vectors,
    #     whose dimensionality is 4,096, from the region of object of interests as defined by
    #     the bounding boxes predicted by an object detector (YOLO or Mask R-CNN). By aligning
    #     the features to the same and largest dimensionality, GIP can initialise the nodes in
    #     the graph with feature vectors of the same dimensionality for the subsequent processing
    #     via the Graph Neural Network.
    #     """
    #     self.resnet_feat_size = 2048

    #     self.fc_resnet = nn.Linear(
    #         self.resnet_feat_size,  # 2,048
    #         self.gip_feat_size      # 4,096 (VGG-16)
    #     )

    #     # Initialise layer
    #     self.fc_resnet.weight.data.normal_(0, 0.01)
    #     self.fc_resnet.bias.data.zero_()
