#!/usr/bin/env python
#
# Python script for
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/05
# Modified Date: 2024/04/30
#
# -----------------------------------------------------------------------------

import inspect
import os
import sys

current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.insert(0, pp_dir)

import numpy as np
import pandas as pd

import json

from tqdm import tqdm

import torch
import torch.utils.data as data
from torch.autograd import Variable

from srcs.datasets.utils import (
    privacy_classes,
    # get_image_name
)


### PyTorch Geometric graphs
# import time
# from srcs.utils import convert_adj_to_edge_index_weight
# from torch_geometric.data import Data

#############################################################################

# class GraphData(Data):
#     def __init__(self, x, edges, edge_weights):

#         self.x = x
#         self.edges = edges
#         self.edge_weights = edge_weights


class GraphImagePrivacyDataset(data.Dataset):
    """Class for the Image Privacy dataset in graph format.

    The Image Privacy Dataset (or IPD) has 11,211 images labelled as private
    and 23,351 images labelled as public in the corresponding .txt files.
    Public images are all from the PicAlert dataset, whereas 6,392 private
    images are from VISPR and 4,819 private images are from PicAlert.
    The total number of images is 34,562."""

    def __init__(
        self,
        repo_dir=".",
        data_dir="",
        partition="final",
        split="test",
        graph_mode="obj_scene",
        # adj_mat_fn="",
        num_classes=2,
        fold_id=0,
        n_graph_nodes=447,
        node_feat_size=2,
        b_use_card=True,
        b_use_conf=False,
        b_filter_imgs=False,
    ):
        """
        Arguments:
            - repo_dir:
            - data_dir: directory where the dataset (images) is stored.
            - bbox_dir: directory where the pre-computed bounding boxes are stored.
            - partition: Options: crossval, final. Default: final.
            - split: Options: train, val, test. Default: train.
            - num_classes: number of the output classes. Default: 2 (private and public).
            - fold_id:
        """
        super(GraphImagePrivacyDataset, self).__init__()

        # Name of the dataset
        self.dataset_name = "IPD"  # Image Privacy Dataset

        self.graph_mode = graph_mode

        # Paths
        self.repo_dir = repo_dir
        self.data_dir = data_dir

        # Training sets (partitions, and fold)
        self.partition = partition
        self.split_mode = split
        self.fold_id = fold_id

        # Binary classification task
        self.n_out_cls = num_classes
        self.n_graph_nodes = n_graph_nodes
        self.node_feat_size = node_feat_size

        # Boolean variables
        self.b_use_card = b_use_card  # Boolean to use cardinality as feature
        self.b_use_conf = b_use_conf  # Boolean to use confidence  as feature

        self.b_filter_imgs = b_filter_imgs  # For only object categories case

        # Load annotation file
        self.get_annotations()
        self.set_normalization()

    def set_normalization(self):
        """ """
        print("Features are normalised!")
        try:
            if self.partition == "crossval":
                if self.node_feat_size == 2:
                    self.data_mean = torch.tensor([0.5302, 0.4115])
                    self.data_std = torch.tensor([0.1883, 0.2428])
                elif self.node_feat_size == 1:
                    if self.b_use_card:
                        self.data_mean = torch.tensor([0.5302])
                        self.data_std = torch.tensor([0.1883])
                    elif self.b_use_conf:
                        self.data_mean = torch.tensor([0.4115])
                        self.data_std = torch.tensor([0.2428])

            else:
                raise ValueError(
                    "Training statistics for data partition {:s} are not yet computed!".format(
                        self.partition
                    )
                )
        except (ValueError, IndexError):
            exit("Feature normalization: Could not complete request.")

    def normalise_features(self, x, mode="z-score"):
        """ """
        assert mode in ["z-score", "min-max"]

        if mode == "z-score":
            return torch.div(torch.sub(x, self.data_mean), self.data_std)
        elif mode == "min-max":
            ### TODO
            return torch.div(torch.sub(x, self.data_mean), self.data_std)

    def get_annotations(self):
        """Read annotation files (CSV manifest) and set class variables.

        Information/variables read:
            - self.imgs:    list of image filenames
            - self.labels:  list of annotation labels (for each corresponding image)
            - self.weights: list of class weights (repeated for each
                            corresponding image) to handle class imbalance
            - self.cls_weights: vector of class weights to handle class
                                imbalance (globally)
        """
        df = pd.read_csv(
            os.path.join(
                self.data_dir,
                "annotations",
                "labels_splits.csv",
            ),
            delimiter=",",
            index_col=False,
        )

        img_name = df["Image Name"]
        labels = df["Label {:d}-class".format(self.n_out_cls)]

        assert self.partition in ["crossval", "final"]

        if self.partition == "crossval":
            assert self.split_mode in ["train", "val", "test"]

        if self.partition == "final":
            assert self.split_mode in ["train", "test"]

        if self.partition == "crossval":
            fold_str = "Fold {:d}".format(self.fold_id)
        elif self.partition == "final":
            fold_str = "Final"

        if self.split_mode == "train":
            l_imgs = img_name[df[fold_str] == 0].values
            l_lables = labels[df[fold_str] == 0].values

        elif self.split_mode == "val":
            l_imgs = img_name[df[fold_str] == 1].values
            l_lables = labels[df[fold_str] == 1].values

        elif self.split_mode == "test":
            l_imgs = img_name[df[fold_str] == 2].values
            l_lables = labels[df[fold_str] == 2].values

        # As filenames alone cannot help retrieve the images, we append the
        # full filepath and extension. We therefore create an updated list
        json_out = os.path.join(
            self.data_dir, "annotations", "ipd_imgs_curated.json"
        )
        ipd_curated = json.load(open(json_out, "r"))
        annotations = ipd_curated["annotations"]
        l_img_ann = [x["image"] for x in annotations]
        l_data_prefix = [
            ipd_curated["datasets"][x["dataset"]]["data_dir"]
            for x in annotations
        ]

        idx_valid = np.where(l_lables != -1)[0].tolist()
        l_lables = l_lables[idx_valid]

        new_data_prefix_l = []
        new_img_l = []
        for idx, img in enumerate(tqdm(l_imgs)):
            if idx not in idx_valid:
                continue

            idx2 = l_img_ann.index(img)
            full_img_path = annotations[idx2]["fullpath"]
            new_img_l.append(full_img_path)
            new_data_prefix_l.append(l_data_prefix[idx2])

        self.imgs = new_img_l
        self.labels = l_lables.tolist()
        self.data_prefix = new_data_prefix_l

        if self.b_filter_imgs:
            self.filter_images_labels()
        else:
            self.update_weights()

    def filter_images_labels(self):
        """ """
        # new_node_feats_list = []
        new_img_list = []
        new_labels_list = []
        new_data_prefix_l = []

        json_out = os.path.join(
            self.data_dir,
            "resources",
            "n_obj_img_{:s}.json".format(self.get_name_low()),
        )

        with open(json_out) as f:
            obj_img = json.load(f)

        for idx, img_fn in enumerate(self.imgs):
            image_name = get_image_name(img_fn)

            if obj_img[image_name] > 1:
                new_img_list.append(img_fn)
                new_labels_list.append(self.labels[idx])
                new_data_prefix_l.append(self.data_prefix[idx])
                # new_node_feats_list.append(self.node_feats_list[idx])

        # for idx in np.where(obj_img == 1)[0]:
        #     new_node_feats_list.append(self.node_feats_list[idx])
        #     new_img_list.append(self.imgs[idx])
        #     new_labels_list.append(self.labels[idx])

        # self.node_feats_list = new_node_feats_list
        self.imgs = new_img_list
        self.labels = new_labels_list
        self.data_prefix = new_data_prefix_l

        self.update_weights()

    def update_weights(self):
        """ """
        cls_weights = np.zeros(self.n_out_cls)
        labels = np.unique(self.labels)
        l_weights = np.array(self.labels)

        if self.n_out_cls == 2:
            cls_names = privacy_classes[self.dataset_name]["binary"]
        # elif self.n_out_cls == 3:
        #     cls_names = privacy_classes["ternary"]
        # elif self.n_out_cls == 5:
        #     cls_names = privacy_classes["quinary"]

        for l_idx in labels:
            n_imgs_cls = np.count_nonzero(self.labels == l_idx)
            l_weights[self.labels == l_idx] = 1.0 - n_imgs_cls / len(self.imgs)

            print(
                "Number of {:s} images: {:d}".format(
                    cls_names[l_idx], n_imgs_cls
                )
            )

            cls_weights[l_idx] = 1.0 - n_imgs_cls / len(self.imgs)

        self.weights = l_weights
        self.cls_weights = cls_weights

    def get_class_weights(self):
        return self.cls_weights

    def get_name_low(self):
        """Return the name of the dataset in low case"""
        return self.dataset_name.lower()

    def get_labels(self):
        return self.labels

    def get_data_dir(self):
        return self.data_dir

    def load_node_features(self, index, graph_mode, b_use_cls_nodes=True):
        """ """

        # Get node features filename
        img_fn = self.imgs[index]

        if img_fn.endswith("\n"):
            img_fn = self.imgs[index].split("\n")[0]

        if self.dataset_name == "IPD":
            if "vispr" in self.data_prefix[index]:
                dataset_name = "VISPR"
            elif "picalert" in self.data_prefix[index]:
                dataset_name = "PicAlert"
        else:
            dataset_name = self.dataset_name

        path_elems = img_fn.split("/")
        if len(path_elems) == 2 and "batch" in path_elems[-2]:
            filename = os.path.join(
                self.repo_dir,
                "resources",
                dataset_name,
                "graph_data",
                # self.config["method_name"],
                "node_feats",
                path_elems[-2],
                path_elems[-1].split(".")[0] + ".json",
            )
        else:
            filename = os.path.join(
                self.repo_dir,
                "resources",
                dataset_name,
                "graph_data",
                # self.config["method_name"],
                "node_feats",
                path_elems[-1].split(".")[0] + ".json",
            )
        if not os.path.isfile(filename):
            print(filename)

        assert self.node_feat_size > 0

        if self.b_use_card != self.b_use_conf:
            assert self.node_feat_size == 1
        elif (self.b_use_card == self.b_use_conf) and (self.b_use_card):
            assert self.node_feat_size == 2

        node_feats_var = Variable(
            torch.zeros(self.n_graph_nodes, self.node_feat_size),
            requires_grad=False,
        )

        node_feats = json.load(open(filename))

        for node in node_feats:
            n_id = node["node_id"]

            if not b_use_cls_nodes:
                if n_id < self.n_out_cls:
                    continue
                else:
                    n_id -= self.n_out_cls

            n_feat = node["node_feature"]
            # assert(len(n_feat) > 0 and len(n_feat) < 3)

            if self.node_feat_size == 1:
                if len(n_feat) > 1:
                    if self.b_use_card:
                        n_feat = [n_feat[0]]
                    elif self.b_use_conf:
                        n_feat = [n_feat[1]]

            elif self.node_feat_size == 2:
                n_feat = n_feat[:2]

            node_feats_var[n_id, :] = torch.Tensor(n_feat)

        # Thresholding - hardcoded value (not used for the moment)
        # if (self.node_feat_size == 1) and (self.b_use_conf):
        #     node_feats_var = (node_feats_var > 0.5).type(node_feats_var.type())

        return node_feats_var, img_fn

    def __getitem__(self, index):
        """Return one item of the iterable data

        The function returns the node features, the privacy label (target),
        the class weights, and the filename of the corresponding image in the
        dataset. In addition to the single image data, the adjacency_matrix of
        the prior graph is provided as output of the function.
        """
        node_feats, img_fn = self.load_node_features(index, self.graph_mode)
        node_feats = self.normalise_features(node_feats)

        target = self.labels[index]

        weight = self.weights[index]

        return node_feats, target, weight, img_fn

    # def __getitem__(self, index):
    #     """Return one item of the iterable data

    #     The function returns the node features, the privacy label (target),
    #     the class weights, and the filename of the corresponding image in the
    #     dataset. In addition to the single image data, the adjacency_matrix of
    #     the prior graph is provided as output of the function.
    #     """
    #     node_feats, img_fn = self.load_node_features(index, self.graph_mode)
    #     node_feats = self.normalise_features(node_feats)

    #     data = Data(
    #         x=torch.from_numpy(node_feats),
    #         edge_index=torch.from_numpy(self.prior_graph_edge_index),
    #         edge_attr=torch.from_numpy(self.prior_graph_edge_weights),
    #         y=self.labels[index]
    #     )

    #     data.weights = torch.tensor(self.weights[index])
    #     data.image_name = img_fn

    #     if self.b_use_prior_graph:
    #         return data
    #     else:
    #         return data.subgraph(subset)

    def __len__(self):
        return len(self.imgs)
