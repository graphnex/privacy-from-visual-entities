#!/usr/bin/env python
#
# Brief description here
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2024/04/30
# Modified Date: 2024/04/30
#
# -----------------------------------------------------------------------------

import json  # Open standard file format to store data
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.autograd import Variable
from tqdm import tqdm  # smart progress meter for loops

import warnings

from srcs.datasets.utils import privacy_classes


#############################################################################
## Class for Subset of the Image Privacy dataset converted into node features.
#


class GraphImagePrivacyDatasets(data.Dataset):
    """Class for the PrivacyAlert dataset."""

    def __init__(
        self,
        dataset_name="PrivacyAlert",
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
        super(GraphImagePrivacyDatasets, self).__init__()

        # Name of the dataset
        assert dataset_name in [
            "GIPS",
            "PrivacyAlert",
            "PicAlert",
            "VISPR",
            "IPD",
        ]
        self.dataset_name = dataset_name

        self.graph_mode = graph_mode

        # Paths
        self.repo_dir = repo_dir
        self.data_dir = data_dir

        # Training sets (partitions, and fold)
        self.partition = partition
        self.split = split
        self.fold_id = fold_id

        # Binary classification task
        self.n_out_cls = num_classes
        self.n_graph_nodes = n_graph_nodes
        self.node_feat_size = node_feat_size

        # Boolean variables
        self.b_use_card = b_use_card  # Boolean to use cardinality as feature
        self.b_use_conf = b_use_conf  # Boolean to use confidence  as feature

        if "b_normalise_feats" in config["net_params"]:
            self.b_normalise_feats = config["net_params"]["b_normalise_feats"]
        else:
            warnings.warn(
                "Parameter 'b_normalise_feats' not present in the config file! Parameter set to False by default!"
            )
            self.b_normalise_feats = False

        self.b_filter_imgs = b_filter_imgs

        self.b_use_cls_nodes = True

        # Load annotation file
        self.get_annotations()

        input("Is variable for feature normalisation set?")
        if self.b_normalise_feats:
            self.set_normalization()

    def set_normalization(self):
        """ """
        print("Features are normalised!")
        try:
            if self.dataset_name == "PrivacyAlert":
                print(self.partition)

                if self.partition == "original":
                    if self.node_feat_size == 2:
                        self.data_mean = torch.tensor([0.5590, 0.4189])
                        self.data_std = torch.tensor([0.1805, 0.2410])
                    elif self.node_feat_size == 1:
                        if self.b_use_card:
                            self.data_mean = torch.tensor([0.5590])
                            self.data_std = torch.tensor([0.1805])
                        elif self.b_use_conf:
                            self.data_mean = torch.tensor([0.4189])
                            self.data_std = torch.tensor([0.2410])
                else:
                    raise ValueError(
                        "Training statistics for a data partition different from original are not yet computed!"
                    )

            if self.dataset_name == "IPD":
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

            # if self.partition == "original":
            #     if self.node_feat_size == 2:
            #         self.data_mean = torch.tensor([0.7392, 0.6100])
            #         self.data_std = torch.tensor([0.1360, 0.1660])
            #     elif self.node_feat_size == 1:
            #         if self.b_use_card:
            #             self.data_mean = torch.tensor([0.7392])
            #             self.data_std = torch.tensor([0.1360])
            #         elif self.b_use_conf:
            #             self.data_mean = torch.tensor([0.6100])
            #             self.data_std = torch.tensor([0.1660])

            # else:
            #     raise ValueError(
            #         "Training statistics for a data partition different from original are not yet computed!")
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

    def get_all_image_names(self):
        """ """
        df = pd.read_csv(
            os.path.join(
                self.data_dir,
                "annotations",
                "labels_splits.csv",
            ),
            delimiter=",",
            index_col=False,
        )

        if self.dataset_name == "GIPS":
            img_name = df["Image Name"].astype(str)

            img_list = img_name.values

        elif self.dataset_name == "PicAlert":
            img_name = df["Image Name"]

            # As filenames alone cannot help retrieve the images, we append the
            # full filepath and extension. We therefore create an updated list
            json_out = os.path.join(
                self.data_dir, "annotations", "ipd_imgs_curated.json"
            )
            ipd_curated = json.load(open(json_out, "r"))
            annotations = ipd_curated["annotations"]
            l_img_ann = [
                x["image"]
                for x in annotations
                if x["dataset"] == self.dataset_name
            ]
            l_data_prefix = [
                ipd_curated["datasets"][x["dataset"]]["data_dir"]
                for x in annotations
                if x["dataset"] == self.dataset_name
            ]

            new_data_prefix_l = []
            new_img_l = []
            for idx, img in enumerate(tqdm(img_name.values)):
                if img in l_img_ann:
                    idx2 = l_img_ann.index(img)
                    full_img_path = annotations[idx2]["fullpath"]
                    new_img_l.append(full_img_path)

            img_list = new_img_l

        elif self.dataset_name == "VISPR":
            img_name = (
                "batch"
                + df["batch"].astype(str).str.zfill(2)
                + "/"
                + df["Image Name"].astype(str)
            )

            img_list = img_name.values

        else:
            img_name = (
                "batch"
                + df["batch"].astype(str)
                + "/"
                + df["Image Name"].astype(str)
            )

            img_list = img_name.values

        return img_list

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

        assert self.partition in ["crossval", "final", "original"]

        if (self.partition == "crossval") | (self.partition == "original"):
            assert self.split in ["train", "val", "test"]

        if self.partition == "final":
            assert self.split in ["train", "test"]

        if self.partition == "final":
            fold_str = "Final"
        elif self.partition == "original":
            fold_str = "Original"
        else:
            fold_str = "Fold {:d}".format(self.fold_id)

        img_name = (
            "batch"
            + df["batch"].astype(str)
            + "/"
            + df["Image Name"].astype(str)
        )
        # img_name = df["Image Name"].astype(str)
        # img_name = df["batch"].astype(str)
        labels = df["Label {:d}-class".format(self.n_out_cls)]

        assert self.split in ["train", "val", "test"]

        if self.split == "train":
            l_imgs = img_name[df[fold_str] == 0].values
            l_lables = labels[df[fold_str] == 0].values

        elif self.split == "val":
            l_imgs = img_name[df[fold_str] == 1].values
            l_lables = labels[df[fold_str] == 1].values

        elif self.split == "test":
            l_imgs = img_name[df[fold_str] == 2].values
            l_lables = labels[df[fold_str] == 2].values

        idx_valid = np.where(l_lables != -1)[0].tolist()
        l_lables = l_lables[idx_valid]
        l_imgs = l_imgs[idx_valid]

        self.imgs = l_imgs
        self.labels = l_lables.tolist()
        # self.data_prefix = [ipd_curated["datasets"][x["dataset"]]["data_dir"] for x in annotations]

        if self.b_filter_imgs:
            self.filter_images_labels()
        else:
            self.update_weights()

    def filter_images_labels(self):
        """ """
        # new_node_feats_list = []
        new_img_list = []
        new_labels_list = []
        # new_data_prefix_l = []

        json_out = os.path.join(
            self.data_dir,
            "resources",
            "n_obj_img_{:s}}.json".format(self.get_name_low),
        )

        with open(json_out) as f:
            obj_img = json.load(f)

        for idx, img_fn in enumerate(self.imgs):
            # bp()
            # if img_fn.endswith("\n"):
            #     img_fn2 = self.imgs[idx].split("\n")[0]
            #     image_name = get_image_name(img_fn2)
            # else:
            #     image_name = get_image_name(img_fn)

            if obj_img[img_fn.split("/")[1]] > 1:
                new_img_list.append(img_fn)
                new_labels_list.append(self.labels[idx])
                # new_data_prefix_l.append(self.data_prefix[idx])
                # new_node_feats_list.append(self.node_feats_list[idx])

        # for idx in np.where(obj_img == 1)[0]:
        #     new_node_feats_list.append(self.node_feats_list[idx])
        #     new_img_list.append(self.imgs[idx])
        #     new_labels_list.append(self.labels[idx])

        # self.node_feats_list = new_node_feats_list
        self.imgs = new_img_list
        self.labels = new_labels_list
        # self.data_prefix = new_data_prefix_l

        self.update_weights()

    def update_weights(self):
        """ """
        cls_weights = np.zeros(self.n_out_cls)
        labels = np.unique(self.labels)
        l_weights = np.array(self.labels)

        if self.n_out_cls == 2:
            cls_names = privacy_classes[self.dataset_name]["binary"]

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

    def load_node_features(self, filename):
        """ """
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

            if not self.b_use_cls_nodes:
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

        return node_feats_var

    def __getitem__(self, index):
        """Return one item of the iterable data

        The function returns the node features, the privacy label (target),
        the class weights, and the filename of the corresponding image in the
        dataset. In addition to the single image data, the adjacency_matrix of
        the prior graph is provided as output of the function.
        """
        img_fn = self.imgs[index]

        if img_fn.endswith("\n"):
            img_fn = self.imgs[index].split("\n")[0]
        filename = img_fn.split(".")[0] + ".json"

        filename = os.path.join(
            self.data_dir,
            "graph_data",
            self.graph_mode,
            "node_feats",
            filename,
        )
        if not os.path.isfile(filename):
            print(filename)

        node_feats = self.load_node_features(filename)

        target = self.labels[index]

        weight = self.weights[index]

        if self.b_normalise_feats:
            node_feats_norm = self.normalise_features(node_feats)

            return node_feats_norm, target, weight, img_fn

        return node_feats, target, weight, img_fn

    def __len__(self):
        return len(self.imgs)
