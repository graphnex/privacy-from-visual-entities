#!/usr/bin/env python
#
# Brief description here
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/05/15
# Modified Date: 2024/08/14
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
import json  # Open standard file format to store data

from tqdm import tqdm

import torch.utils.data as data

from srcs.datasets.utils import load_image, privacy_classes


#############################################################################
## Class for Subset of the Image Privacy dataset converted into node features.
#


class PrivacyAlertDataset(data.Dataset):
    """Class for the PrivacyAlert dataset."""

    def __init__(
        self,
        repo_dir=".",
        data_dir="",
        partition="final",
        split="test",
        num_classes=2,
        fold_id=0,
        b_filter_imgs=False,
        img_size=448,
    ):
        super(PrivacyAlertDataset, self).__init__()

        # Name of the dataset
        self.dataset_name = "PrivacyAlert"

        # Paths
        self.repo_dir = repo_dir
        self.data_dir = data_dir

        # Training sets (partitions, and fold)
        self.partition = partition
        self.split = split
        self.fold_id = fold_id

        # Binary classification task
        self.n_out_cls = num_classes

        # Boolean variables
        self.b_filter_imgs = b_filter_imgs

        # Load annotation file
        self.get_annotations()

        self.im_size = img_size

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
            "n_obj_img_{:s}.json".format(self.get_name_low()),
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

    def __getitem__(self, index):
        """Return one item of the iterable data

        The function returns the node features, the privacy label (target),
        the class weights, and the filename of the corresponding image in the
        dataset. In addition to the single image data, the adjacency_matrix of
        the prior graph is provided as output of the function.
        """
        img_fn = self.update_filename(self.imgs[index])

        # if img_fn.endswith("\n"):
        #     img_fn = self.imgs[index].split("\n")[0]
        # filename = img_fn.split(".")[0] + ".json"

        # filename = os.path.join(
        #     self.data_dir, "graph_data", self.graph_mode, "node_feats", filename
        # )
        # if not os.path.isfile(filename):
        #     print(filename)

        # image_name = get_image_name(self.imgs[index])
        try:
            full_im, w, h = load_image(
                self.imgs[index] + ".jpg",
                os.path.join(self.data_dir, "imgs"),
                img_size=self.im_size,
            )
        except:
            full_im, w, h = load_image(
                self.imgs[index] + ".png",
                os.path.join(self.data_dir, "imgs"),
                img_size=self.im_size,
            )

        # print(full_im.shape)

        # full_im = self.full_transform(full_im)

        target = self.labels[index]

        weight = self.weights[index]

        return full_im, target, weight, img_fn, np.array([w, h])

    def __len__(self):
        return len(self.imgs)

    def update_filename(self, img_fn):
        """ """
        if img_fn.endswith("\n"):
            img_fn = img_fn.split("\n")[0]

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

        return filename
