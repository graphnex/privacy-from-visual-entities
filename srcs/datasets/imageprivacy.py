#!/usr/bin/env python
#
# Python script for
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2024/09/06
# Modified Date: 2024/09/06
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

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

from pdb import set_trace as bp  # This is only for debugging

#############################################################################

privacy_classes = {
    "IPD": {
        "binary": {0: "private", 1: "public"},
    },
    "PrivacyAlert": {
        "binary": {0: "public", 1: "private"},
    },
}

##############################################################################


def get_image_name(filename):
    """Get the image name without path and extension."""
    image_name = filename.split("/")[-1]  # e.g; '2017_80112549.json'
    image_name = image_name.split(".")[-2]

    return image_name


def set_img_transform(img_size=448, b_normalize=True):
    """ """
    if b_normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )  # imagenet values

        im_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )  # what about horizontal flip

    else:
        im_transform = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size)
                ),  # transforms.Resize((448, 448)),
                transforms.ToTensor(),
            ]
        )

    return im_transform


def load_image(filename, data_dir, b_transform=True, img_size=448):
    """Load an image from a path and apply any transformation required for the model.

    The function loads an image from a location into PIL format.
    If the flag tranform is enabled, a sequence of transformations is applied to the image to be
    used as input to a PyTorch model. These transformations include resizing and standardization.

    Arguments:
        - filename: name of the image with extension.
        - data_dir: path to the directory where the image is stored.
        - b_transform [boolean]: flag to enable the transformation of the image in PyTorch by applying a resize, tensor conversion, and standardization with imagenet values.
        - img_size: resolution of the transformed image if b_transform is enabled.

    Return:
        - full_im: the loaded image in PIL format. If transformation is enabled,
            then the image is transformed and resized to a fixed dimensionality.
        - w: width of the original image (no transformations applied).
        - h: width of the original image (no transformations applied).
    """
    # For normalize
    # filename = self.imgs[index]
    if filename.endswith("\n"):
        filename = filename.split("\n")[0]

    filename = os.path.join(data_dir, filename)
    if not os.path.isfile(filename):
        print(filename)

    # https://pillow.readthedocs.io/en/stable/reference/Image.html
    img = Image.open(filename).convert("RGB")
    (im_width, im_height) = img.size  # e.g; (1024, 1019)

    if b_transform:
        im_transform = set_img_transform(img_size=img_size)
        full_im = im_transform(img)
        # e.g; for index 10 full_im.shape = [3, 448, 448]
    else:
        full_im = img

    return full_im, im_width, im_height


class ImagePrivacyDatasets(data.Dataset):
    """Class for the image privacy datasets."""

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
        name="",
    ):
        """
        Arguments:
            - repo_dir:
            - data_dir: directory where the dataset (images) is stored.
            - partition: Options: crossval, final. Default: final.
            - split: Options: train, val, test. Default: train.
            - num_classes: number of the output classes. Default: 2 (private and public).
            - fold_id:
        """
        super(ImagePrivacyDatasets, self).__init__()

        # Name of the dataset
        assert name in ["PicAlert", "VISPR", "IPD", "PrivacyAlert"]
        self.dataset_name = name

        # Paths
        self.repo_dir = repo_dir
        self.data_dir = data_dir

        # Training sets (partitions, and fold)
        self.partition = partition
        self.split_mode = split
        self.fold_id = fold_id

        self.check_data_partition()

        # Binary classification task
        self.n_out_cls = num_classes

        self.im_size = img_size

        self.b_filter_imgs = b_filter_imgs  # For only object categories case

        # Load annotation file
        self.get_annotations()

    def check_data_partition(self):
        """ """
        assert self.split_mode in ["train", "val", "test"]

        assert self.partition in ["crossval", "final", "original"]

        if (self.partition == "crossval") | (self.partition == "original"):
            assert self.split_mode in ["train", "val", "test"]

        if self.partition == "final":
            assert self.split_mode in ["train", "test"]

    def get_folder_string(self):
        """ """
        if self.partition == "final":
            fold_str = "Final"
        elif self.partition == "original":
            fold_str = "Original"
        else:
            fold_str = "Fold {:d}".format(self.fold_id)

        return fold_str

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

        # IPD
        if self.dataset_name == "PrivacyAlert":
            img_name = (
                "batch"
                + df["batch"].astype(str)
                + "/"
                + df["Image Name"].astype(str)
            )
        else:
            img_name = df["Image Name"]
        labels = df["Label {:d}-class".format(self.n_out_cls)]

        fold_str = self.get_folder_string()

        if self.split_mode == "train":
            l_imgs = img_name[df[fold_str] == 0].values
            l_lables = labels[df[fold_str] == 0].values

        elif self.split_mode == "val":
            l_imgs = img_name[df[fold_str] == 1].values
            l_lables = labels[df[fold_str] == 1].values

        elif self.split_mode == "test":
            l_imgs = img_name[df[fold_str] == 2].values
            l_lables = labels[df[fold_str] == 2].values

        idx_valid = np.where(l_lables != -1)[0].tolist()
        l_lables = l_lables[idx_valid]

        if self.dataset_name == "IPD":
            new_img_l, l_data_prefix = self.load_ipd_data_prefix(
                l_imgs, idx_valid
            )

            self.data_prefix = l_data_prefix
            self.imgs = new_img_l
        else:
            self.imgs = l_imgs

        self.labels = l_lables.tolist()

        if self.b_filter_imgs:
            self.filter_images_labels()
        else:
            self.update_weights()

    def load_ipd_data_prefix(self, l_imgs, idx_valid):
        """Load the prefix path of individual datasets forming IPD.

        IPD is a composed dataset of two datasets: PicAlert and VISPR.
        Because of this, files are located to each specific dataset and
        hence we need to retrieve them.

        As filenames alone cannot help retrieve the images, we append the
        full filepath and extension. We therefore create an updated list.
        """
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

        new_data_prefix_l = []
        new_img_l = []
        for idx, img in enumerate(tqdm(l_imgs)):
            if idx not in idx_valid:
                continue

            idx2 = l_img_ann.index(img)
            full_img_path = annotations[idx2]["fullpath"]
            new_img_l.append(full_img_path)
            new_data_prefix_l.append(l_data_prefix[idx2])

        return new_img_l, new_data_prefix_l

    def filter_images_labels(self):
        """ """
        new_img_list = []
        new_labels_list = []

        if self.dataset_name == "IPD":
            new_data_prefix_l = []

        json_out = os.path.join(
            self.data_dir,
            "resources",
            "n_obj_img_{:s}.json".format(self.get_name_low()),
        )

        with open(json_out) as f:
            obj_img = json.load(f)

        for idx, img_fn in enumerate(self.imgs):
            if self.dataset_name == "IPD":
                image_name = get_image_name(img_fn)
            else:
                image_name = img_fn.split("/")[1]

            if obj_img[image_name] > 1:
                new_img_list.append(img_fn)
                new_labels_list.append(self.labels[idx])

                if self.dataset_name == "IPD":
                    new_data_prefix_l.append(self.data_prefix[idx])

        self.imgs = new_img_list
        self.labels = new_labels_list

        if self.dataset_name == "IPD":
            self.data_prefix = new_data_prefix_l

        self.update_weights()

    def update_weights(self):
        """ """
        cls_weights = np.zeros(self.n_out_cls)
        labels = np.unique(self.labels)
        l_weights = np.array(self.labels)

        # if self.n_out_cls == 2:
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

    def get_data_dir(self):
        return self.data_dir

    def __getitem__(self, index):
        """Return one item of the iterable data

        The function returns the node features, the privacy label (target),
        the class weights, and the filename of the corresponding image in the
        dataset. In addition to the single image data, the adjacency_matrix of
        the prior graph is provided as output of the function.
        """
        # Not sure this is needed - to double check
        # img_fn = self.update_filename(self.imgs[index], index)

        if self.dataset_name == "IPD":
            full_im, w, h = load_image(
                self.imgs[index],
                os.path.join(self.data_prefix[index], "imgs"),
                img_size=self.im_size,
            )
        elif self.dataset_name == "PrivacyAlert":
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

        target = self.labels[index]

        weight = self.weights[index]

        return full_im, target, weight, self.imgs[index], np.array([w, h])

    def __len__(self):
        return len(self.imgs)

    def update_filename(self, img_fn, index):
        """ """
        if img_fn.endswith("\n"):
            img_fn = img_fn.split("\n")[0]

        # print(self.dataset_name)
        if self.dataset_name == "IPD":
            if "vispr" in self.data_prefix[index]:
                dataset_name = "VISPR"
            elif "picalert" in self.data_prefix[index]:
                dataset_name = "PicAlert"
        else:
            dataset_name = self.dataset_name

        # print(dataset_name)
        # print(self.repo_dir)

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

        # print(filename)

        return filename
