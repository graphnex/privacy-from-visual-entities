#!/usr/bin/env python
#
# Module to process and convert an image into a set of concepts and features.
#
# Run as:
#        python srcs/datasets/normalization.py              \
#           --config        configs/normalise_data.json     \
#           --dataset       PrivacyAlert                    \
#           --training_mode original
#
###############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2024/04/29
# Modified Date: 2024/04/29
#
# MIT License

# Copyright (c) 2024 GraphNEx

import argparse
import inspect

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# System libraries
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

from tqdm import tqdm
import json

# PyTorch classes, functions
import torch
from torch.utils.data import DataLoader

# Package modules
from srcs.datasets.wrapper import WrapperDatasets
from srcs.utils import set_seed, device

# Module/library for debugging by placing a bp() command at the line where you
# want to debug the code. At running time from shell, the code pauses at the
# line where you placed bp() and you can perform any operation/query with python
# commands directly from the shell.
# Details and info at https://docs.python.org/3/library/pdb.html


class DatasetStatistics:
    def __init__(self, config, dataset):
        """ """

        set_seed(config["params"]["seed"])

        self.repo_dir = config["paths"]["root_dir"]

        self.data_dir = os.path.join(
            config["paths"]["data_prefix"],
            config["datasets"][dataset]["data_dir"],
        )

        self.b_filter_imgs = config["b_filter_imgs"]

        self.dataset = dataset

        self.params = config["params"]
        self.num_workers = config["num_workers"]

        self.training_mode = self.params["training_mode"]
        self.split_mode = "train"

        self.s_sum = None
        self.s_min = None
        self.s_max = None
        self.s_mean = None
        self.s_std = None

    def load_training_data(self):
        """ """
        data_wrapper = WrapperDatasets(
            root_dir=self.repo_dir,
            data_dir=self.data_dir,
            num_classes=self.params["num_out_classes"],
            fold_id=self.params["fold_id"],
            graph_mode=self.params["graph_type"],
            n_graph_nodes=self.params["n_graph_nodes"],
            node_feat_size=self.params["node_feat_size"],
        )

        assert self.split_mode == "train"

        data_wrapper.load_split_set(
            self.dataset,
            partition=self.params["training_mode"],
            mode=self.split_mode,
            b_use_card=self.params["use_card"],
            b_use_conf=self.params["use_conf"],
            b_filter_imgs=self.b_filter_imgs,
        )

        set_seed(self.params["seed"])  # for replicability of the shuffling

        training_loader = DataLoader(
            data_wrapper.get_data_split(self.split_mode),
            batch_size=self.params["batch_size_train"],
            shuffle=True,
            num_workers=self.num_workers,
            # drop_last=True,
        )

        return training_loader

    def compute_training_statistics(self, training_set):
        """ """
        num_imgs = len(training_set)

        for batch_idx, (
            node_feats,
            target,
            weights,
            image_name,
        ) in enumerate(tqdm(training_set, ascii=True)):
            if self.s_sum is None:
                self.s_sum = torch.sum(node_feats, (0, 1))
            else:
                self.s_sum += torch.sum(node_feats, (0, 1))

            if self.s_min is None:
                self.s_min = torch.min(
                    node_feats.view(-1, self.params["node_feat_size"]), 0
                ).values
            else:
                batch_min = torch.min(
                    node_feats.view(-1, self.params["node_feat_size"]), 0
                ).values
                self.s_min = torch.minimum(batch_min, self.s_min)

            if self.s_max is None:
                self.s_max = torch.max(
                    node_feats.view(-1, self.params["node_feat_size"]), 0
                ).values
            else:
                batch_max = torch.max(
                    node_feats.view(-1, self.params["node_feat_size"]), 0
                ).values
                self.s_max = torch.maximum(batch_max, self.s_max)

        self.s_mean = self.s_sum / (num_imgs * self.params["n_graph_nodes"])

        s_sum_std = None

        for batch_idx, (
            node_feats,
            target,
            weights,
            image_name,
        ) in enumerate(tqdm(training_set, ascii=True)):
            node_feats_mean = torch.pow(torch.sub(node_feats, self.s_mean), 2)

            if s_sum_std is None:
                s_sum_std = torch.sum(node_feats_mean, (0, 1))
            else:
                s_sum_std += torch.sum(node_feats_mean, (0, 1))

        self.s_std = torch.rsqrt(
            s_sum_std / (num_imgs * self.params["n_graph_nodes"])
        )

    def print_training_statistics(self):
        """ """
        print("### {:s} ###".format(self.dataset))
        print("Min:")
        print(self.s_min)
        print("Max:")
        print(self.s_max)
        print("Mean:")
        print(self.s_mean)
        print("Standard deviation:")
        print(self.s_std)

    def save_training_statistics(self):
        """ """
        if self.training_mode == "original":
            fn = os.path.join(
                self.repo_dir,
                "resources",
                self.dataset,
                "training_stats_original.txt",
            )
        elif self.training_mode == "final":
            fn = os.path.join(
                self.repo_dir,
                "resources",
                self.dataset,
                "training_stats_final.txt",
            )
        elif self.training_mode == "crossval":
            fn = os.path.join(
                self.repo_dir,
                "resources",
                self.dataset,
                "training_stats_crossval_fold{:d}.txt".format(
                    self.params["fold_id"]
                ),
            )

        if not os.path.exists(os.path.dirname(fn)):
            os.makedirs(os.path.dirname(fn), exist_ok=True)

        fh = open(fn, "w")
        fh.write("### {:s} ###\n".format(self.dataset))
        fh.write("Min:")
        fh.write(", ".join(["{:.4f}".format(x_) for x_ in self.s_min]))
        fh.write("\nMax:")
        fh.write(", ".join(["{:.4f}".format(x_) for x_ in self.s_max]))
        fh.write("\nMean:")
        fh.write(", ".join(["{:.4f}".format(x_) for x_ in self.s_mean]))
        fh.write("\nStandard deviation:")
        fh.write(", ".join(["{:.4f}".format(x_) for x_ in self.s_std]))
        fh.write("\n")
        fh.close()

    def run(self):
        """ """
        training_set = self.load_training_data()

        self.compute_training_statistics(training_set)
        self.print_training_statistics()
        self.save_training_statistics()


#############################################################################


def GetParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Please provide a config.json file"
    )
    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["GIPS", "PrivacyAlert", "PicAlert", "VISPR", "IPD"],
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        choices=["final", "crossval", "original"],
        required=True,
        help="Choose to run K-fold cross-validation or train the final model (full training set without validation split)",
    )
    parser.add_argument("--fold_id", type=int)
    return parser


if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))
    print("PyTorch {}".format(torch.__version__))
    print("Using {}".format(device))

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Add dataset configurations
    with open(os.path.join("configs", "datasets.json")) as f:
        data_config = json.load(f)

    config["paths"] = data_config["paths"]
    config["datasets"] = data_config["datasets"]

    if args.training_mode is not None:
        config["params"]["training_mode"] = args.training_mode

    if args.fold_id is not None:
        config["params"]["training_mode"] = args.fold_id

    data_stats = DatasetStatistics(config, args.dataset)
    data_stats.run()
