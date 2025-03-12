#!/usr/bin/env python
#
# Brief description here
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/09/05
# Modified Date: 2024/09/06
#
# -----------------------------------------------------------------------------

import json
import os

import pandas as pd
from tqdm import tqdm

from srcs.datasets.imageprivacy import ImagePrivacyDatasets

# from srcs.datasets.gips import GraphImagePrivacySubset
from srcs.datasets.graph_image_privacy import GraphImagePrivacyDatasets
from srcs.datasets.ipd_graph import GraphImagePrivacyDataset
# from srcs.datasets.picalert_graph import PicAlertDataset
from srcs.datasets.privacyalert_graph import PrivacyAlertDataset
# from srcs.datasets.vispr_graph import VisprDataset


#############################################################################
## Wrapper for multiple datasets
class WrapperDatasetsBase(object):
    def __init__(
        self,
        root_dir=".",
        data_dir="",
        num_classes=2,
        fold_id=0,
    ):
        """Initialisation of the Wrapper Class

        If you change the input parameters, you would need to change it also in the
        function load_training_data() of the trainer_base file and in the
        load_testing_data() of the tester base
        """
        super(WrapperDatasetsBase, self).__init__()

        self.root_dir = root_dir
        self.data_dir = data_dir

        self.n_out_classes = num_classes

        self.fold_id = fold_id

    def print_load_set(self, split_mode):
        """ """
        if split_mode == "train":
            print("\nLoading training set ...")

        if split_mode == "val":
            print("\nLoading validation set ...")

        if split_mode == "test":
            print("\nLoading testing set ...")

    def get_training_set(self):
        return self.training_set

    def get_validation_set(self):
        return self.validation_set

    def get_testing_set(self):
        return self.testing_set

    def get_data_split(self, mode="test"):
        """ """
        assert mode in ["train", "val", "test"]

        if mode == "train":
            return self.training_set

        if mode == "val":
            return self.validation_set

        if mode == "test":
            return self.testing_set

    def get_class_weights(self):
        """Compute class weights for weighted loss in training."""
        return self.training_set.get_class_weights()

    def get_dataset_name(self, mode):
        """ """
        data_split = self.get_data_split(mode=mode)
        return data_split.get_name_low()

    def get_data_dir(self):
        return self.data_dir


#############################################################################


class WrapperImageDatasets(WrapperDatasetsBase):
    """Wrapper class to load a dataset"""

    def __init__(
        self,
        root_dir=".",
        data_dir="",
        num_classes=2,
        fold_id=0,
        image_size=448,
    ):
        """Initialisation of the Wrapper Class

        If you change the input parameters, you would need to change it also in the
        function load_training_data() of the trainer_base file and in the
        load_testing_data() of the tester base
        """
        super(WrapperImageDatasets, self).__init__(
            root_dir=root_dir,
            data_dir=data_dir,
            num_classes=num_classes,
            fold_id=fold_id,
        )

        self.image_size = image_size

    def load_split(self, dataset_name, partition, split_mode, _b_filter_imgs):
        """ """
        self.print_load_set(split_mode)

        data_split = ImagePrivacyDatasets(
            repo_dir=self.root_dir,
            data_dir=self.data_dir,
            partition=partition,
            split=split_mode,
            num_classes=self.n_out_classes,
            fold_id=self.fold_id,
            b_filter_imgs=_b_filter_imgs,
            img_size=self.image_size,
            name=dataset_name,
        )

        return data_split

    def load_split_set(
        self,
        dataset_name,
        partition="final",
        mode="train",
        b_filter_imgs=False,
    ):
        """
        - dataset_name
        - mode: string, either train or test
        """
        assert dataset_name in [
            "GIPS",
            "PrivacyAlert",
            "PicAlert",
            "VISPR",
            "IPD",
        ]
        assert partition in ["crossval", "final", "original"]
        assert mode in ["train", "test"]

        if mode == "train":
            training_set = self.load_split(
                dataset_name,
                partition,
                "train",
                b_filter_imgs,
            )

            if (partition == "crossval") | (partition == "original"):
                validation_set = self.load_split(
                    dataset_name, partition, "val", b_filter_imgs
                )

        elif mode == "test":
            testing_set = self.load_split(
                dataset_name, partition, "test", b_filter_imgs
            )

        # Set the class variables (this is important if more datasets are used above)
        if mode == "train":
            self.training_set = training_set

            if (partition == "crossval") | (partition == "original"):
                self.validation_set = validation_set

        elif mode == "test":
            self.testing_set = testing_set

        print()


#############################################################################
## Wrapper for multiple datasets


class WrapperDatasets(object):
    """Wrapper class to load a dataset"""

    def __init__(
        self,
        root_dir=".",
        data_dir="",
        graph_mode="obj_scene",
        num_classes=2,
        fold_id=0,
        n_graph_nodes=447,
        node_feat_size=2,
    ):
        """Initialisation of the Wrapper Class

        If you change the input parameters, you would need to change it also in the
        function load_training_data() of the trainer_base file and in the
        load_testing_data() of the tester base
        """
        super(WrapperDatasets, self).__init__()

        self.root_dir = root_dir
        self.data_dir = data_dir

        self.n_out_classes = num_classes

        self.fold_id = fold_id

        # self.adjacency_filename = adj_mat_fn

        self.graph_mode = graph_mode
        self.n_graph_nodes = n_graph_nodes
        self.node_feature_size = node_feat_size

    def print_load_set(self, split_mode):
        """ """
        if split_mode == "train":
            print("\nLoading training set ...")

        if split_mode == "val":
            print("\nLoading validation set ...")

        if split_mode == "test":
            print("\nLoading testing set ...")

    def load_split_dataset(
        self,
        dataset_name,
        partition,
        split_mode,
        _b_use_card,
        _b_use_conf,
        _b_filter_imgs,
    ):
        """ """
        self.print_load_set(split_mode)

        data_split = GraphImagePrivacyDatasets(
            dataset_name=dataset_name,
            repo_dir=self.root_dir,
            data_dir=self.data_dir,
            partition=partition,
            split=split_mode,
            graph_mode=self.graph_mode,
            num_classes=self.n_out_classes,
            fold_id=self.fold_id,
            # adj_mat_fn=self.adjacency_filename,
            n_graph_nodes=self.n_graph_nodes,
            node_feat_size=self.node_feature_size,
            b_use_card=_b_use_card,
            b_use_conf=_b_use_conf,
            b_filter_imgs=_b_filter_imgs,
        )

        return data_split

    def load_split_gips(
        self, partition, split_mode, _b_use_card, _b_use_conf, _b_filter_imgs
    ):
        """ """

        self.print_load_set(split_mode)

        data_split = GraphImagePrivacySubset(
            repo_dir=self.root_dir,
            data_dir=self.data_dir,
            partition=partition,
            split=split_mode,
            graph_mode=self.graph_mode,
            num_classes=self.n_out_classes,
            fold_id=self.fold_id,
            # adj_mat_fn=self.adjacency_filename,
            n_graph_nodes=self.n_graph_nodes,
            node_feat_size=self.node_feature_size,
            b_use_card=_b_use_card,
            b_use_conf=_b_use_conf,
            b_filter_imgs=_b_filter_imgs,
        )

        return data_split

    def load_split_privacyalert(
        self, partition, split_mode, _b_use_card, _b_use_conf, _b_filter_imgs
    ):
        """ """

        self.print_load_set(split_mode)

        data_split = PrivacyAlertDataset(
            repo_dir=self.root_dir,
            data_dir=self.data_dir,
            partition=partition,
            split=split_mode,
            graph_mode=self.graph_mode,
            num_classes=self.n_out_classes,
            fold_id=self.fold_id,
            # adj_mat_fn=self.adjacency_filename,
            n_graph_nodes=self.n_graph_nodes,
            node_feat_size=self.node_feature_size,
            b_use_card=_b_use_card,
            b_use_conf=_b_use_conf,
            b_filter_imgs=_b_filter_imgs,
        )

        return data_split

    def load_split_picalert(
        self, partition, split_mode, _b_use_card, _b_use_conf, _b_filter_imgs
    ):
        """ """

        self.print_load_set(split_mode)

        data_split = PicAlertDataset(
            repo_dir=self.root_dir,
            data_dir=self.data_dir,
            partition=partition,
            split=split_mode,
            graph_mode=self.graph_mode,
            num_classes=self.n_out_classes,
            fold_id=self.fold_id,
            # adj_mat_fn=self.adjacency_filename,
            n_graph_nodes=self.n_graph_nodes,
            node_feat_size=self.node_feature_size,
            b_use_card=_b_use_card,
            b_use_conf=_b_use_conf,
            b_filter_imgs=_b_filter_imgs,
        )

        return data_split

    def load_split_vispr(
        self, partition, split_mode, _b_use_card, _b_use_conf, _b_filter_imgs
    ):
        """ """

        self.print_load_set(split_mode)

        data_split = VisprDataset(
            repo_dir=self.root_dir,
            data_dir=self.data_dir,
            partition=partition,
            split=split_mode,
            graph_mode=self.graph_mode,
            num_classes=self.n_out_classes,
            fold_id=self.fold_id,
            # adj_mat_fn=self.adjacency_filename,
            n_graph_nodes=self.n_graph_nodes,
            node_feat_size=self.node_feature_size,
            b_use_card=_b_use_card,
            b_use_conf=_b_use_conf,
            b_filter_imgs=_b_filter_imgs,
        )

        return data_split

    def load_split_graphipd(
        self, partition, split_mode, _b_use_card, _b_use_conf, _b_filter_imgs
    ):
        """ """

        self.print_load_set(split_mode)

        data_split = GraphImagePrivacyDataset(
            repo_dir=self.root_dir,
            data_dir=self.data_dir,
            partition=partition,
            split=split_mode,
            graph_mode=self.graph_mode,
            num_classes=self.n_out_classes,
            fold_id=self.fold_id,
            # adj_mat_fn=self.adjacency_filename,
            n_graph_nodes=self.n_graph_nodes,
            node_feat_size=self.node_feature_size,
            b_use_card=_b_use_card,
            b_use_conf=_b_use_conf,
            b_filter_imgs=_b_filter_imgs,
        )

        return data_split

    def load_split_set(
        self,
        dataset_name,
        partition="final",
        mode="train",
        b_use_card=True,
        b_use_conf=False,
        b_filter_imgs=False,
    ):
        """

        - dataset_name
        - mode: string, either train or test
        """
        assert dataset_name in [
            "GIPS",
            "PrivacyAlert",
            "PicAlert",
            "VISPR",
            "IPD",
        ]
        assert partition in ["crossval", "final", "original"]
        assert mode in ["train", "test"]

        # PicAlert!
        if dataset_name == "PicAlert":
            if mode == "train":
                training_set = self.load_split_picalert(
                    partition, "train", b_use_card, b_use_conf, b_filter_imgs
                )

                if partition == "crossval":
                    validation_set = self.load_split_picalert(
                        partition, "val", b_use_card, b_use_conf, b_filter_imgs
                    )

            elif mode == "test":
                testing_set = self.load_split_picalert(
                    partition, "test", b_use_card, b_use_conf, b_filter_imgs
                )

        # PicAlert!
        if dataset_name == "VISPR":
            if mode == "train":
                training_set = self.load_split_vispr(
                    partition, "train", b_use_card, b_use_conf, b_filter_imgs
                )

                if (partition == "crossval") | (partition == "original"):
                    validation_set = self.load_split_vispr(
                        partition, "val", b_use_card, b_use_conf, b_filter_imgs
                    )

            elif mode == "test":
                testing_set = self.load_split_vispr(
                    partition, "test", b_use_card, b_use_conf, b_filter_imgs
                )

        # Privacy Alert dataset
        if dataset_name == "PrivacyAlert":
            if mode == "train":
                training_set = self.load_split_privacyalert(
                    partition, "train", b_use_card, b_use_conf, b_filter_imgs
                )

                if (partition == "crossval") | (partition == "original"):
                    validation_set = self.load_split_privacyalert(
                        partition, "val", b_use_card, b_use_conf, b_filter_imgs
                    )

            elif mode == "test":
                testing_set = self.load_split_privacyalert(
                    partition, "test", b_use_card, b_use_conf, b_filter_imgs
                )

        # Graph Image Privacy Subset
        if dataset_name == "GIPS":
            if mode == "train":
                training_set = self.load_split_gips(
                    partition, "train", b_use_card, b_use_conf, b_filter_imgs
                )

                if (partition == "crossval") | (partition == "original"):
                    validation_set = self.load_split_gips(
                        partition, "val", b_use_card, b_use_conf, b_filter_imgs
                    )

            elif mode == "test":
                testing_set = self.load_split_gips(
                    partition, "test", b_use_card, b_use_conf, b_filter_imgs
                )

        # Image Privacy dataset
        if dataset_name == "IPD":
            if mode == "train":
                training_set = self.load_split_graphipd(
                    partition, "train", b_use_card, b_use_conf, b_filter_imgs
                )

                if partition == "crossval":
                    validation_set = self.load_split_graphipd(
                        partition, "val", b_use_card, b_use_conf, b_filter_imgs
                    )

            elif mode == "test":
                testing_set = self.load_split_graphipd(
                    partition, "test", b_use_card, b_use_conf, b_filter_imgs
                )

        # Set the class variables (this is important if more datasets are used above)
        if mode == "train":
            self.training_set = training_set

            if (partition == "crossval") | (partition == "original"):
                self.validation_set = validation_set

        elif mode == "test":
            self.testing_set = testing_set

        print()

    def get_training_set(self):
        return self.training_set

    def get_validation_set(self):
        return self.validation_set

    def get_testing_set(self):
        return self.testing_set

    def get_data_split(self, mode="test"):
        """ """
        assert mode in ["train", "val", "test"]

        if mode == "train":
            return self.training_set

        if mode == "val":
            return self.validation_set

        if mode == "test":
            return self.testing_set

    def get_class_weights(self):
        """Compute class weights for weighted loss in training."""
        return self.training_set.get_class_weights()

    def get_dataset_name(self, mode):
        """ """
        data_split = self.get_data_split(mode=mode)
        return data_split.get_name_low()

    def get_data_dir(self):
        return self.data_dir

    def get_all_image_names(self, dataset_name):
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

        if dataset_name == "GIPS":
            img_name = df["Image Name"].astype(str)

            img_list = img_name.values

        elif dataset_name == "PicAlert":
            img_name = df["Image Name"]

            # As filenames alone cannot help retrieve the images, we append the
            # full filepath and extension. We therefore create an updated list
            json_out = os.path.join(
                self.data_dir, "annotations", "ipd_imgs_curated.json"
            )
            ipd_curated = json.load(open(json_out, "r"))
            annotations = ipd_curated["annotations"]
            l_img_ann = [
                x["image"] for x in annotations if x["dataset"] == dataset_name
            ]
            l_data_prefix = [
                ipd_curated["datasets"][x["dataset"]]["data_dir"]
                for x in annotations
                if x["dataset"] == dataset_name
            ]

            new_data_prefix_l = []
            new_img_l = []
            for idx, img in enumerate(tqdm(img_name.values)):
                if img in l_img_ann:
                    idx2 = l_img_ann.index(img)
                    full_img_path = annotations[idx2]["fullpath"]
                    new_img_l.append(full_img_path)

            img_list = new_img_l

        elif dataset_name == "VISPR":
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
