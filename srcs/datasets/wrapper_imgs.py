#!/usr/bin/env python
#
# Brief description here
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/06/29
# Modified Date: 2023/09/05
#
# -----------------------------------------------------------------------------

from srcs.datasets.imageprivacy import ImagePrivacyDatasets


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
                dataset_name, validation_set = self.load_split(
                    partition, "val", b_filter_imgs
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
