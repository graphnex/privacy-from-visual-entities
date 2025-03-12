#!/usr/bin/env python
#
# Brief description here
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/21
# Modified Date: 2023/09/21
#
# MIT License

# Copyright (c) 2023 GraphNEx

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
# -----------------------------------------------------------------------------

import argparse  # Parser for command-line options, arguments and sub-commands
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # smart progress meter for loops


##############################################################################
class Splitter(object):
    """Class to prepare the annotation file with data splits for the dataset."""

    def __init__(self, repo_dir, data_dir, n_out_cls=2):
        """ """
        self.repo_dir = repo_dir
        self.data_dir = data_dir
        self.dataset = dataset

        self.n_out_cls = n_out_cls

        self.get_annotations()

    def get_annotations(self):
        """Read annotation files (CSV manifest) and set class variables.

        Information/variables read:
            - self.imgs:    list of image filenames
            - self.labels:  list of annotation labels (for each corresponding image)
        """
        df = pd.read_csv(
            os.path.join(
                self.data_dir,
                "annotations",
                "labels.csv",
            ),
            delimiter=",",
            index_col=False,
        )

        img_name = df["Image Name"].values
        labels = df["Label {:d}-class".format(self.n_out_cls)].values

        self.imgs = img_name
        self.labels = labels.tolist()

    def create_annotation_file(self, K=5, ratio_split=0.30):
        """Prepare the annotation file with the pre-defined format.

        The GIPS dataset is curated to prepare an annotation file with the
        following fields:
            - ID: unique identifcation number (int) for each sample in the data
            - Image name: name of the sample (str).
            - Label 2-class: label of the privacy classes
                (0:'Primary Tumor', 1:'Solid Tissue Normal') annotated for each
                sample (int). This column is only for binary classification.
            - Final: label of the data split that each sample is associated to
                for the training and testing split for the final model to train (int).
                Values are 0: training set, 1: validation set, 2: testing set.
            - Fold 0: label of the data split that each sample is associated to
                for the first fold in a K-Fold Cross-Validation (int).
                Values are 0: training set, 1: validation set, 2: testing set.
            - Fold 1: label of the data split that each sample is associated to
                for the second fold in a K-Fold Cross-Validation (int).
                Values are 0: training set, 1: validation set, 2: testing set.
            ...

        Tumor labels and data splits (training, validation, testing sets) are annotated
        in the annotation file in comma-based CSV format. The annotation file is
        saved into the directory of the data (self.data_dir).

        The function takes as input
            - K: the number of folds for the K-fold cross-validation,
            - ratio_split: the ratio to split the testing set from the training set.

        The dataset is split into training and testing splits with random shuffling (but
        fixed random state for replicability). This first split will be used for the
        final training of the model after K-Fold Cross-Validation. The training set is
        then split into K consecutive folds (without shuffling). The suffling of the folds
        will be handled by the data loader in the trainer and tester classes.
        """
        cols_header = ["ID", "Image Name", "Label 2-class"]

        ids = []
        img_names = []
        labels = []

        for idx, ID in enumerate(tqdm(self.imgs, ascii=True)):
            ids.append(idx)
            img_names.append(ID)
            labels.append(self.labels[idx])

        zipped = list(zip(ids, img_names, labels))

        # Create Pandas dataframe with list of values for each column
        df = pd.DataFrame(zipped, columns=cols_header)

        # Split the data into train and test
        ids_train, ids_test = train_test_split(
            df["ID"].values, test_size=ratio_split, random_state=43
        )

        df["Final"] = -1
        df.loc[np.sort(ids_train), "Final"] = 0
        df.loc[np.sort(ids_test), "Final"] = 2

        df_train = df.loc[np.sort(ids_train), :].copy()
        df_test = df.loc[np.sort(ids_test), :].copy()
        df_train_labels = np.array(labels)[np.sort(ids_train)]

        assert (
            df_train_labels == df.loc[np.sort(ids_train), "Label 2-class"]
        ).all()

        if K > 0:
            kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=43)

            for fold, (train_ids, val_ids) in enumerate(
                kfold.split(df_train, df_train_labels)
            ):
                df["Fold {:d}".format(fold)] = -1

                train_tmp_ids = df_train.iloc[train_ids]["ID"].values
                val_tmp_ids = df_train.iloc[val_ids]["ID"].values

                df.loc[train_tmp_ids, "Fold {:d}".format(fold)] = 0
                df.loc[val_tmp_ids, "Fold {:d}".format(fold)] = 1
                df.loc[ids_test, "Fold {:d}".format(fold)] = 2

        # Save dataframe to file
        df.to_csv(
            os.path.join(
                self.data_dir,
                "annotations",
                "labels_splits.csv",
            ),
            index=False,
            sep=",",
        )


def GetParser():
    parser = argparse.ArgumentParser(
        description="Prepare PrivacyAlert dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--n_folds", type=int)
    parser.add_argument("--test_ratio_split", type=float, default=0.3)

    return parser


if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))
    print()

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    data_splitter = Splitter(
        repo_dir=args.root_dir,
        data_dir=args.data_dir,
    )

    data_splitter.create_annotation_file(args.n_folds, args.test_ratio_split)

    print("Finished!")
