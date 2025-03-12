#!/usr/bin/env python
#
# Evaluation toolkit for image privacy
#
################################################################################
# Authors:
# - Alessio Xompero
#
# Email: a.xompero@qmul.ac.uk
#
#  Created Date: 2023/02/09
# Modified Date: 2023/09/20
#
# ----------------------------------------------------------------------------

import argparse
import inspect
import os
import sys
from datetime import datetime

# setting path
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.insert(0, pp_dir)

import numpy as np
import pandas as pd

import csv

from srcs.perfmeas_tracker import PerformanceMeasuresTracker
from srcs.datasets.wrapper import WrapperDatasets


# ----------------------------------------------------------------------------


class EvaluationToolkit:
    """ """

    def __init__(self, args):
        self.dataset = args.dataset

        self.model_results_csv = args.model_results_csv

        self.model_name = args.model_name
        self.model_mode = args.model_mode

        self.out_file = args.out_file

        self.repo_dir = args.root_dir
        self.data_dir = args.data_dir

        self.n_cls = args.n_out_classes

        self.partition = args.training_mode
        self.split_mode = args.split_mode
        self.fold_id = args.fold_id

        self.perf_measures = PerformanceMeasuresTracker(
            dataset=self.dataset, n_cls=self.n_cls
        )
        self.perf_measures.set_beta(args.beta)

        self.load_annotations(args.b_filter_imgs)

    def load_annotations(self, _b_filter_imgs=False):
        """Load into memory the lables of the training set."""
        data_wrapper = WrapperDatasets(
            root_dir=self.repo_dir,
            data_dir=self.data_dir,
            num_classes=self.n_cls,
            fold_id=self.fold_id,
        )

        data_wrapper.load_split_set(
            self.dataset,
            partition=self.partition,
            mode="train" if self.split_mode == "val" else self.split_mode,
            b_filter_imgs=_b_filter_imgs,
        )

        data_split = data_wrapper.get_data_split(self.split_mode)
        self.annotations = np.array(data_split.get_labels())

    def get_headers_binary(self):
        headers = [
            "date",
            "model name",
            "model_mode",
            "dataset",
            "partition",
            "fold",
            "split",
            "P_0",
            "R_0",
            "F1_0",
            "P_1",
            "R_1",
            "F1_1",
            "P",
            "R (BA)",
            "ACC",
            "MF1",
            "wF1",
            # "BA",
            # "Beta_F1",
        ]

        return headers

    def get_model_res_binary(self):
        model_res = [
            datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            self.model_name,
            self.model_mode,
            self.dataset,
            self.partition,
            self.fold_id,
            self.split_mode,
            self.perf_measures.get_measure("precision_0"),
            self.perf_measures.get_measure("recall_0"),
            self.perf_measures.get_measure("f1_score_0"),
            self.perf_measures.get_measure("precision_1"),
            self.perf_measures.get_measure("recall_1"),
            self.perf_measures.get_measure("f1_score_1"),
            self.perf_measures.get_measure("precision"),
            self.perf_measures.get_measure("recall"),
            self.perf_measures.get_measure("accuracy"),
            self.perf_measures.get_measure("macro_f1_score"),
            self.perf_measures.get_measure("weighted_f1_score"),
            # self.perf_measures.get_measure("balanced_accuracy"),
        ]

        return model_res

    def get_headers_ternary(self):
        headers = [
            "model name",
            "P_0",
            "R_0",
            "F1_0",
            "P_1",
            "R_1",
            "F1_1",
            "P_2",
            "R_2",
            "F1_2",
            "P",
            "R",
            "MF1",
            "wF1",
            "ACC",
            "BA",
        ]

        return headers

    def get_model_res_ternary(self):
        model_res = [
            self.model_name,
            self.perf_measures.get_measure("precision_0"),
            self.perf_measures.get_measure("recall_0"),
            self.perf_measures.get_measure("f1_score_0"),
            self.perf_measures.get_measure("precision_1"),
            self.perf_measures.get_measure("recall_1"),
            self.perf_measures.get_measure("f1_score_1"),
            self.perf_measures.get_measure("precision_2"),
            self.perf_measures.get_measure("recall_2"),
            self.perf_measures.get_measure("f1_score_2"),
            self.perf_measures.get_measure("precision"),
            self.perf_measures.get_measure("recall"),
            self.perf_measures.get_measure("macro_f1_score"),
            self.perf_measures.get_measure("weighted_f1_score"),
            self.perf_measures.get_measure("accuracy"),
            self.perf_measures.get_measure("balanced_accuracy"),
        ]

        return model_res

    def get_headers_quinary(self):
        headers = [
            "model name",
            "P_0",
            "R_0",
            "F1_0",
            "P_1",
            "R_1",
            "F1_1",
            "P_2",
            "R_2",
            "F1_2",
            "P_3",
            "R_3",
            "F1_3",
            "P_4",
            "R_4",
            "F1_4",
            "BA",
            "wF1",
            "Beta_F1",
        ]

        return headers

    def get_model_res_quinary(self):
        model_res = [
            datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            self.model_name,
            self.model_mode,
            self.dataset,
            self.partition,
            self.fold_id,
            self.split_mode,
            self.perf_measures.get_measure("precision_clearly_public"),
            self.perf_measures.get_measure("recall_clearly_public"),
            self.perf_measures.get_measure("f1_score_clearly_public"),
            self.perf_measures.get_measure("precision_public"),
            self.perf_measures.get_measure("recall_public"),
            self.perf_measures.get_measure("f1_score_public"),
            self.perf_measures.get_measure("precision_undecidable"),
            self.perf_measures.get_measure("recall_undecidable"),
            self.perf_measures.get_measure("f1_score_undecidable"),
            self.perf_measures.get_measure("precision_private"),
            self.perf_measures.get_measure("recall_private"),
            self.perf_measures.get_measure("f1_score_private"),
            self.perf_measures.get_measure("precision_clearly_private"),
            self.perf_measures.get_measure("recall_clearly_private"),
            self.perf_measures.get_measure("f1_score_clearly_private"),
            # self.perf_measures.get_measure("balanced_accuracy"),
            self.perf_measures.get_measure("weighted_f1_score"),
            # self.perf_measures.get_measure("beta_f1_score"),
        ]

        return model_res

    def get_headers(self):
        if self.n_cls == 2:
            headers = self.get_headers_binary()

        if self.n_cls == 3:
            headers = self.get_headers_ternary()

        if self.n_cls == 5:
            headers = self.get_headers_quinary()

        return headers

    def get_model_res(self):
        if self.n_cls == 2:
            model_res = self.get_model_res_binary()

        if self.n_cls == 3:
            model_res = self.get_model_res_ternary()

        if self.n_cls == 5:
            model_res = self.get_model_res_quinary()

        return model_res

    def save_model_res_to_csv(self, m_res):
        model_res = []

        for x in m_res[:7]:
            model_res.append(x)

        for x in m_res[7:]:
            model_res.append(
                "{:.2f}".format(x * 100)
            )  # make performance in percentages

        if os.path.exists(self.out_file):
            fh = open(self.out_file, "a")

            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = csv.writer(fh)

            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(model_res)

            # Close the file object
            fh.close()

        else:
            headers = self.get_headers()

            fh = open(self.out_file, "w")

            writer_object = csv.writer(fh)

            writer_object.writerow(headers)
            writer_object.writerow(model_res)

            # Close the file object
            fh.close()

    def run(self):
        # Read submission
        es = pd.read_csv(self.model_results_csv, sep=",", index_col=False)

        preds = es["pred_class"].values

        self.perf_measures.compute_all_metrics(self.annotations, preds)

        model_res = self.get_model_res()

        self.save_model_res_to_csv(model_res)

        print("Performance metrics saved in " + self.out_file)


def GetParser():
    parser = argparse.ArgumentParser(
        description="Image Privacy Classification - Evaluation Toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model_results_csv", default="random.csv", type=str)

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["IPD", "PrivacyAlert", "GIPS", "PicAlert", "VISPR"],
        required=True,
    )
    parser.add_argument("--model_name", default="random", type=str)
    parser.add_argument(
        "--model_mode", type=str, required=True, choices=["best", "last"]
    )
    parser.add_argument("--beta", default=2.0, type=float)

    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--root_dir", default=".", type=str)
    parser.add_argument("--data_dir", default=".", type=str)

    parser.add_argument("--n_out_classes", default=2, type=int, choices=[2])

    parser.add_argument(
        "--training_mode",
        type=str,
        choices=["final", "crossval", "original"],
        required=True,
        help="Choose to run K-fold cross-validation or train the final model (full training set without validation split)",
    )
    parser.add_argument("--fold_id", type=int, default=-1)

    parser.add_argument(
        "--split_mode",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Choose the data split for the evaluation (training, validation, testing)!",
    )

    if sys.version_info[0] >= 3:
        if sys.version_info[1] < 9:
            parser.add_argument(
                "--b_filter_imgs",
                action="store_true",
                help="Force to use binary cross-entropy.",
            )
            parser.set_defaults(feature=False)
        else:
            parser.add_argument(
                "--b_filter_imgs", action=argparse.BooleanOptionalAction
            )
    else:
        parser.add_argument(
            "--b_filter_imgs",
            action="store_true",
            help="Force to use binary cross-entropy.",
        )
        parser.add_argument(
            "--no-b_filter_imgs", dest="b_filter_imgs", action="store_false"
        )
        parser.set_defaults(feature=False)

    return parser


if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    eval_toolkit = EvaluationToolkit(args)
    eval_toolkit.run()
