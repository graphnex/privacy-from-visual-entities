#!/usr/bin/env python
#
# Parent/base class for training different machine learning models.
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/01/30
# Modified Date: 2025/01/30
#
# MIT License

# Copyright (c) 2023-2025 GraphNEx

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

import os
import sys
import time

import numpy as np

np.set_printoptions(threshold=sys.maxsize, precision=2)

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

# Package modules
from srcs.load_net import gnn_model
from srcs.graph.prior_graph import get_prior_graph
from srcs.perfmeas_tracker import PerformanceMeasuresTracker, AverageMeter
from srcs.datasets.wrapper import WrapperDatasets, WrapperImageDatasets

from srcs.utils import (
    save_model,
    device,
    n_gpus,
    print_model_parameters,
    crossval_stats_summary,
    plot_learning_curves,
    set_seed,
)

from srcs.logging_gnex import Logging

from pdb import set_trace as bp  # This is only for debugging


#
#############################################################################
# Base Class for the training of a model
#
class TrainerBaseClass(object):
    def __init__(self, config, args):
        # Paths
        self.root_dir = config["paths"]["root_dir"]
        self.out_dir = os.path.join(
            self.root_dir,
            "trained_models",
            # use_case_dir,
            args.dataset.lower(),
        )
        self.data_dir = os.path.join(
            config["paths"]["data_prefix"],
            config["datasets"][args.dataset]["data_dir"],
        )

        # Training parameters
        self.params = config["params"]

        # Boolean for using binary cross-entropy loss
        self.b_bce = args.use_bce

        # Boolean for using the weighted loss
        self.b_use_weight_loss = self.params["weight_loss"]
        print("Use of weight loss: {:s}".format(str(self.b_use_weight_loss)))

        self.b_use_wandb = True if "use_wandb" in args else False

        self.resume = self.params["resume"]
        self.resume_measure = self.params["measure"]

        self.num_workers = config["num_workers"]

        # --------------------------------------------
        # Model/network parameters
        self.n_out_classes = config["net_params"]["num_out_classes"]

        self.net = None

    def initialise_checkpoint_dir(self, model_name, n_out_classes):
        """ """
        checkpoint_dir = os.path.join(
            self.out_dir,
            "{:d}-class".format(n_out_classes),
            model_name,
        )
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoint_dir = checkpoint_dir

    def initialise_performance_trackers(self, dataset, n_out_classes):
        """Monitor the performance measures (classes)."""
        self.train_metrics_tracker = PerformanceMeasuresTracker(
            dataset=dataset, n_cls=n_out_classes
        )
        self.val_metrics_tracker = PerformanceMeasuresTracker(
            dataset=dataset, n_cls=n_out_classes
        )
        self.best_metrics_tracker = PerformanceMeasuresTracker(
            dataset=dataset, n_cls=n_out_classes
        )

    def configure_optimizer(self, net, config):
        """ """

        if config["params"]["optimizer"] == "SGD":
            self.optimizer = getattr(optim, config["params"]["optimizer"])(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=config["params"]["init_lr"],
                weight_decay=config["params"]["weight_decay"],
                momentum=config["params"]["momentum"],
            )

            self.scheduler = lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=config["params"]["init_lr"],
                max_lr=config["params"]["max_lr"],
                mode="triangular",
                cycle_momentum=False,
            )

            self.optimizer_name = "SGD"
            self.scheduler_name = "CyclicLR"

        elif config["params"]["optimizer"] == "Adam":
            # Configuration based on Benchmarking GNNs -
            # SuperPixel Graph Classification CIFAR10
            # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/configs/superpixels_graph_classification_GCN_CIFAR10_100k.json

            self.optimizer = getattr(optim, config["params"]["optimizer"])(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=config["params"]["init_lr"],
                weight_decay=config["params"]["weight_decay"],
            )

            if config["params"]["training_mode"] == "final":
                optim_mode = "max"
            else:
                optim_mode = "min"

            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=optim_mode,
                factor=config["params"]["lr_reduce_factor"],
                patience=config["params"]["lr_schedule_patience"],
                verbose=True,
            )

            self.optimizer_name = "Adam"
            self.scheduler_name = "ReduceLROnPlateau"
        else:
            self.optimizer = getattr(optim, config["params"]["optimizer"])(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=config["params"]["init_lr"],
                weight_decay=config["params"]["weight_decay"],
            )

            self.scheduler = lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=config["params"]["init_lr"],
                max_lr=config["params"]["max_lr"],
                mode="triangular",
                cycle_momentum=False,
            )

            self.optimizer_name = config["params"]["optimizer"]
            self.scheduler_name = "CyclicLR"

    def get_filename(self, model_name, extension=".csv", prefix="", suffix=""):
        """Create a filename based on the fold ID and model name.

        The function returns the filename with any additional prefix appended,
        and the extension based the argument passed (.csv as default).
        """
        if self.params["training_mode"] == "crossval":
            filename = "{:s}-{:d}".format(model_name, self.params["fold_id"])

        if self.params["training_mode"] == "final":
            filename = "{:s}-final".format(model_name)

        if self.params["training_mode"] == "original":
            filename = "{:s}-original".format(model_name)

        filename = prefix + filename + suffix + extension

        return filename

    def initialise_log(self, model_name, n_out_classes):
        """Create training log and initialise the preamble."""
        filename = self.get_filename(
            model_name, extension=".txt", suffix="_training_log"
        )

        self.log = Logging()
        self.log.initialise(os.path.join(self.checkpoint_dir, filename))
        self.log.write_preamble(model_name, n_out_classes)
        self.log.write_training_parameters(
            self.params["batch_size_train"],
            self.params["batch_size_val"],
            self.params["max_num_epochs"],
            self.params["fold_id"],
            self.params["training_mode"],
        )
        self.log.write_line()

    def compute_and_save_info(
        self, perf_tracker, cm_pred, cm_targets, mode="epoch"
    ):
        perf_tracker.compute_all_metrics(cm_targets, cm_pred)

        assert mode in ["batch", "epoch"]
        if mode == "batch":
            self.log.write_batch_info(perf_tracker)
        elif mode == "epoch":
            perf_tracker.print_metrics()
            perf_tracker.write_metrics_to_log(self.log.get_log())

    def compare_metrics_and_save_model(self, model_name, epoch, es):
        """ """
        ext = ".pth"
        if ((epoch % 1) == 0) or (epoch == self.params["max_num_epochs"] - 1):
            es += 1

            eval_measures = [
                "balanced_accuracy",
                "weighted_f1_score",
                "accuracy",
                "macro_f1_score",
            ]

            assert self.params["measure"] in eval_measures
            measure = self.params["measure"]

            val_measure = self.val_metrics_tracker.get_measure(measure)

            if self.best_metrics_tracker.is_classification_report():
                best_measure = self.best_metrics_tracker.get_measure(measure)
            else:
                best_measure = 0
                self.best_metrics_tracker.set_classification_report(
                    self.val_metrics_tracker.get_classification_report()
                )

            if val_measure >= best_measure:
                self.best_metrics_tracker.set_measure(measure, val_measure)

                tmp_str = "model at epoch: {:d}\n".format(epoch)

                if measure == "balanced_accuracy":
                    filename = "acc"
                    print("\nSaved best balanced accuracy (%) " + tmp_str)
                    self.log.write_info(
                        "\nSaved best balanced accuracy (%) " + tmp_str
                    )

                if measure == "weighted_f1_score":
                    filename = "weighted_f1"
                    self.log.write_info("\nBest W-F1 " + tmp_str)

                if measure == "accuracy":
                    filename = "acc"
                    print("\nSaved best UBA(%) " + tmp_str)
                    self.log.write_info("\nSaved best UBA(%) " + tmp_str)

                if measure == "macro_f1_score":
                    filename = "macro_f1"
                    self.log.write_info("\nBest UW-F1 " + tmp_str)

                save_model(
                    self.net,
                    val_measure,
                    self.checkpoint_dir,
                    self.get_filename(model_name, ext, filename + "_"),
                    mode="best",
                    epoch=epoch,
                )

                self.log.write_info("Saved model at epoch {:d}".format(epoch))

                es = 0

        return es

    def set_performance_measures_best_epoch(self, best_epoch):
        """ """
        best_metrics = {
            "Fold id": self.params["fold_id"],
            "Epoch": best_epoch,
            "P-T": self.train_metrics_tracker.get_precision_overall() * 100,
            "P-V": self.val_metrics_tracker.get_precision_overall() * 100,
            "BA-T": self.train_metrics_tracker.get_balanced_accuracy() * 100,
            "BA-V": self.val_metrics_tracker.get_balanced_accuracy() * 100,
            # "R-T": self.train_metrics_tracker.get_recall_overall() * 100,
            # "R-V": self.val_metrics_tracker.get_recall_overall() * 100,
            "UBA-T": self.train_metrics_tracker.get_accuracy() * 100,
            "UBA-V": self.val_metrics_tracker.get_accuracy() * 100,
            "wF1-T": self.train_metrics_tracker.get_weighted_f1score() * 100,
            "wF1-V": self.val_metrics_tracker.get_weighted_f1score() * 100,
            "MF1-T": self.train_metrics_tracker.get_macro_f1_score() * 100,
            "MF1-V": self.val_metrics_tracker.get_macro_f1_score() * 100,
        }

        return best_metrics

    def save_best_epoch_measures(self, best_metrics, model_name):
        """Save to file the performance measures at the best epoch at the end
        of the training.
        """

        fullpathname = os.path.join(
            self.checkpoint_dir, model_name + "_crossval_best.txt"
        )

        if os.path.isfile(fullpathname):
            fh = open(fullpathname, "a")
        else:
            fh = open(fullpathname, "w")
            fh.write(
                "Fold\tEpoch\tP-T\tP-V\tBA-T\tBA-V\tUBA-T\tUBA-V\twF-T\twF-V\tMF-T\tMF-V\n"
            )
            fh.flush()

        fh.write(
            "{:d}\t{:3d}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\n".format(
                best_metrics["Fold id"],
                best_metrics["Epoch"],
                best_metrics["P-T"],
                best_metrics["P-V"],
                best_metrics["BA-T"],
                best_metrics["BA-V"],
                best_metrics["UBA-T"],
                best_metrics["UBA-V"],
                best_metrics["wF1-T"],
                best_metrics["wF1-V"],
                best_metrics["MF1-T"],
                best_metrics["MF1-V"],
            )
        )
        fh.flush()

        fh.close()

        # Save stats of the cross-validation using the resume_measure
        out_stats_fn = os.path.join(
            self.checkpoint_dir, model_name + "_crossval_stats.csv"
        )
        crossval_stats_summary(fullpathname, out_stats_fn, self.resume_measure)

    def train_and_val(self):
        t0 = time.time()

        start_epoch = 0
        es = 0
        file_mode = "w"

        ext = ".pth"

        model_name = self.net.get_model_name()

        num_epochs = self.params["max_num_epochs"]

        # acc_train = []
        # acc_val = []

        # Resume training for the new learning rate #
        if self.resume:
            start_epoch = self.load_checkpoint()
            file_mode = "a"

        filename = self.get_filename(
            model_name, suffix="_learning_curve", extension=".txt"
        )
        fh = open("{}/{}".format(self.checkpoint_dir, filename), file_mode)
        fh.write(
            "Epoch\tloss-T\tloss-V\tP-T\tP-V\tBA-T\tBA-V\tUBA-T\tUBA-V\twF-T\twF-V\tMF-T\tMF-V\n"
        )
        fh.flush()

        best_metrics = None
        for epoch in range(start_epoch, num_epochs):
            print("\nEpoch: {:d}/{:d}".format(epoch + 1, num_epochs))
            self.log.write_info("Epoch: {:d}/{:d}\n".format(epoch, num_epochs))
            sys.stdout.flush()

            # if model_name in ["s2p"]:
            #     loss_value = self.train_cnn_epoch(epoch)
            #     val_loss_value = self.val_cnn_epoch()
            # else:
            #     loss_value = self.train_epoch(epoch)
            #     val_loss_value = self.val_epoch()

            loss_value = self.train_epoch(epoch)
            val_loss_value = self.val_epoch()

            es = self.compare_metrics_and_save_model(model_name, epoch, es)

            if self.scheduler_name == "CyclicLR":
                self.scheduler.step()

                my_lr = self.scheduler.get_last_lr()[
                    0
                ]  # This is only for CyclingLR
                print("Learning rate: {:.6f}".format(my_lr))
            elif self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(val_loss_value)

            if es == 0:
                best_metrics = self.set_performance_measures_best_epoch(epoch)

            fh.write(
                "{:3d}\t{:9.6f}\t{:9.6f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\n".format(
                    epoch,
                    loss_value,
                    val_loss_value,
                    self.train_metrics_tracker.get_precision_overall() * 100,
                    self.val_metrics_tracker.get_precision_overall() * 100,
                    self.train_metrics_tracker.get_balanced_accuracy() * 100,
                    self.val_metrics_tracker.get_balanced_accuracy() * 100,
                    self.train_metrics_tracker.get_accuracy() * 100,
                    self.val_metrics_tracker.get_accuracy() * 100,
                    self.train_metrics_tracker.get_weighted_f1score() * 100,
                    self.val_metrics_tracker.get_weighted_f1score() * 100,
                    self.train_metrics_tracker.get_macro_f1_score() * 100,
                    self.val_metrics_tracker.get_macro_f1_score() * 100,
                )
            )
            fh.flush()

            if self.b_use_wandb:
                import wandb

                wandb.log(
                    {
                        "train_loss": loss_value,
                        "val_loss": val_loss_value,
                        "recall_cls0_train": self.train_metrics_tracker.get_measure(
                            "recall_0"
                        )
                        * 100,
                        "recall_cls0_val": self.val_metrics_tracker.get_measure(
                            "recall_0"
                        )
                        * 100,
                        "recall_cls1_train": self.train_metrics_tracker.get_measure(
                            "recall_1"
                        )
                        * 100,
                        "recall_cls1_val": self.val_metrics_tracker.get_measure(
                            "recall_1"
                        )
                        * 100,
                        "precision_train": self.train_metrics_tracker.get_precision_overall()
                        * 100,
                        "precision_val": self.val_metrics_tracker.get_precision_overall()
                        * 100,
                        "balanced_accuracy_train": self.train_metrics_tracker.get_balanced_accuracy()
                        * 100,
                        "balanced_accuracy_val": self.val_metrics_tracker.get_balanced_accuracy()
                        * 100,
                        "accuracy_train": self.train_metrics_tracker.get_accuracy()
                        * 100,
                        "accuracy_val": self.val_metrics_tracker.get_accuracy()
                        * 100,
                    }
                )

            if self.optimizer.param_groups[0]["lr"] < self.params["min_lr"]:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            # Stop training after params['max_time'] hours
            if time.time() - t0 > self.params["max_time"] * 3600:
                print("-" * 89)
                print(
                    "Max_time for training elapsed {:.2f} hours, so stopping".format(
                        self.params["max_time"]
                    )
                )
                break

            # n_epochs_flag = 30
            # if es >= n_epochs_flag:
            #     # print("Loss did not decrease for the last {:d} epochs!".format(n_epochs_flag))
            #     print(
            #         "Accuracy did not improve for the last {:d} epochs!".format(
            #             n_epochs_flag
            #         )
            #     )
            #     if epoch > min(self.num_epochs, n_epochs_flag):
            #         break

        fh.close()

        if best_metrics:
            self.save_best_epoch_measures(best_metrics, model_name)

        save_model(
            self.net,
            self.params["measure"],
            self.checkpoint_dir,
            self.get_filename(model_name, ext, prefix="acc_"),
            mode="last",
            epoch=epoch,
        )
        self.log.write_info("Saved model at last epoch {:d}".format(epoch))

        outfilename = self.get_filename(
            model_name, suffix="_learning_curve", extension="_acc.png"
        )
        plot_learning_curves(
            self.checkpoint_dir,
            filename,
            outfilename,
            self.params["measure"],
            mode=self.params["training_mode"],
        )

        outfilename = self.get_filename(
            model_name, suffix="_learning_curve", extension="_loss.png"
        )
        plot_learning_curves(
            self.checkpoint_dir,
            filename,
            outfilename,
            "loss",
            mode=self.params["training_mode"],
        )

    def train_final_model(self):
        t0 = time.time()

        start_epoch = 0
        file_mode = "w"

        ext = ".pth"

        model_name = self.net.get_model_name()

        num_epochs = self.params["max_num_epochs"]

        # Resume training for the new learning rate #
        if self.resume:
            start_epoch = self.load_checkpoint()
            file_mode = "a"

        filename = self.get_filename(
            model_name, suffix="_learning_curve", extension=".txt"
        )
        fh = open("{}/{}".format(self.checkpoint_dir, filename), file_mode)
        fh.write("Epoch\tloss\tP-T\tBA-T\tUBA-T\twF-T\tMF-T\n")
        fh.flush()

        for epoch in range(start_epoch, num_epochs):
            print("Epoch: {:d}/{:d}\n".format(epoch, num_epochs))
            self.log.write_info("Epoch: {:d}/{:d}\n".format(epoch, num_epochs))
            sys.stdout.flush()

            if model_name in ["s2p", "s2pmlp"]:
                loss_value = self.train_cnn_epoch(epoch)
            else:
                loss_value = self.train_epoch(epoch)

            if self.scheduler_name == "CyclicLR":
                self.scheduler.step()

                my_lr = self.scheduler.get_last_lr()[0]
                print("Learning rate: {:.6f}".format(my_lr))

            elif self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(
                    self.train_metrics_tracker.get_balanced_accuracy() * 100
                )

            save_model(
                self.net,
                self.params["measure"],
                self.checkpoint_dir,
                self.get_filename(model_name, ext, prefix="acc_"),
                mode="last",
                epoch=epoch,
            )

            self.log.write_info("Saved model at epoch {:d}".format(epoch))

            fh.write(
                "{:3d}\t{:9.6f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\n".format(
                    epoch,
                    loss_value,
                    self.train_metrics_tracker.get_precision_overall() * 100,
                    # self.train_metrics_tracker.get_recall_overall() * 100,
                    self.train_metrics_tracker.get_balanced_accuracy() * 100,
                    self.train_metrics_tracker.get_accuracy() * 100,
                    self.train_metrics_tracker.get_weighted_f1score() * 100,
                    self.train_metrics_tracker.get_macro_f1_score() * 100,
                    # self.train_metrics_tracker.get_beta_f1_score() * 100,
                )
            )
            fh.flush()

            if self.optimizer.param_groups[0]["lr"] < self.params["min_lr"]:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            if (self.scheduler_name == "CyclicLR") & (
                self.optimizer.param_groups[0]["lr"] > self.params["max_lr"]
            ):
                print("\n!! LR EQUAL TO MAX LR SET.")
                break

            # Stop training after params['max_time'] hours
            if time.time() - t0 > self.params["max_time"] * 3600:
                print("-" * 89)
                print(
                    "Max_time for training elapsed {:.2f} hours, so stopping".format(
                        params["max_time"]
                    )
                )
                break

        fh.close()

        outfilename = self.get_filename(
            model_name, suffix="_learning_curve", extension="_loss.png"
        )
        plot_learning_curves(
            self.checkpoint_dir, filename, outfilename, "loss", mode="final"
        )

    def load_checkpoint(self):
        print("==> Resuming from checkpoint..")

        ext = ".pth"
        fullpathname = os.path.join(
            self.checkpoint_dir,
            self.get_filename(
                self.net.get_model_name(), ext, prefix="last_acc_"
            ),
        )

        checkpoint = torch.load(fullpathname)
        self.net.load_state_dict(checkpoint["net"])
        self.val_metrics_tracker.set_measure(
            self.resume_measure, checkpoint["measure"]
        )

        bp()
        start_epoch = checkpoint["epoch"]
        # print(
        #     "Best {:s} so far: {:.4f}".format(
        #         self.resume_measure, checkpoint["measure"]
        #     )
        # )

        return start_epoch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.net.set_batch_size(batch_size)

    def run(self):
        """ """
        if (self.params["training_mode"] == "crossval") | (
            self.params["training_mode"] == "original"
        ):
            self.train_and_val()

        if self.params["training_mode"] == "final":
            self.train_final_model()

        self.log.write_ending()


#############################################################################
#


class TrainerGraphModels(TrainerBaseClass):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.load_training_data(
            config["dataset"], config["net_params"], config["b_filter_imgs"]
        )

        self.initialise_performance_trackers(
            config["dataset"], self.n_out_classes
        )

        self.net = gnn_model(config["net_name"], config)

        if config["net_name"] in [
            "GPARev",
            "GPA",
            "GIP",
            "GPARevPool",
            "G4SOP",
        ]:
            prior_graph = get_prior_graph(config, self.data_wrapper)
            self.net.initialise_prior_graph(prior_graph)

            # if config["net_name"] == "GPA":
            #     self.net.set_batch_size()

        # self.net.print_number_parameters()

        print("\n{:s} parameters: ".format(config["net_name"]))
        print_model_parameters(self.net)

        # self.learning_rate = config["params"]["init_lr"]
        self.configure_optimizer(self.net, config)

        self.initialise_checkpoint_dir(
            self.net.get_model_name(), self.n_out_classes
        )
        self.initialise_log(self.net.get_model_name(), self.n_out_classes)

    def load_training_data(
        self, dataset_name, net_params, b_filter_imgs=False
    ):
        """ """
        data_wrapper = WrapperDatasets(
            root_dir=self.root_dir,
            data_dir=self.data_dir,
            num_classes=self.n_out_classes,
            fold_id=self.params["fold_id"],
            graph_mode=net_params["graph_type"],
            n_graph_nodes=net_params["n_graph_nodes"],
            node_feat_size=net_params["node_feat_size"],
        )
        data_wrapper.load_split_set(
            dataset_name,
            self.params["training_mode"],
            "train",
            b_use_card=net_params["use_card"],
            b_use_conf=net_params["use_conf"],
            b_filter_imgs=b_filter_imgs,
        )

        set_seed(self.params["seed"])  # for replicability of the shuffling

        self.training_loader = DataLoader(
            data_wrapper.get_training_set(),
            batch_size=self.params["batch_size_train"],
            shuffle=True,
            num_workers=self.num_workers,
            # drop_last=True,
        )

        if (self.params["training_mode"] == "crossval") | (
            self.params["training_mode"] == "original"
        ):
            set_seed(self.params["seed"])  # for replicability of the shuffling

            self.validation_loader = DataLoader(
                data_wrapper.get_validation_set(),
                batch_size=self.params["batch_size_val"],
                shuffle=True,
                num_workers=self.num_workers,
                # drop_last=False,
            )

        self.cls_weights = data_wrapper.get_class_weights()
        self.data_wrapper = data_wrapper

    def train_epoch(self, epoch):
        """Train 1 epoch of the model"""
        self.set_batch_size(self.params["batch_size_train"])

        # if torch.cuda.is_available():
        #     self.net = nn.DataParallel(self.net, device_ids=[x for x in range(n_gpus)])

        self.net = self.net.to(device)
        self.net.train()

        # Initialise the instance of the average meter class to monitor the loss
        train_losses = AverageMeter()

        self.log.write_epoch_info("Training")

        # Initialise training
        start_epoch_time = time.time()
        start_batch_time = start_epoch_time

        cm_pred = []
        cm_targets = []

        # for batch_idx, (
        #     target,
        #     full_im,
        #     categories,
        #     image_name,
        #     weights,
        #     rois,
        # ) in enumerate(tqdm(self.training_loader)):
        # for batch_idx, (
        #     target,
        #     node_feats,
        #     image_name,
        #     weights,
        # ) in enumerate(tqdm(self.training_loader)):
        for batch_idx, (
            node_feats,
            target,
            weights,
            image_name,
        ) in enumerate(tqdm(self.training_loader, ascii=True)):
            #  Take GT labels
            # target = target.cuda(non_blocking=True)  # async=True)
            # target_var = Variable(target).to(device)
            target_var = Variable(target).to(device, non_blocking=True)
            targets = target_var.data.cpu().numpy()

            batch_stats = np.array(
                [
                    len(targets[targets == x]) / len(targets) * 100
                    for x in range(self.n_out_classes)
                ]
            )

            if batch_stats.any() == 0.0:
                print("One class not sampled!")

            self.optimizer.zero_grad()

            # categories_var = Variable(categories).to(device)
            # full_im_var = Variable(full_im).to(device)
            # outputs = self.net(full_im_var, categories_var)

            # node_feats_var = Variable(node_feats).to(device)
            # outputs = self.net(node_feats_var)

            outputs = self.net(node_feats.to(device))

            weights_var = Variable(weights)
            weights_var = weights_var.to(device).to(torch.float32)

            if self.b_bce:
                assert self.n_out_classes == 2

                target_var = target_var.to(torch.float32)

                assert len(outputs.shape) in [1, 2]

                if len(outputs.shape) == 1:
                    # This is for the GRM case and should go first to avoid errors in accessing the dimensionality
                    out_logits = outputs
                elif (len(outputs.shape) == 2) & (outputs.shape[1] == 1):
                    # This is for the GAT case
                    out_logits = outputs[:, 0]

                if self.b_use_weight_loss:
                    bce = torch.nn.BCEWithLogitsLoss(pos_weight=weights_var)
                else:
                    bce = torch.nn.BCEWithLogitsLoss()

                loss = bce(out_logits, target_var)

                out_probs = torch.sigmoid(out_logits)
                preds = out_probs.round().data.cpu().numpy().tolist()

            else:
                assert self.n_out_classes >= 2

                cls_weights_var = torch.from_numpy(self.cls_weights)
                cls_weights_var = cls_weights_var.to(device).to(torch.float32)

                if self.b_use_weight_loss:
                    ce_loss = torch.nn.CrossEntropyLoss(weight=cls_weights_var)
                else:
                    ce_loss = torch.nn.CrossEntropyLoss()

                loss = ce_loss(outputs, target_var)

                output_np = F.softmax(outputs, dim=1).data.cpu().numpy()

                preds = list(np.argmax(output_np, axis=1))

            loss.backward()
            self.optimizer.step()

            train_losses.update(loss.item())

            # Take predictions from Graph model
            cm_pred = np.concatenate([cm_pred, preds])
            cm_targets = np.concatenate([cm_targets, targets])

            if batch_idx % 20 == 0 and batch_idx > 1:
                self.compute_and_save_info(
                    self.train_metrics_tracker,
                    cm_pred,
                    cm_targets,
                    "batch",
                )

            start_batch_time = time.time()

        self.compute_and_save_info(
            self.train_metrics_tracker, cm_pred, cm_targets, "epoch"
        )

        print(
            "Epoch processing time: {:.4f} seconds".format(
                time.time() - start_epoch_time
            )
        )

        return train_losses.get_average()

    def val_epoch(self):
        """ """
        print("\nValidating ...")

        self.net.set_batch_size(self.params["batch_size_val"])

        # if torch.cuda.is_available():
        #     self.net = nn.DataParallel(self.net, device_ids=[x for x in range(n_gpus)])

        self.net = self.net.to(device)
        self.net.eval()

        self.log.write_epoch_info("Validating")

        # Initialise the instance of the average meter class to monitor the loss
        val_losses = AverageMeter()

        # Initialise validation variables
        cm_pred = []
        cm_targets = []

        prediction_scores = []
        target_scores = []

        img_arr = []

        # Initialise training
        start_epoch_time = time.time()
        start_batch_time = start_epoch_time

        with torch.no_grad():
            # for batch_idx, (
            #     target,
            #     full_im,
            #     categories,
            #     image_name,
            #     weight,
            #     rois,
            # ) in enumerate(tqdm(self.validation_loader, ascii=True)):
            for batch_idx, (
                node_feats,
                target,
                weights,
                image_name,
            ) in enumerate(tqdm(self.validation_loader, ascii=True)):
                # Compute and update predictions
                # full_im_var = Variable(full_im).to(device)
                # categories_var = Variable(categories).to(device)

                # outputs = self.net(full_im_var,categories_var)

                target_var = Variable(target).to(device, non_blocking=True)

                node_feats_var = Variable(node_feats).to(device)
                outputs = self.net(node_feats_var)

                if self.b_bce:
                    assert self.n_out_classes == 2

                    assert len(outputs.shape) in [1, 2]

                    if len(outputs.shape) == 1:
                        out_logits = outputs
                    elif (len(outputs.shape) == 2) & (outputs.shape[1] == 1):
                        out_logits = outputs[:, 0]

                    # convert logits into probabilities
                    out_probs = torch.sigmoid(out_logits)

                    # Round the probabilities to determine the class
                    preds = out_probs.round()

                    # Convert predictions (tensor) into a list
                    preds = preds.data.cpu().numpy().tolist()
                    prediction_scores.append(preds)

                    if self.b_use_weight_loss:
                        bce = torch.nn.BCEWithLogitsLoss(
                            pos_weight=weights_var
                        )
                    else:
                        bce = torch.nn.BCEWithLogitsLoss()

                    val_loss = bce(out_logits, target_var)

                else:
                    assert self.n_out_classes >= 2

                    cls_weights_var = torch.from_numpy(self.cls_weights)
                    cls_weights_var = cls_weights_var.to(device).to(
                        torch.float32
                    )

                    if self.b_use_weight_loss:
                        ce_loss = torch.nn.CrossEntropyLoss(
                            weight=cls_weights_var
                        )
                    else:
                        ce_loss = torch.nn.CrossEntropyLoss()

                    val_loss = ce_loss(outputs, target_var)

                    outputs_np = F.softmax(outputs, dim=1).data.cpu().numpy()
                    preds = np.argmax(outputs_np, axis=1)
                    prediction_scores.append(outputs_np[:, 0])

                val_losses.update(val_loss.item())

                # Add the predictions in current batch to all predictions
                cm_pred = np.concatenate([cm_pred, preds])

                # Update targets (labels)
                targets = target_var.data.cpu().numpy()
                cm_targets = np.concatenate([cm_targets, targets])
                target_scores.append(targets)

                img_arr.append(image_name)

                if batch_idx % 20 == 0 and batch_idx > 1:
                    self.compute_and_save_info(
                        self.val_metrics_tracker, cm_pred, cm_targets, "batch"
                    )

                # Reset the batch time
                start_batch_time = time.time()

            self.compute_and_save_info(
                self.val_metrics_tracker, cm_pred, cm_targets, "epoch"
            )

            print(
                "Epoch processing time: {:.4f} seconds".format(
                    time.time() - start_epoch_time
                )
            )

        return val_losses.get_average()


#############################################################################


class TrainerImageModels(TrainerBaseClass):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.load_training_data(
            config["dataset"], config["net_params"], config["b_filter_imgs"]
        )

        self.initialise_performance_trackers(
            config["dataset"], self.n_out_classes
        )

        self.net = gnn_model(config["net_name"], config)

        if config["net_name"] in ["GPA", "GIP"]:
            prior_graph = get_prior_graph(config, self.data_wrapper)
            self.net.initialise_prior_graph(prior_graph)

        print("\n{:s} parameters: ".format(config["net_name"]))
        print_model_parameters(self.net)

        self.configure_optimizer(self.net, config)

        self.initialise_checkpoint_dir(
            self.net.get_model_name(), self.n_out_classes
        )
        self.initialise_log(self.net.get_model_name(), self.n_out_classes)

    def load_training_data(
        self, dataset_name, net_params, b_filter_imgs=False
    ):
        """ """
        data_wrapper = WrapperImageDatasets(
            root_dir=self.root_dir,
            data_dir=self.data_dir,
            num_classes=self.n_out_classes,
            fold_id=self.params["fold_id"],
            image_size=net_params["img_size"],
        )
        data_wrapper.load_split_set(
            dataset_name,
            self.params["training_mode"],
            "train",
        )

        set_seed(self.params["seed"])  # for replicability of the shuffling

        self.training_loader = DataLoader(
            data_wrapper.get_training_set(),
            batch_size=self.params["batch_size_train"],
            shuffle=True,
            num_workers=self.num_workers,
            # drop_last=True,
        )

        if (self.params["training_mode"] == "crossval") | (
            self.params["training_mode"] == "original"
        ):
            set_seed(self.params["seed"])  # for replicability of the shuffling

            self.validation_loader = DataLoader(
                data_wrapper.get_validation_set(),
                batch_size=self.params["batch_size_val"],
                shuffle=True,
                num_workers=self.num_workers,
                # drop_last=False,
            )

        self.cls_weights = data_wrapper.get_class_weights()
        self.data_wrapper = data_wrapper

    def train_epoch(self, epoch):
        """Train 1 epoch of the model"""
        self.set_batch_size(self.params["batch_size_train"])

        model_name = self.net.get_model_name()

        # if torch.cuda.is_available():
        #     # self.net = nn.DataParallel(self.net, device_ids=[x for x in range(n_gpus)])
        #     model = nn.DataParallel(self.net, device_ids=[x for x in range(n_gpus)])
        # else:
        #     model = self.net

        self.net = self.net.to(device)
        self.net.train()

        # model = model.to(device)
        # model.train()

        # Initialise the instance of the average meter class to monitor the loss
        train_losses = AverageMeter()

        self.log.write_epoch_info("Training")

        # Initialise training
        start_epoch_time = time.time()
        start_batch_time = start_epoch_time

        cm_pred = []
        cm_targets = []

        for batch_idx, (
            img,
            target,
            weights,
            image_name,
            o_im_size,
        ) in enumerate(tqdm(self.training_loader)):
            #  Take GT labels
            target_var = Variable(target).to(device, non_blocking=True)
            targets = target_var.data.cpu().numpy()

            batch_stats = np.array(
                [
                    len(targets[targets == x]) / len(targets) * 100
                    for x in range(self.n_out_classes)
                ]
            )

            if batch_stats.any() == 0.0:
                print("One class not sampled!")

            self.optimizer.zero_grad()

            img_var = Variable(img).to(device)

            if model_name in ["s2p", "s2pmlp"]:
                outputs, _ = self.net(img_var)
                # outputs, _ = model(img_var)

            elif model_name == "rnp2ftp":
                outputs = self.net(img_var)

            elif model_name in ["gip"]:
                outputs, _ = self.net(
                    img_var, image_name, Variable(o_im_size).to(device)
                )

            else:
                outputs, _ = self.net(img_var, image_name)
                # outputs, _ = model(img_var, image_name)

            weights_var = Variable(weights)
            weights_var = weights_var.to(device).to(torch.float32)

            if self.b_bce:
                assert self.n_out_classes == 2

                target_var = target_var.to(torch.float32)

                assert len(outputs.shape) in [1, 2]

                if len(outputs.shape) == 1:
                    # This is for the GRM case and should go first to avoid errors in accessing the dimensionality
                    out_logits = outputs
                elif (len(outputs.shape) == 2) & (outputs.shape[1] == 1):
                    # This is for the GAT case
                    out_logits = outputs[:, 0]

                if self.b_use_weight_loss:
                    bce = torch.nn.BCEWithLogitsLoss(pos_weight=weights_var)
                else:
                    bce = torch.nn.BCEWithLogitsLoss()

                loss = bce(out_logits, target_var)

                out_probs = torch.sigmoid(out_logits)
                preds = out_probs.round().data.cpu().numpy().tolist()

            else:
                assert self.n_out_classes >= 2

                cls_weights_var = torch.from_numpy(self.cls_weights)
                cls_weights_var = cls_weights_var.to(device).to(torch.float32)

                if self.b_use_weight_loss:
                    ce_loss = torch.nn.CrossEntropyLoss(weight=cls_weights_var)
                else:
                    ce_loss = torch.nn.CrossEntropyLoss()

                loss = ce_loss(outputs, target_var)

                output_np = F.softmax(outputs, dim=1).data.cpu().numpy()

                preds = list(np.argmax(output_np, axis=1))

            loss.backward()
            self.optimizer.step()

            train_losses.update(loss.item())

            # Take predictions from Graph model
            cm_pred = np.concatenate([cm_pred, preds])
            cm_targets = np.concatenate([cm_targets, targets])

            if batch_idx % 20 == 0 and batch_idx > 1:
                self.compute_and_save_info(
                    self.train_metrics_tracker,
                    cm_pred,
                    cm_targets,
                    "batch",
                )

            start_batch_time = time.time()

        self.compute_and_save_info(
            self.train_metrics_tracker, cm_pred, cm_targets, "epoch"
        )

        print(
            "Epoch processing time: {:.4f} seconds".format(
                time.time() - start_epoch_time
            )
        )

        return train_losses.get_average()

    def val_epoch(self):
        """ """
        print("\nValidating ...")

        self.net.set_batch_size(self.params["batch_size_val"])

        model_name = self.net.get_model_name()

        # if torch.cuda.is_available():
        #     self.net = nn.DataParallel(self.net, device_ids=[x for x in range(n_gpus)])

        self.net = self.net.to(device)
        self.net.eval()

        self.log.write_epoch_info("Validating")

        # Initialise the instance of the average meter class to monitor the loss
        val_losses = AverageMeter()

        # Initialise validation variables
        cm_pred = []
        cm_targets = []

        prediction_scores = []
        target_scores = []

        img_arr = []

        # Initialise training
        start_epoch_time = time.time()
        start_batch_time = start_epoch_time

        with torch.no_grad():
            for batch_idx, (
                imgs,
                target,
                weights,
                image_name,
                o_im_size,
            ) in enumerate(tqdm(self.validation_loader, ascii=True)):
                # Compute and update predictions
                target_var = Variable(target).to(device, non_blocking=True)

                imgs_var = Variable(imgs).to(device)

                if model_name in ["s2p", "s2pmlp"]:
                    outputs, _ = self.net(imgs_var)
                    # outputs, _ = model(img_var)

                elif model_name == "rnp2ftp":
                    outputs = self.net(imgs_var)

                elif model_name in ["gip"]:
                    outputs, _ = self.net(
                        imgs_var, image_name, Variable(o_im_size).to(device)
                    )

                else:
                    outputs, _ = self.net(imgs_var, image_name)
                    # outputs, _ = model(img_var, image_name)

                if self.b_bce:
                    assert self.n_out_classes == 2

                    assert len(outputs.shape) in [1, 2]

                    if len(outputs.shape) == 1:
                        out_logits = outputs
                    elif (len(outputs.shape) == 2) & (outputs.shape[1] == 1):
                        out_logits = outputs[:, 0]

                    # convert logits into probabilities
                    out_probs = torch.sigmoid(out_logits)

                    # Round the probabilities to determine the class
                    preds = out_probs.round()

                    # Convert predictions (tensor) into a list
                    preds = preds.data.cpu().numpy().tolist()
                    prediction_scores.append(preds)

                    if self.b_use_weight_loss:
                        bce = torch.nn.BCEWithLogitsLoss(
                            pos_weight=weights_var
                        )
                    else:
                        bce = torch.nn.BCEWithLogitsLoss()

                    val_loss = bce(out_logits, target_var)

                else:
                    assert self.n_out_classes >= 2

                    cls_weights_var = torch.from_numpy(self.cls_weights)
                    cls_weights_var = cls_weights_var.to(device).to(
                        torch.float32
                    )

                    if self.b_use_weight_loss:
                        ce_loss = torch.nn.CrossEntropyLoss(
                            weight=cls_weights_var
                        )
                    else:
                        ce_loss = torch.nn.CrossEntropyLoss()

                    val_loss = ce_loss(outputs, target_var)

                    outputs_np = F.softmax(outputs, dim=1).data.cpu().numpy()
                    preds = np.argmax(outputs_np, axis=1)
                    prediction_scores.append(outputs_np[:, 0])

                val_losses.update(val_loss.item())

                cm_pred = np.concatenate([cm_pred, preds])

                # Update targets (labels)
                targets = target_var.data.cpu().numpy()
                cm_targets = np.concatenate([cm_targets, targets])
                target_scores.append(targets)

                img_arr.append(image_name)

                if batch_idx % 20 == 0 and batch_idx > 1:
                    self.compute_and_save_info(
                        self.val_metrics_tracker, cm_pred, cm_targets, "batch"
                    )

                # Reset the batch time
                start_batch_time = time.time()

            self.compute_and_save_info(
                self.val_metrics_tracker, cm_pred, cm_targets, "epoch"
            )

            print(
                "Epoch processing time: {:.4f} seconds".format(
                    time.time() - start_epoch_time
                )
            )

        return val_losses.get_average()


#############################################################################

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import pickle


class TrainerCNNandSVM(object):
    def __init__(self, config, args):
        # Paths
        self.root_dir = config["paths"]["root_dir"]

        self.out_dir = os.path.join(
            self.root_dir,
            "trained_models",
            # use_case_dir,
            args.dataset.lower(),
        )
        self.data_dir = os.path.join(
            config["paths"]["data_prefix"],
            config["datasets"][args.dataset]["data_dir"],
        )

        # Training parameters
        self.params = config["params"]

        self.net_params = config["net_params"]

        self.num_workers = config["num_workers"]

        # --------------------------------------------
        # Model/network parameters
        self.n_out_classes = config["net_params"]["num_out_classes"]

        self.load_training_data(
            config["dataset"], config["net_params"], config["b_filter_imgs"]
        )

        self.initialise_performance_trackers(
            config["dataset"], self.n_out_classes
        )

        self.net = gnn_model(config["net_name"], config)

        print("\n{:s} parameters: ".format(config["net_name"]))
        print_model_parameters(self.net)

        self.initialise_checkpoint_dir(
            self.net.get_model_name(), self.n_out_classes
        )
        self.initialise_log(self.net.get_model_name(), self.n_out_classes)

    def load_training_data(
        self, dataset_name, net_params, b_filter_imgs=False
    ):
        """ """
        data_wrapper = WrapperImageDatasets(
            root_dir=self.root_dir,
            data_dir=self.data_dir,
            num_classes=self.n_out_classes,
            fold_id=self.params["fold_id"],
            image_size=net_params["img_size"],
        )
        data_wrapper.load_split_set(
            dataset_name,
            self.params["training_mode"],
            "train",
        )

        set_seed(self.params["seed"])  # for replicability of the shuffling

        self.training_loader = DataLoader(
            data_wrapper.get_training_set(),
            batch_size=self.params["batch_size_train"],
            shuffle=True,
            num_workers=self.num_workers,
            # drop_last=True,
        )

        if (self.params["training_mode"] == "crossval") | (
            self.params["training_mode"] == "original"
        ):
            set_seed(self.params["seed"])  # for replicability of the shuffling

            self.validation_loader = DataLoader(
                data_wrapper.get_validation_set(),
                batch_size=self.params["batch_size_val"],
                shuffle=True,
                num_workers=self.num_workers,
                # drop_last=False,
            )

        self.cls_weights = data_wrapper.get_class_weights()
        self.data_wrapper = data_wrapper

    def initialise_checkpoint_dir(self, model_name, n_out_classes):
        """ """
        checkpoint_dir = os.path.join(
            self.out_dir,
            "{:d}-class".format(n_out_classes),
            model_name,
        )
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoint_dir = checkpoint_dir

    def initialise_performance_trackers(self, dataset, n_out_classes):
        """Monitor the performance measures (classes)."""
        self.train_metrics_tracker = PerformanceMeasuresTracker(
            dataset=dataset, n_cls=n_out_classes
        )
        self.val_metrics_tracker = PerformanceMeasuresTracker(
            dataset=dataset, n_cls=n_out_classes
        )
        self.best_metrics_tracker = PerformanceMeasuresTracker(
            dataset=dataset, n_cls=n_out_classes
        )

    def get_filename(self, model_name, extension=".csv", prefix="", suffix=""):
        """Create a filename based on the fold ID and model name.

        The function returns the filename with any additional prefix appended,
        and the extension based the argument passed (.csv as default).
        """
        if self.params["training_mode"] == "crossval":
            filename = "{:s}-{:d}".format(model_name, self.params["fold_id"])

        if self.params["training_mode"] == "final":
            filename = "{:s}-final".format(model_name)

        if self.params["training_mode"] == "original":
            filename = "{:s}-original".format(model_name)

        filename = prefix + filename + suffix + extension

        return filename

    def initialise_log(self, model_name, n_out_classes):
        """Create training log and initialise the preamble."""
        filename = self.get_filename(
            model_name, extension=".txt", suffix="_training_log"
        )

        self.log = Logging()
        self.log.initialise(os.path.join(self.checkpoint_dir, filename))
        self.log.write_preamble(model_name, n_out_classes)
        self.log.write_training_parameters(
            self.params["batch_size_train"],
            self.params["batch_size_val"],
            self.params["max_num_epochs"],
            self.params["fold_id"],
            self.params["training_mode"],
        )
        self.log.write_line()

    def compute_and_save_info(
        self, perf_tracker, cm_pred, cm_targets, mode="epoch"
    ):
        perf_tracker.compute_all_metrics(cm_targets, cm_pred)

        assert mode in ["batch", "epoch"]
        if mode == "batch":
            self.log.write_batch_info(perf_tracker)
        elif mode == "epoch":
            perf_tracker.print_metrics()
            perf_tracker.write_metrics_to_log(self.log.get_log())

    def predict_cnn_features(self, data_split):
        """ """
        print("Feature extraction with CNN: {:s} set".format(data_split))

        if data_split == "train":
            data_loader = self.training_loader
            self.net.set_batch_size(self.params["batch_size_val"])

        elif data_split == "val":
            data_loader = self.validation_loader
            self.net.set_batch_size(self.params["batch_size_train"])

        self.net = self.net.to(device)
        self.net.eval()

        # Initialise validation variables
        vision_feats_all = []
        cm_targets = []

        with torch.no_grad():
            for batch_idx, (
                imgs,
                target,
                weights,
                image_name,
                o_im_size,
            ) in enumerate(tqdm(data_loader, ascii=True)):
                vision_feats = self.net(imgs.to(device))

                vision_feats_np = vision_feats.cpu().numpy()
                if batch_idx == 0:
                    vision_feats_all = vision_feats_np
                else:
                    vision_feats_all = np.concatenate(
                        [vision_feats_all, vision_feats_np]
                    )

                # Update targets (labels)
                targets = target.cpu().numpy()
                cm_targets = np.concatenate([cm_targets, targets])

        return cm_targets, vision_feats_all

    def train_and_val(self):
        """ """
        t0 = time.time()

        model_name = self.net.get_model_name()

        # Feature extraction
        targets_train, scene_feats_train = self.predict_cnn_features("train")
        targets_val, scene_feats_val = self.predict_cnn_features("val")

        # Standardize features
        if self.net_params["b_normalise"]:
            print("Standardize features")
            scaler = StandardScaler()
            scene_feats_train = scaler.fit_transform(scene_feats_train)
            scene_feats_val = scaler.transform(scene_feats_val)
        else:
            scaler = None

        # Hyperparameter
        parameters = {
            "C": [0.1, 0.5, 1, 1.5, 2, 5, 10, 100, 1000],
            "class_weight": [None, "balanced"],
        }

        metrics_str = ["balanced_accuracy", "accuracy", "precision"]

        svm_classifier = svm.LinearSVC(
            max_iter=10000,
            loss="squared_hinge",
            penalty="l2",
        )

        clf = GridSearchCV(
            svm_classifier,
            parameters,
            scoring=metrics_str,
            n_jobs=-1,
            verbose=4,
            refit="balanced_accuracy",
        )

        ### Training
        print("Training SVM classifier with GridSearchCV")
        clf.fit(scene_feats_train, targets_train)

        best_svm_clf = clf.best_estimator_
        best_params = clf.best_params_

        best_svm_clf.fit(scene_feats_train, targets_train)

        print("Prediction on training data")
        preds_train = best_svm_clf.predict(scene_feats_train)
        self.compute_and_save_info(
            self.train_metrics_tracker, preds_train, targets_train
        )

        ### Validation
        print("Prediction on validation data")
        preds_val = best_svm_clf.predict(scene_feats_val)
        self.compute_and_save_info(
            self.val_metrics_tracker, preds_val, targets_val
        )

        ### Save SVM model
        pathfilename = os.path.join(
            self.checkpoint_dir, self.get_filename(model_name, ".pkl")
        )

        ml_classifier = {
            "estimator": best_svm_clf,
            "scaler": scaler,
            "params": best_params,
        }

        pickle.dump(ml_classifier, open(pathfilename, "wb"))
        self.log.write_info("Saved model!")

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.net.set_batch_size(batch_size)

    def run(self):
        """ """
        if (self.params["training_mode"] == "crossval") | (
            self.params["training_mode"] == "original"
        ):
            self.train_and_val()

        if self.params["training_mode"] == "final":
            raise Exception("Final training mode is not yet implemented!")

        self.log.write_ending()
