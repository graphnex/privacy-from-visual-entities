#!/usr/bin/env python
#
# Parent/base class for testing different machine learning models.
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/01/30
# Modified Date: 2024/02/19
# -----------------------------------------------------------------------------

import os
import sys

import numpy as np

np.set_printoptions(threshold=sys.maxsize, precision=2)

import pandas as pd

from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Package modules
from srcs.load_net import gnn_model
from srcs.graph.prior_graph import get_prior_graph
from srcs.datasets.wrapper import WrapperDatasets, WrapperImageDatasets

from srcs.utils import device, print_model_parameters
from srcs.logging_gnex import Logging


#
#############################################################################
# Parent class for testing a model
#
class TesterBaseClass(object):
    def __init__(self, config, args):
        # Paths
        self.root_dir = config["paths"][
            "root_dir"
        ]  # directory of the repository

        # directory of the dataset
        self.data_dir = os.path.join(
            config["paths"]["data_prefix"],
            config["datasets"][args.dataset]["data_dir"],
        )

        self.model_dir = os.path.join(
            self.root_dir,
            "trained_models",
            # use_case_dir,
            args.dataset.lower(),
        )  # directory where the model is saved

        self.res_dir = os.path.join(
            self.root_dir,
            "results",
            # use_case_dir,
            args.dataset.lower(),
        )  # directory where to save the predictions

        self.params = config["params"]
        self.num_workers = config["num_workers"]

        # Boolean for using binary cross-entropy loss
        self.b_bce = args.use_bce

        # --------------------------------------------
        # Model network
        self.n_out_classes = config["net_params"]["num_out_classes"]

        # --------------------------------------------
        self.net = None

        self.model_mode = args.model_mode

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

    def get_model(self, model_filename, model_name):
        # create .txt file to log results
        self.checkpoint_dir = os.path.join(
            self.model_dir,
            "{:d}-class".format(self.n_out_classes),
            model_name,
        )

        fullpathname = os.path.join(self.checkpoint_dir, model_filename)
        checkpoint = torch.load(fullpathname)

        # self.s2p_net.load_state_dict(checkpoint["net"])
        return checkpoint

    def load_model(self, training_mode, model_mode):
        """ """
        if training_mode == "final":
            prefix_net = "last_acc_"
        else:
            prefix_net = model_mode + "_acc_"

        checkpoint = self.get_model(
            self.get_filename(
                self.net.get_model_name(),
                ".pth",  # extension of the models
                prefix=prefix_net,
            ),
            self.net.get_model_name(),
        )

        self.net.load_state_dict(checkpoint["net"])

    def save_predictions(
        self, model_name, img_arr, prediction_scores, cm_pred, _suffix=""
    ):
        """ """
        df_data = {
            "image": img_arr,
            "probability": prediction_scores,
            "pred_class": cm_pred,
        }
        df = pd.DataFrame(df_data)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir, exist_ok=True)

        filename = self.get_filename(
            model_name, extension=".csv", suffix=_suffix
        )

        df.to_csv(
            os.path.join(self.res_dir, filename),
            index=False,
        )

        print("Predictions saved in " + os.path.join(self.res_dir, filename))

    def run(self):
        """ """
        filename = self.get_filename(
            self.net.get_model_name(), extension=".txt", suffix="_testing_log"
        )

        self.log = Logging()
        self.log.initialise(os.path.join(self.checkpoint_dir, filename))
        self.log.write_preamble(self.net.get_model_name(), self.n_out_classes)

        if self.net.get_model_name() in ["personrule"]:
            self.test_rule()
        else:
            self.test()

        self.log.write_ending()


#############################################################################
#


class TesterGraphModels(TesterBaseClass):
    def __init__(self, config, args):
        super().__init__(config, args)

        if config["net_name"] == "PersonRule":
            self.adjacency_filename = None

        self.load_testing_data(
            config["params"]["training_mode"],
            config["dataset"],
            config["net_params"],
            b_filter_imgs=config["b_filter_imgs"],
            split_mode=args.split_mode,
        )

        # Model network
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
        else:
            self.adjacency_filename = None

        if config["net_name"] in ["GIP", "GPA"]:
            self.net.set_batch_size(config["params"]["batch_size"])

        print("\n{:s} parameters: ".format(config["net_name"]))
        print_model_parameters(self.net)

        if config["net_name"] == "PersonRule":
            self.checkpoint_dir = os.path.join(
                self.model_dir,
                # "{:d}-class".format(self.n_out_classes),
                config["net_name"],
            )

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir, exist_ok=True)

        else:
            self.load_model(config["params"]["training_mode"], args.model_mode)

    def load_testing_data(
        self,
        partition,
        dataset_name,
        net_params,
        b_filter_imgs=False,
        split_mode="test",
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
            partition=partition,
            mode="train" if split_mode == "val" else split_mode,
            b_use_card=net_params["use_card"],
            b_use_conf=net_params["use_conf"],
            b_filter_imgs=b_filter_imgs,
        )

        self.testing_loader = DataLoader(
            data_wrapper.get_data_split(split_mode),
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.data_wrapper = data_wrapper

    def test_rule(self):
        """Test the model"""
        print("\nTesting ...")

        # Initialise testing variables
        cm_pred = []
        prediction_scores = []
        sample_arr = []

        with torch.no_grad():
            for batch_idx, (
                node_feats_var,
                target,
                weight,
                # adj_mat,
                sample_name,
            ) in enumerate(tqdm(self.testing_loader, ascii=True)):
                # Run forward of the model (return logits)
                outputs = self.net(node_feats_var, None)

                # Compute the predicted class for all data with softmax
                outputs_np = F.softmax(outputs, dim=1).data.cpu().numpy()
                preds = np.argmax(outputs_np, axis=1)
                prediction_scores.append(outputs_np[:, 0])

                # Add the predictions in current batch to all predictions
                cm_pred = np.concatenate([cm_pred, preds])

                # Reset the batch time
                sample_arr = np.concatenate([sample_arr, list(sample_name)])

        # Prepare data for saving
        sample_arr2 = [sample for sample in sample_arr]
        pred_scores_l = [
            num for sublist in prediction_scores for num in sublist
        ]

        self.save_predictions(
            self.net.get_model_name(),
            sample_arr2,
            pred_scores_l,
            cm_pred,
            _suffix="_" + self.model_mode,
        )

    def test(self):
        """Test the model"""
        print("\nTesting ...")

        self.net = self.net.to(device)
        self.net.eval()

        # Initialise testing variables
        cm_pred = []
        prediction_scores = []
        sample_arr = []

        with torch.no_grad():
            # for batch_idx, (
            #     target,
            #     full_im,
            #     categories,
            #     image_name,
            #     weights,
            #     rois,
            for batch_idx, (
                node_feats,
                target,
                weights,
                image_name,
            ) in enumerate(tqdm(self.testing_loader, ascii=True)):
                # Run forward of the model (return logits)
                outputs = self.net(node_feats.to(device))

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

                else:
                    assert self.n_out_classes >= 2

                    # Compute the predicted class for all data with softmax
                    outputs_np = F.softmax(outputs, dim=1).data.cpu().numpy()
                    preds = np.argmax(outputs_np, axis=1)
                    prediction_scores.append(outputs_np[:, 0])

                # Add the predictions in current batch to all predictions
                cm_pred = np.concatenate([cm_pred, preds])

                sample_arr = np.concatenate([sample_arr, list(image_name)])

        # Prepare data for saving
        sample_arr2 = [sample for sample in sample_arr]
        pred_scores_l = [
            num for sublist in prediction_scores for num in sublist
        ]

        self.save_predictions(
            self.net.get_model_name(),
            sample_arr2,
            pred_scores_l,
            cm_pred,
            _suffix="_" + self.model_mode,
        )


#############################################################################
#


class TesterImageModels(TesterBaseClass):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.load_testing_data(
            config["params"]["training_mode"],
            config["dataset"],
            config["net_params"],
            b_filter_imgs=config["b_filter_imgs"],
            split_mode=args.split_mode,
        )

        # Model network
        self.net = gnn_model(config["net_name"], config)

        if config["net_name"] in ["GPA", "GIP"]:
            prior_graph = get_prior_graph(config, self.data_wrapper)
            self.net.initialise_prior_graph(prior_graph)
        else:
            self.adjacency_filename = None

        if config["net_name"] in ["GIP", "GPA"]:
            self.net.set_batch_size(config["params"]["batch_size"])

        print("\n{:s} parameters: ".format(config["net_name"]))
        print_model_parameters(self.net)

        self.load_model(config["params"]["training_mode"], args.model_mode)

    def load_testing_data(
        self,
        partition,
        dataset_name,
        net_params,
        b_filter_imgs=False,
        split_mode="test",
    ):
        """ """
        if "img_size" not in net_params:
            net_params["img_size"] = 448

        data_wrapper = WrapperImageDatasets(
            root_dir=self.root_dir,
            data_dir=self.data_dir,
            num_classes=self.n_out_classes,
            fold_id=self.params["fold_id"],
            image_size=net_params["img_size"],
        )

        data_wrapper.load_split_set(
            dataset_name,
            partition=partition,
            mode="train" if split_mode == "val" else split_mode,
            b_filter_imgs=b_filter_imgs,
        )

        self.testing_loader = DataLoader(
            data_wrapper.get_data_split(split_mode),
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.data_wrapper = data_wrapper

    def test(self):
        """Test the model"""
        print("\nTesting ...")

        self.net.set_batch_size(self.params["batch_size"])
        model_name = self.net.get_model_name()

        self.net = self.net.to(device)
        self.net.eval()

        # Initialise testing variables
        cm_pred = []
        prediction_scores = []
        sample_arr = []

        with torch.no_grad():
            for batch_idx, (
                imgs,
                target,
                weights,
                image_name,
                o_im_size,
            ) in enumerate(tqdm(self.testing_loader, ascii=True)):
                # Run forward of the model (return logits)
                imgs_var = Variable(imgs).to(device)

                if model_name in ["s2p", "s2pmlp"]:
                    outputs, _ = self.net(imgs_var)
                    # outputs, _ = model(img_var)

                elif model_name in ["gip"]:
                    outputs, _ = self.net(
                        imgs_var, image_name, Variable(o_im_size).to(device)
                    )

                elif model_name == "rnp2ftp":
                    outputs = self.net(imgs_var)

                else:
                    outputs, _ = self.net(imgs_var, image_name)

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

                else:
                    assert self.n_out_classes >= 2

                    # Compute the predicted class for all data with softmax
                    outputs_np = F.softmax(outputs, dim=1).data.cpu().numpy()
                    preds = np.argmax(outputs_np, axis=1)
                    prediction_scores.append(np.max(outputs_np, axis=1))

                # Add the predictions in current batch to all predictions
                cm_pred = np.concatenate([cm_pred, preds])

                sample_arr = np.concatenate([sample_arr, list(image_name)])

        # Prepare data for saving
        sample_arr2 = [sample for sample in sample_arr]
        pred_scores_l = [
            num for sublist in prediction_scores for num in sublist
        ]

        self.save_predictions(
            self.net.get_model_name(),
            sample_arr2,
            pred_scores_l,
            cm_pred,
            _suffix="_" + self.model_mode,
        )


#############################################################################
import pickle


#
class TesterCNNandSVM(object):
    def __init__(self, config, args):
        # Paths
        self.root_dir = config["paths"][
            "root_dir"
        ]  # directory of the repository

        # directory of the dataset
        self.data_dir = os.path.join(
            config["paths"]["data_prefix"],
            config["datasets"][args.dataset]["data_dir"],
        )

        self.model_dir = os.path.join(
            self.root_dir,
            "trained_models",
            # use_case_dir,
            args.dataset.lower(),
        )  # directory where the model is saved

        self.res_dir = os.path.join(
            self.root_dir,
            "results",
            # use_case_dir,
            args.dataset.lower(),
        )  # directory where to save the predictions

        self.params = config["params"]
        self.net_params = config["net_params"]

        self.num_workers = config["num_workers"]

        # --------------------------------------------
        # Model network
        self.n_out_classes = config["net_params"]["num_out_classes"]

        # --------------------------------------------
        self.load_testing_data(
            config["params"]["training_mode"],
            config["dataset"],
            config["net_params"],
            b_filter_imgs=config["b_filter_imgs"],
            split_mode=args.split_mode,
        )

        self.net = gnn_model(config["net_name"], config)

        self.model_mode = args.model_mode

        self.checkpoint_dir = os.path.join(
            self.model_dir,
            "{:d}-class".format(self.n_out_classes),
            self.net.get_model_name(),
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

    def load_testing_data(
        self,
        partition,
        dataset_name,
        net_params,
        b_filter_imgs=False,
        split_mode="test",
    ):
        """ """
        if "img_size" not in net_params:
            net_params["img_size"] = 448

        data_wrapper = WrapperImageDatasets(
            root_dir=self.root_dir,
            data_dir=self.data_dir,
            num_classes=self.n_out_classes,
            fold_id=self.params["fold_id"],
            image_size=net_params["img_size"],
        )

        data_wrapper.load_split_set(
            dataset_name,
            partition=partition,
            mode="train" if split_mode == "val" else split_mode,
            b_filter_imgs=b_filter_imgs,
        )

        self.testing_loader = DataLoader(
            data_wrapper.get_data_split(split_mode),
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.data_wrapper = data_wrapper

    def predict_cnn_features(self):
        """ """
        data_loader = self.testing_loader
        self.net.set_batch_size(self.params["batch_size"])

        self.net = self.net.to(device)
        self.net.eval()

        # Initialise validation variables
        scene_feats_all = []
        sample_arr = []

        with torch.no_grad():
            for batch_idx, (
                imgs,
                target,
                weights,
                image_name,
                o_im_size,
            ) in enumerate(tqdm(data_loader, ascii=True)):
                scene_feats = self.net(imgs.to(device))

                scene_feats_np = scene_feats.cpu().numpy()

                if batch_idx == 0:
                    scene_feats_all = scene_feats_np
                else:
                    scene_feats_all = np.concatenate(
                        [scene_feats_all, scene_feats_np]
                    )

                sample_arr = np.concatenate([sample_arr, list(image_name)])

        return scene_feats_all, sample_arr

    def test(self):
        """Test the model"""
        print("\nTesting ...")

        model_name = self.net.get_model_name()

        # Feature extraction
        scene_feats, sample_arr = self.predict_cnn_features()

        # Load ML classifier (SVM)
        pathfilename = os.path.join(
            self.checkpoint_dir, self.get_filename(model_name, ".pkl")
        )

        file = open(pathfilename, "rb")
        svm_gridcv = pickle.load(file)
        file.close()

        # Standardize features
        if self.net_params["b_normalise"]:
            scene_feats = svm_gridcv["scaler"].transform(scene_feats)

        # Compute prediction with classifier
        preds = svm_gridcv["estimator"].predict(scene_feats)
        probs = svm_gridcv["estimator"].decision_function(scene_feats)
        probs[:] = 1.0  # This LinearSVC does not return probabilities

        # Prepare data for saving
        sample_arr2 = [sample for sample in sample_arr]
        # pred_scores_l = np.max(probs,axis=1).tolist()
        pred_scores_l = probs.tolist()

        self.save_predictions(
            self.net.get_model_name(),
            sample_arr2,
            pred_scores_l,
            preds,
            _suffix="_" + self.model_mode,
        )

    def save_predictions(
        self, model_name, img_arr, prediction_scores, cm_pred, _suffix=""
    ):
        """ """
        df_data = {
            "image": img_arr,
            "probability": prediction_scores,
            "pred_class": cm_pred,
        }
        df = pd.DataFrame(df_data)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir, exist_ok=True)

        filename = self.get_filename(
            model_name, extension=".csv", suffix=_suffix
        )

        df.to_csv(
            os.path.join(self.res_dir, filename),
            index=False,
        )

        print("Predictions saved in " + os.path.join(self.res_dir, filename))

    def run(self):
        """ """
        filename = self.get_filename(
            self.net.get_model_name(), extension=".txt", suffix="_testing_log"
        )

        self.log = Logging()
        self.log.initialise(os.path.join(self.checkpoint_dir, filename))
        self.log.write_preamble(self.net.get_model_name(), self.n_out_classes)

        self.test()

        self.log.write_ending()
