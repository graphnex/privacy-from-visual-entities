#!/usr/bin/env python
#
# General utilities used in the repository
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/01/17
# Modified Date: 2024/02/19
#
# ----------------------------------------------------------------------------

import argparse
import csv
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import cv2


# ----------------------------------------------------------------------------
# CONSTANTS
#
device = "cuda" if torch.cuda.is_available() else "cpu"

n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0


# ----------------------------------------------------------------------------
def set_seed(seed_val) -> None:
    """ """
    np.random.seed(seed_val)
    random.seed(seed_val)

    cv2.setRNGSeed(seed_val)

    torch.manual_seed(seed_val)

    if device == "cuda":
        torch.cuda.manual_seed(seed_val)

        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed_val)
    print(f"Random seed set as {seed_val}")


def save_model_epoch(net, measure, epoch, pathfilename):
    state = {
        "net": net.state_dict(),
        "measure": measure,
        "epoch": epoch,
    }
    torch.save(state, pathfilename)


def save_model(net, measure, outdir, filename, mode="last", epoch=-1):
    """

    Arguments:
        - measure
        - epoch
        - outdir
        - filename
        - mode
    """
    assert mode in ["best", "last"]

    # Prepare state to save the model via PyTorch
    state = {
        "net": net.state_dict(),
        "measure": measure,
        "epoch": epoch,
        "type": mode,
    }

    pathfilename = os.path.join(outdir, mode + "_" + filename)

    # Save the mode at the given path with given filename
    torch.save(state, pathfilename)


def print_model_parameters(net):
    """Print to screen the total number of parameters of a model.

    The functions displays both the optimised parameters and the trainable
    parameters (including those that have been fixed) of a model provided as
    input.

    Arguments:
        - net: the input model.
    """
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total optimised parameters: {}".format(param_num))

    param_num = sum(p.numel() for p in net.parameters())
    print("Total trainable parameters: {}".format(param_num))


def check_if_square(A):
    m, n = A.shape
    return m == n


def check_if_symmetric(A, therr=1e-8):
    return np.all(np.abs(A - A.T) < therr)


def set_path_cwd():
    """
    Set current working directory (repo) into the path
    """
    current_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parent_dir = os.path.dirname(current_dir)
    pp_dir = os.path.dirname(parent_dir)
    # print(pp_dir)

    sys.path.insert(0, pp_dir)


def crossval_stats_summary(in_filename, out_filename, measure):
    """Output the statistics of a model trained with K-Fold Cross-Validation.

    The function reads the performance measures of a model trained on a given
    dataset using the stratified K-Fold Cross-Validation strategy and computes
    min, max, average and standard deviation for each performance measure.

    The statistics are outputted into a .csv file specific for the model and
    in the directory where the trained model is saved, e.g.,
    /trained_models/biology/brca/mlp_crossval_stats.csv.

    The performance measures are:
        * UBA-T: (unbalanced) accuracy in the training split.
        * UBA-V: (unbalanced) accuracy in the validation split.
        * BA-T: balanced accuracy in the training split.
        * BA-V: balanced accuracy in the validation split.
        * wF1-T: weighted F1-score in the training split.
        * wF1-V: weighted F1-score in the validation split.
        * mF1-T: macro F1-score in the training split.
        * mF1-V: macro F1-score in the validation split.

    The input is a .txt file with 10 columns separated by a tab. The first two
    columns are fold id and epoch (best epoch where the model was saved using
    early stopping). The last 8 columns are the performance measures defined
    above.
    """
    assert measure in [
        "precision",
        # "recall",
        "accuracy",
        "balanced_accuracy",
        "weighted_f1_score",
        "macro_f1_score",
    ]

    if measure == "precision":
        col_str = "P-V"
    # elif measure == "recall":
    #     col_str = "R-V"
    elif measure == "accuracy":
        col_str = "UBA-V"
    elif measure == "balanced_accuracy":
        col_str = "BA-V"
    elif measure == "weighted_f1_score":
        col_str = "wF1-V"
    elif measure == "macro_f1_score":
        col_str = "MF1-V"

    df = pd.read_csv(in_filename, sep="\t", index_col=False)

    idxmin = np.where(df[col_str] == df[col_str].min())[0]
    idxmax = np.where(df[col_str] == df[col_str].max())[0]

    headers = [
        "measure",
        "P-T",
        "P-V",
        # "R-T",
        # "R-V",
        "BA-T",
        "BA-V",
        "UBA-T",
        "UBA-V",
        "wF1-T",
        "wF1-V",
        "MF1-T",
        "MF1-V",
    ]

    df_out = pd.DataFrame(columns=headers)
    df_out.loc[0] = ["min"] + df.iloc[idxmin[0], 2:].values.tolist()
    df_out.loc[1] = ["max"] + df.iloc[idxmax[0], 2:].values.tolist()
    df_out.loc[2] = ["avg"] + df.iloc[:, 2:].mean().values.tolist()
    df_out.loc[3] = ["std"] + df.iloc[:, 2:].std().values.tolist()

    df_out.to_csv(out_filename, header=headers, index=False)


def plot_learning_curves(
    outdir, infilename, outfilename, measure, mode="final"
):
    """

    Arguments:
        - outdir: directory where the txt file with the learning curves was saved.
        - infilename: name of the txt file with the learning curves
        - measure: performance measure or to plot
        - mode: mode in which the model was training
    """
    assert mode in ["crossval", "final", "original"]

    assert measure in [
        "loss",
        "precision",
        # "recall",
        "accuracy",
        "balanced_accuracy",
        "weighted_f1_score",
        "macro_f1_score",
    ]

    # Read file
    df = pd.read_csv(
        "{}/{}".format(outdir, infilename), sep="\t", index_col=False
    )

    # Select the epochs to plot as x-axis
    x = df["Epoch"].values

    # Select measure
    if measure == "loss":
        col_str = "loss"
    elif measure == "precision":
        col_str = "P"
    # elif measure == "recall":
    # col_str = "R"
    elif measure == "accuracy":
        col_str = "UBA"
    elif measure == "balanced_accuracy":
        col_str = "BA"
    elif measure == "weighted_f1_score":
        col_str = "wF1"
    elif measure == "macro_f1_score":
        col_str = "MF1"

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if mode == "final":
        if measure == "loss":
            y = df[col_str].values
        else:
            y = df[col_str + "-T"].values

        # plot the data
        ax.plot(x, y, color="tab:blue")

        # set the limits
        ax.set_xlim([0, np.max(x)])

        if measure == "loss":
            ax.set_ylim([0, np.max(y) * 1.2])
        else:
            # ax.set_ylim([np.max((0, np.min(y) * 0.8)), np.max(y) * 1.2])
            ax.set_ylim([0, 100])

    else:
        # if measure == "loss":
        #     y1 = df[col_str].values
        #     y2 = df[col_str].values
        # else:
        #     y1 = df[col_str + "-T"].values
        #     y2 = df[col_str + "-V"].values

        y1 = df[col_str + "-T"].values
        y2 = df[col_str + "-V"].values

        # plot the data
        ax.plot(x, y1, color="tab:blue")
        ax.plot(x, y2, color="tab:orange")

        # set the limits
        max_y = np.max((np.max(y1), np.max(y2)))
        ax.set_xlim([0, np.max(x)])

        if measure == "loss":
            ax.set_ylim([0, max_y * 1.2])
        else:
            # ax.set_ylim([np.max((0, max_y * 0.8)), max_y * 1.2])
            ax.set_ylim([0, 100])

    ax.set_title("Learning curves")

    # plt.savefig(fig, format="png")
    plt.savefig(os.path.join(outdir, outfilename), bbox_inches="tight")


def convert_adj_to_edge_index(adjacency_matrix):
    """
    Taken from https://github.com/gordicaleksa/pytorch-GAT/blob/main/utils/utils.py

    Handles both adjacency matrices as well as connectivity masks used in softmax (check out Imp2 of the GAT model)
    Connectivity masks are equivalent to adjacency matrices they just have -inf instead of 0 and 0 instead of 1.
    I'm assuming non-weighted (binary) adjacency matrices here obviously and this code isn't meant to be as generic
    as possible but a learning resource.
    """
    assert isinstance(
        adjacency_matrix, np.ndarray
    ), f"Expected NumPy array got {type(adjacency_matrix)}."
    height, width = adjacency_matrix.shape
    assert (
        height == width
    ), f"Expected square shape got = {adjacency_matrix.shape}."

    # If there are infs that means we have a connectivity mask and 0s are where the edges in connectivity mask are,
    # otherwise we have an adjacency matrix and 1s symbolize the presence of edges.
    active_value = 0 if np.isinf(adjacency_matrix).any() else 1

    edge_index = []
    for src_node_id in range(height):
        for trg_nod_id in range(width):
            if adjacency_matrix[src_node_id, trg_nod_id] == active_value:
                edge_index.append([src_node_id, trg_nod_id])

    return np.asarray(
        edge_index
    ).transpose()  # change shape from (N,2) -> (2,N)


def convert_adj_to_edge_index_weight(adjacency_matrix, edge_th=0):
    """
    Taken from https://github.com/gordicaleksa/pytorch-GAT/blob/main/utils/utils.py

    Handles both adjacency matrices as well as connectivity masks used in softmax (check out Imp2 of the GAT model)
    Connectivity masks are equivalent to adjacency matrices they just have -inf instead of 0 and 0 instead of 1.
    I'm assuming non-weighted (binary) adjacency matrices here obviously and this code isn't meant to be as generic
    as possible but a learning resource.
    """
    assert isinstance(
        adjacency_matrix, np.ndarray
    ), f"Expected NumPy array got {type(adjacency_matrix)}."
    height, width = adjacency_matrix.shape
    assert (
        height == width
    ), f"Expected square shape got = {adjacency_matrix.shape}."

    # If there are infs that means we have a connectivity mask and 0s are where the edges in connectivity mask are,
    # otherwise we have an adjacency matrix and 1s symbolize the presence of edges.
    active_value = 0 if np.isinf(adjacency_matrix).any() else 1

    edge_index = []
    weight_list = []
    for src_node_id in range(height):
        for trg_nod_id in range(width):
            if adjacency_matrix[src_node_id, trg_nod_id] > edge_th:
                edge_index.append([src_node_id, trg_nod_id])
                weight_list.append([adjacency_matrix[src_node_id, trg_nod_id]])

    # change shape from (N,2) -> (2,N)
    return np.asarray(edge_index).transpose(), np.asarray(
        weight_list
    ).transpose().astype(np.float32)


def convert_adj_list_to_edge_index(adj_list, b_undirected=False):
    """ """
    cnt = 0

    edge_index = []
    for key, value in adj_list.items():
        if len(value) == 0:
            continue
        else:
            for v in value:
                if type(v) == list:
                    edge_index.append([int(key), v[0]])

                    if b_undirected:
                        edge_index.append([v[0], int(key)])

                else:
                    edge_index.append([int(key), v])

                    if b_undirected:
                        edge_index.append([v, int(key)])

    return np.array(edge_index).squeeze().transpose()


# ----------------------------------------------------------------------------


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def save_checkpoint(classifier, optimizer, epoch):
    checkpoint_state = {
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    checkpoint_path = (
        "./checkpoints/model.ckpt-{}_" + model_name + ".pt"
    ).format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))

    return checkpoint_path


def non_max_suppression(
    prediction, num_classes, conf_thres=0.5, nms_thres=0.4
):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Code originally from https://pjreddie.com/darknet/yolo/.
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1
        )
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(
                detections_class[:, 4], descending=True
            )
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections
                if output[image_i] is None
                else torch.cat((output[image_i], max_detections))
            )

    return output


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    Code originally from https://pjreddie.com/darknet/yolo/.
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            box1[:, 0],
            box1[:, 1],
            box1[:, 2],
            box1[:, 3],
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            box2[:, 0],
            box2[:, 1],
            box2[:, 2],
            box2[:, 3],
        )

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(
        inter_rect_x2 - inter_rect_x1 + 1, min=0
    ) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, "r")
    lines = file.read().split("\n")
    lines = [x for x in lines if x and not x.startswith("#")]
    lines = [
        x.rstrip().lstrip() for x in lines
    ]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith("["):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]["type"] = line[1:-1].rstrip()
            if module_defs[-1]["type"] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_config(path):
    """Parses the data configuration file
    Code originally from https://pjreddie.com/darknet/yolo/"""
    options = dict()
    options["gpus"] = "0,1,2,3"
    options["num_workers"] = "10"
    with open(path, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        key, value = line.split("=")
        options[key.strip()] = value.strip()
    return options


def save_predictions(img_name, preds, truths, filename):
    with open(filename, "w", newline="") as csvfile:
        # Create a csv writer object
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(["Image Name", "Predictions", "Ground-truths"])
        # Write the data rows
        for i in range(len(img_name)):
            for j in range(len(img_name[i])):
                writer.writerow([img_name[i][j], preds[i][j], truths[i][j]])
    csvfile.close()


##############################################################################
def update_config_file(config, args):
    """ """
    if args.seed is not None:
        config["params"]["seed"] = args.seed

    config["dataset"] = args.dataset

    # set_seed(config["params"]["seed"])

    # Add dataset configurations
    with open(os.path.join("configs", "datasets.json")) as f:
        data_config = json.load(f)

    config["paths"] = data_config["paths"]
    config["datasets"] = data_config["datasets"]

    # if args.root_dir is not None:
    #     config["paths"]["root_dir"] = args.root_dir

    # if args.data_dir is not None:
    #     config["paths"]["data_dir"] = args.data_dir

    if args.training_mode is not None:
        config["params"]["training_mode"] = args.training_mode

    if args.split_mode is not None:
        config["params"]["split_mode"] = args.split_mode

    if args.fold_id is not None:
        config["params"]["fold_id"] = args.fold_id

    if args.graph_mode is not None:
        config["net_params"]["graph_mode"] = args.graph_mode

    config["category_mode"] = -1  # default

    return config


def extend_parser_training(parser):
    """Extend with TRAINING parameters"""
    #
    parser.add_argument("--batch_sz_train", type=int)
    parser.add_argument("--batch_sz_val", type=int)

    parser.add_argument("--num_epochs", type=int)

    parser.add_argument(
        "--optim",
        type=str,
        help="optimizer to use (default: Adam)",
    )
    parser.add_argument("--learning_rate", type=float)

    # Resuming training parameters
    parser.add_argument(
        "--measure",
        type=str,
        choices=[
            "balanced_accuracy",
            "accuracy",
            "macro_f1_score",
            "weighted_f1_score",
        ],
        help="measure to use for the resume from checkpoint",
    )

    if sys.version_info[0] >= 3:
        if sys.version_info[1] < 9:
            parser.add_argument(
                "--weight_loss",
                action="store_true",
                help="Force to use a weighted loss to handle imbalance dataset.",
            )
            parser.add_argument(
                "--resume",
                "-r",
                action="store_true",
                help="resume from checkpoint",
            )
            parser.set_defaults(feature=False)
        else:
            parser.add_argument(
                "--weight_loss", action=argparse.BooleanOptionalAction
            )
            parser.add_argument(
                "--resume", action=argparse.BooleanOptionalAction
            )
    else:
        parser.add_argument(
            "--weight_loss",
            action="store_true",
            help="Force to use a weighted loss to handle imbalance dataset.",
        )
        parser.add_argument(
            "--no-weight_loss", dest="weight_loss", action="store_false"
        )
        parser.add_argument(
            "--resume",
            "-r",
            action="store_true",
            help="resume from checkpoint",
        )
        parser.add_argument("--no-resume", dest="resume", action="store_false")
        parser.set_defaults(feature=False)

    return parser


def extend_parser_testing(parser):
    """ """
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--res_dir", type=str)

    # parser.add_argument("--model_path", type=str)
    # parser.add_argument("--model_filename", type=str)

    parser.add_argument("--batch_size", type=int)

    return parser


def GetParser(desc=""):
    """ """

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--seed", type=int)

    # PATHS
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir", type=str)

    parser.add_argument("--bbox_dir", type=str)

    parser.add_argument("--n_workers", type=int)

    parser.add_argument(
        "--training_mode",
        type=str,
        choices=["final", "crossval", "original"],
        required=False,
        help="Choose to run K-fold cross-validation or train the final model (full training set without validation split)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["PrivacyAlert", "IPD"],
    )

    parser.add_argument(
        "--config", required=True, help="Please provide a config.json file"
    )

    parser.add_argument(
        "--mode",
        required=True,
        help="Please provide either training or testing mode.",
        choices=["training", "testing"],
    )

    parser.add_argument("--n_out_classes", type=int, choices=[2, 3])

    parser.add_argument("--fold_id", type=int)

    if sys.version_info[0] >= 3:
        if sys.version_info[1] < 9:
            parser.add_argument(
                "--use_bce",
                action="store_true",
                help="Force to use binary cross-entropy.",
            )
            # parser.add_argument(
            #     "--use_wandb",
            #     action="store_true",
            #     help="Force to use wandb",
            # )
            # parser.add_argument(
            #     "--weight_loss",
            #     action="store_true",
            #     help="Force to use a weighted loss to handle imbalance dataset.",
            # )
            parser.set_defaults(feature=False)
        else:
            parser.add_argument(
                "--use_bce", action=argparse.BooleanOptionalAction
            )
            # parser.add_argument(
            #     "--use_wandb", action=argparse.BooleanOptionalAction
            # )
            # parser.add_argument(
            #     "--weight_loss", action=argparse.BooleanOptionalAction
            # )
    else:
        parser.add_argument(
            "--use_bce",
            action="store_true",
            help="Force to use binary cross-entropy.",
        )
        parser.add_argument(
            "--no-use_bce", dest="use_bce", action="store_false"
        )
        # parser.add_argument(
        #     "--use_wandb",
        #     action="store_true",
        #     help="Force to use binary cross-entropy.",
        # )
        # parser.add_argument(
        #     "--no-use_wandb", dest="use_wandb", action="store_false"
        # )
        # parser.add_argument(
        #     "--weight_loss",
        #     action="store_true",
        #     help="Force to use binary cross-entropy.",
        # )
        # parser.add_argument(
        #     "--no-weight_loss", dest="weight_loss", action="store_false"
        # )
        parser.set_defaults(feature=False)

    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--graph_mode",
        type=str,
        choices=["bipartite", "co_occ_and_bip", "co_occ_then_bip"],
    )

    # Only for testing
    parser.add_argument("--model_mode", type=str, choices=["best", "last"])

    return parser
