#!/usr/bin/env python
#
# Main file to run models (training and testing)
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/21
# Modified Date: 2025/02/05
#
# -----------------------------------------------------------------------------

import argparse
import inspect
import json
import os
import sys

# Package modules
current_path = os.path.abspath(inspect.getfile(inspect.currentframe()))

dirlevel1 = os.path.dirname(current_path)
dirlevel0 = os.path.dirname(dirlevel1)

print(dirlevel0)

sys.path.insert(0, dirlevel0)

from srcs.load_net import IMAGE_MODELS

from srcs.training_pipelines import (
    TrainerImageModels,
    TrainerGraphModels,
    TrainerCNNandSVM,
)
from srcs.tester_pipeline import (
    TesterImageModels,
    TesterGraphModels,
    TesterCNNandSVM,
)

from srcs.utils import device, GetParser, update_config_file, set_seed

#
#############################################################################


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

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_filename", type=str)

    parser.add_argument("--batch_size", type=int)

    return parser


#############################################################################
#
if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))
    # print("PyTorch {}".format(torch.__version__))
    print("Using {}".format(device))

    # Arguments
    parser = GetParser()

    if parser.parse_args().mode == "training":
        parser = extend_parser_training(parser)

    elif parser.parse_args().mode == "testing":
        parser = extend_parser_testing(parser)

    args = parser.parse_args()

    print(args.config)
    with open(args.config) as f:
        config = json.load(f)

    config = update_config_file(config, args)
    set_seed(config["params"]["seed"])

    assert args.mode in ["training", "testing"]
    if args.mode == "training":
        if config["net_name"] in IMAGE_MODELS:
            if config["net_name"] in ["RNP2SVM", "TAGSVM"]:
                processor = TrainerCNNandSVM(config, args)
            else:
                processor = TrainerImageModels(config, args)
        else:
            processor = TrainerGraphModels(config, args)

    elif args.mode == "testing":
        if config["net_name"] in ["PersonRule"]:
            processor = TesterGraphModels(config, args)

        elif config["net_name"] in IMAGE_MODELS:
            if config["net_name"] in ["RNP2SVM", "TAGSVM"]:
                processor = TesterCNNandSVM(config, args)
            else:
                processor = TesterImageModels(config, args)

        else:
            processor = TesterGraphModels(config, args)

    else:
        processor = TrainerGraphModels(config, args)

    processor.run()
