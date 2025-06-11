#!/usr/bin/env python
#
# Main file to run models (S2P) on sample images
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/05/04
# Modified Date: 2025/05/05
#
# -----------------------------------------------------------------------------

import argparse
import inspect
import json
import os
import sys

import glob

from os import listdir
from os.path import isfile, join

# Package modules
current_path = os.path.abspath(inspect.getfile(inspect.currentframe()))

dirlevel1 = os.path.dirname(current_path)
dirlevel0 = os.path.dirname(dirlevel1)

print(dirlevel0)

sys.path.insert(0, dirlevel0)

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import DataLoader

from srcs.load_net import gnn_model
from srcs.datasets.imageprivacy import load_image

from srcs.logging_gnex import Logging
from srcs.utils import (
    device,
    print_model_parameters,
    update_config_file
)

#############################################################################

IMGEXT=('.jpg','.png')

class ImageData(data.Dataset):
    """Class for the image privacy datasets."""

    def __init__(
        self,
        data_dir=".",
        img_size=448,
    ):
        super(ImageData, self).__init__()

        self.data_dir = data_dir
        self.imgs = self.load_image_list(data_dir)

        self.im_size = img_size

        print(self.imgs)

    def load_image_list(self, data_dir):
        img_list = []

        for imgfn in glob.glob(os.path.join(data_dir,'*')):
            if imgfn.endswith(IMGEXT):
                img_list.append(os.path.basename(imgfn))

        return sorted(img_list)

    def __getitem__(self, index):
        """Return one item of the iterable data
        """
        full_im, w, h = load_image(
            self.imgs[index],
            self.data_dir,
            img_size=self.im_size,
        )

        print(self.imgs[index])

        return full_im, self.imgs[index], np.array([w, h])

    def __len__(self):
        return len(self.imgs)


class DemoImageModels():
    def __init__(self, 
        config, 
        use_bce=False
        ):
        # Paths
        self.root_dir = dirlevel0  # directory of the repository

        self.config = config

        self.params = config["params"]
        self.net_params = config["net_params"]

        # Boolean for using binary cross-entropy loss
        self.b_bce = use_bce

        self.log = Logging()
        self.log.initialise(os.path.join(self.root_dir, "demo", "log.txt"))
        
    def load_data(self, data_dir):
        print("Load data")

        print(data_dir)
        self.data_loader = DataLoader(
            ImageData(
                data_dir=data_dir,
                img_size=448,
            ),
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=self.config['num_workers']
        )

    def load_model(self, fullpathname, net_params):
        """ """
        self.n_out_classes = net_params["num_out_classes"]

        # self.model_mode = args.model_mode

        self.net = gnn_model(self.config["net_name"], self.config)
        self.adjacency_filename = None

        print("\n{:s} parameters: ".format(config["net_name"]))
        print_model_parameters(self.net)

        checkpoint = torch.load(fullpathname, weights_only=False)
        self.net.load_state_dict(checkpoint["net"])
        self.net = self.net.to(device)
        self.net.eval()

        self.log.write_preamble(self.net.get_model_name(), self.n_out_classes)

    def save_predictions(
        self, img_arr, prediction_scores, cm_pred
    ):
        """ """
        df_data = {
            "image": img_arr,
            "probability": prediction_scores,
            "pred_class": cm_pred,
        }
        df = pd.DataFrame(df_data)

        filename = 'demo_predictions.csv'

        df.to_csv(
            os.path.join(self.root_dir, "demo", filename),
            index=False,
        )

        print("Predictions saved in " + os.path.join(self.root_dir, "demo", filename))
    

    def predict(self, data_dir):
        print("\nPredict if images in directory are private ...")   
        
        self.net.set_batch_size(self.params["batch_size"])
        model_name = self.net.get_model_name()

        self.load_data(data_dir)

        # Initialise testing variables
        cm_pred = []
        prediction_scores = []
        sample_arr = []

        with torch.no_grad():
            for batch_idx, (
                imgs,
                image_name,
                o_im_size,
            ) in enumerate(tqdm(self.data_loader, ascii=True)):
                # Run forward of the model (return logits)

                imgs_var = imgs.to(device)

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
            # self.net.get_model_name(),
            sample_arr2,
            pred_scores_l,
            cm_pred
        )

        self.log.write_ending()


    def predict_img(self, imgfn):

        # Initialise testing variables
        cm_pred = []
        prediction_scores = []
        sample_arr = []

        dirname = os.path.dirname(imgfn)
        basename = os.path.basename(imgfn)

        full_im, w, h = load_image(
            basename,
            dirname,
            img_size=448,
        )

        if full_im.dim() == 3:
            imgs_var = torch.unsqueeze(full_im, 0).to(device)
        elif full_im.dim() == 4:
            imgs_var = full_im.to(device)

        model_name = self.net.get_model_name()

        if model_name in ["s2p", "s2pmlp"]:
            outputs, _ = self.net(imgs_var)

        elif model_name == "rnp2ftp":
            outputs = self.net(imgs_var)

        else:
            outputs, _ = self.net(imgs_var, basename)

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

        sample_arr.append(basename)

        # Prepare data for saving
        pred_scores_l = [
            num for sublist in prediction_scores for num in sublist
        ]

        self.save_predictions(
            # self.net.get_model_name(),
            sample_arr,
            pred_scores_l,
            cm_pred
        )

        self.log.write_ending()



#############################################################################
#
def GetParser(desc=""):
    """ """

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional argument 1
    parser.add_argument(
        "model", 
        type=str,
        help="Please provide path and filename of the model to use."
    )

    # Positional argument 2
    parser.add_argument(
        "config", 
        type=str,
        help="Please provide a config.json file"
    )

    # Positional argument 3
    parser.add_argument(
        "image", 
        type=str,
        help="Please provide an image file or a directory where a list of images is stored."
    )

    if sys.version_info[0] >= 3:
        if sys.version_info[1] < 9:
            parser.add_argument(
                "--use_bce",
                action="store_true",
                help="Force to use binary cross-entropy.",
            )
            parser.set_defaults(feature=False)
        else:
            parser.add_argument(
                "--use_bce", action=argparse.BooleanOptionalAction
            )
    else:
        parser.add_argument(
            "--use_bce",
            action="store_true",
            help="Force to use binary cross-entropy.",
        )
        parser.add_argument(
            "--no-use_bce", dest="use_bce", action="store_false"
        )
        parser.set_defaults(feature=False)

    return parser


#############################################################################
#
if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))
    print("Using {}".format(device))

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    # print(args.config)
    with open(args.config) as f:
        config = json.load(f)

    # Add dataset configurations
    with open(os.path.join(dirlevel0,"configs", "datasets.json")) as f:
        data_config = json.load(f)

    config["paths"] = data_config["paths"]
    # config["datasets"] = data_config["datasets"]

    processor = DemoImageModels(config, args.use_bce)

    assert(args.model.endswith('.pth')) # Validate that PyTorch file 
    processor.load_model(args.model, config['net_params'])

    if os.path.isdir(args.image):
        processor.predict(args.image)

    elif os.path.isfile(args.image):
        processor.predict_img(args.image)

    print('Finished')
