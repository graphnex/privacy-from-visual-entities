#!/usr/bin/env python
#
# Module to process and convert an image into a set of concepts and features.
#
###############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/09/04
# Modified Date: 2023/09/21
# ----------------------------------------------------------------------------
#
# System libraries
import os
import sys

import argparse
import inspect

# setting path
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.insert(0, pp_dir)

# import random

import numpy as np

np.set_printoptions(threshold=sys.maxsize, precision=4)

import json
from tqdm import tqdm

# PyTorch classes, functions
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Image processing, vision libraries
from PIL import Image
import cv2  # OpenCV Library

# Package modules
from srcs.yolo.object_detection import YoloDetector, load_coco_classes

# from srcs.datasets.gip_feats.resnet_v1 import ResNet, Bottleneck
from srcs.datasets.gip_feats.gip_features import full_image_score

from srcs.datasets.gip_feats.vgg_v1 import vgg16_rois_v1

from srcs.utils import set_seed, device, print_model_parameters


#############################################################################
class ImageToGraphProcessing(nn.Module):
    """Read an image and convert it into a set of nodes and features.

    Attributes:
        repo_dir
        out_dir
        node_feature_size
        max_num_rois
        n_privacy_classes
        n_obj_cats
        n_graph_nodes
        b_normalise_feat
        img_size
    """

    def __init__(self, args, config):
        """Constructor for the class ImageToGraphProcessing"""
        super(ImageToGraphProcessing, self).__init__()

        # Path variables
        self.repo_dir = args.root_dir
        self.data_dir = config["datasets"][args.dataset]["data_dir"]

        # Dimensionality of the node features (fixed for all nodes)
        self.node_feature_size = config["graph_params"]["node_feat_size"]

        # Number of output classes (privacy levels)
        self.n_privacy_classes = config["graph_params"]["num_out_classes"]

        # Size of the image
        self.img_size = config["cnn_params"]["image_size"]

        # Maximum number of Region of Interest (ROIs) to be detected
        self.max_n_rois = config["yolo_params"]["max_num_rois"]

        # Number of object categories (for COCO=80, no background)
        self.n_obj_cats = config["yolo_params"]["num_obj_cat"]

        # Number of nodes in the image graph (fixed) given by the
        # predefined object categories and the extra public and private nodes
        self.n_graph_nodes = self.n_obj_cats + self.n_privacy_classes

        self.config = config

        # Initialise object detector
        # We pass args to initialise the two "objects" with their specific attributes
        self.initialise(args)
        # --------------------------------------------

    def initialise(self, args):
        """Initialise deep learning models and image operator.

        Initialise the class with the pre-trained object detector,
        the pretrained scene-to-rivacy classifier, and the operator
        to transform the input image.
        """
        self.set_img_transform()

        # Load object detector and scene-to-privacy classifier
        self.yolo_det = YoloDetector(args, self.config)

        # Load ResNet-101 for image feature extraction
        self.fis = full_image_score(self.n_privacy_classes, pretrained=True)
        print("\nModel parameters: ResNet-101")
        print_model_parameters(self.fis)
        self.fis.to(device).eval()

        self.full_im_net = vgg16_rois_v1(pretrained=True)
        print("\nModel parameters: VGG-16")
        print_model_parameters(self.full_im_net)
        self.full_im_net.to(device).eval()

    def set_img_transform(self):
        """ """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )  # imagenet values

        self.full_im_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )  # what about horizontal flip

    def load_image(self, image_fn):
        """Load the image from the filename into PIL format.

        The image is also resized and normalised with ImageNet weights.
        Both the PIL and transformed image are returned by the function.
        """
        img_pil = Image.open(image_fn).convert("RGB")

        full_im = self.full_im_transform(img_pil)

        return img_pil, full_im

    def get_node_features_mat(
        self, img_pil, full_im, bboxes_objects, categories
    ):
        """ """
        n_cls = self.n_privacy_classes

        node_mat = np.zeros((self.n_graph_nodes, self.node_feature_size + 3))

        node_mat[:n_cls, 0] = 1  # boolean variable for node exisiting in image
        node_mat[:n_cls, 1] = 1  # class node 1-hot encoding [1,0]
        node_mat[n_cls:, 2] = 1  # object node 1-hot encoding [0,1]

        if categories[0, 0].sum() > 0:
            rois_feature = self.full_im_net(
                full_im[None, :].to(device),
                torch.FloatTensor(bboxes_objects["bboxes"]).to(device),
                categories[None, :].to(device),
            )

            # if categories[0, 0].item() == 0:
            #     continue
            if categories[0, 0].item() < self.max_n_rois:
                cur_rois_num = categories[0, 0].item()
            else:
                # print("Number of objects")
                cur_rois_num = self.max_n_rois

            cur_rois_num = int(cur_rois_num)

            idxs = categories[1 : (cur_rois_num + 1), 0].tolist()

            for k in range(cur_rois_num):
                node_mat[int(idxs[k]) + n_cls, 0] = 1
                node_mat[int(idxs[k]) + n_cls, 3:] = (
                    rois_feature[k, :].detach().cpu().numpy()
                )

        # full image privacy scores
        # x_resnet, x_fc6, x_fc7 = self.fis(full_im[None,:].to(device))
        x_resnet, x_fc6, x_fc7 = self.fis(img_pil)

        resnet_feat_dim = x_resnet.shape[1]
        end_id = 3 + resnet_feat_dim
        class_feats = x_resnet.detach().cpu().numpy()
        for k in range(n_cls):
            node_mat[k, 3:end_id] = class_feats

        return node_mat

    def save_node_features_to_json(self, node_features, image_fn):
        """Save the extracted node features into a json file."""
        nodes = []

        # The first element of the feature vector is discarded (active or not active node)
        node_elem = {
            "node_id": 0,
            "node_name": "private",
            "node_feature": node_features[0, 1:].tolist(),
        }
        nodes.append(node_elem)

        node_elem = {
            "node_id": 1,
            "node_name": "public",
            "node_feature": node_features[1, 1:].tolist(),
        }
        nodes.append(node_elem)

        classes = load_coco_classes(self.repo_dir, self.config)

        for idx, obj_cls in enumerate(classes):
            obj_idx = idx + self.n_privacy_classes

            if node_features[obj_idx, 0] == 0:
                continue

            node_elem = {
                "node_id": idx + self.n_privacy_classes,
                "node_name": obj_cls,
                "node_feature": node_features[obj_idx, 1:].tolist(),
            }
            nodes.append(node_elem)

        path_elems = image_fn.split("/")
        if "batch" in path_elems[-2]:
            fullpath = os.path.join(
                self.data_dir,
                "graph_data",
                "gip_feats",
                "node_feats",
                path_elems[-2],
                path_elems[-1].split(".")[0] + ".json",
            )
        else:
            fullpath = os.path.join(
                self.data_dir,
                "graph_data",
                "gip_feats",
                "node_feats",
                path_elems[-1].split(".")[0] + ".json",
            )

        if not os.path.exists(os.path.dirname(fullpath)):
            os.makedirs(os.path.dirname(fullpath), exist_ok=True)

        with open(fullpath, "w") as fh:
            json.dump(nodes, fh, indent=4, sort_keys=False)
        fh.close()

    def run(self, image_fn, b_save_dets=True, b_save_imbbox=False):
        """Load and convert an image into a set of node and features

        Main function of the class to run the processing of an image, or list of images,
        and transform it into a graph representation of objects.
        """
        img_pil, full_im = self.load_image(image_fn)
        (h, w) = img_pil.size  # Image width and height

        # Run the object detector and convert the dections format
        detections = self.yolo_det.detect_image(img_pil)

        (
            bboxes_objects,
            categories,
            _,
        ) = self.yolo_det.convert_detections_format(detections, w, h)

        # Save the image with detections overlay
        if b_save_imbbox:
            self.yolo_det.save_image_bbox(detections, img_pil)

        if b_save_dets:
            self.yolo_det.save_bboxes_to_json(
                bboxes_objects, image_fn, self.data_dir
            )

        # categories_var = Variable(categories).cuda()
        # categories_var = categories_var.unsqueeze(0)

        # Prepare node features matrix
        # categories_np = categories_var.detach().cpu().numpy()

        # node_features = self.get_node_features_mat(categories_np)
        node_features = self.get_node_features_mat(
            img_pil, full_im, bboxes_objects, categories
        )
        self.save_node_features_to_json(node_features, image_fn)


#############################################################################
#
def GetParser():
    """Parse all the input arguments for execute this module.

    The parser is an object storing all the arguments that can be customised
    by the user at input, and then passed the value of these arguments to
    classes and functions called in this module.

    The parser is specific for each module.

    Best practices:
        - keep all customised variables (parameters, hyperparameters, paths,
          and flags) within this file,
        - take into consideration different compatibilities for flags (boolean),
        - group arguments by type and/or functionality (e.g. all paths together,
          all parameters for network 1, all parameters for network 2, all flags)
        - include type, deafult value, choices, and help (and other fields) as
          appropriate for help the user when running the module.

    """
    parser = argparse.ArgumentParser(
        description="Transforming an image into a graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--image_fn", type=str, default="xyz.png")

    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default=".")

    # Flags
    if sys.version_info[0] >= 3:
        if sys.version_info[1] < 9:
            parser.add_argument(
                "--b_imglist",
                action="store_true",
                help="Force to parse an image list",
            )
            parser.add_argument(
                "--norm_feat",
                action="store_true",
                help="Force to normalise cardinality feature.",
            )
            parser.add_argument(
                "--b_save_dets",
                action="store_true",
                help="Force to parse an image list",
            )
            parser.add_argument(
                "--b_save_imbbox",
                action="store_true",
                help="Force to parse an image list",
            )
            parser.set_defaults(feature=False)
        else:
            parser.add_argument(
                "--norm_feat", action=argparse.BooleanOptionalAction
            )
            parser.add_argument(
                "--b_imglist", action=argparse.BooleanOptionalAction
            )
            parser.add_argument(
                "--b_save_dets", action=argparse.BooleanOptionalAction
            )
            parser.add_argument(
                "--b_save_imbbox", action=argparse.BooleanOptionalAction
            )
    else:
        parser.add_argument(
            "--norm_feat",
            action="store_true",
            help="Force to normalise cardinality feature.",
        )
        parser.add_argument(
            "--no-norm_feat", dest="norm_feat", action="store_false"
        )
        parser.add_argument(
            "--b_imglist",
            action="store_true",
            help="Force to use cardinality information.",
        )
        parser.add_argument(
            "--no-b_imglist", dest="b_imglist", action="store_false"
        )
        parser.add_argument(
            "--b_save_dets",
            action="store_true",
            help="Force to use cardinality information.",
        )
        parser.add_argument(
            "--no-b_save_dets", dest="b_save_dets", action="store_false"
        )
        parser.add_argument(
            "--b_save_imbbox",
            action="store_true",
            help="Force to use cardinality information.",
        )
        parser.add_argument(
            "--no-b_save_imbbox", dest="b_save_imbbox", action="store_false"
        )
        parser.set_defaults(feature=False)

    parser.add_argument("--dataset", type=str, default="GIPS")

    parser.add_argument("--config", help="Please give a config.json file")

    return parser


#############################################################################
#
if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))
    print("PyTorch {}".format(torch.__version__))
    print("Using {}".format(device))
    print("OpenCV {}".format(cv2.__version__))

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    set_seed(config["params"]["seed"])

    img2graph = ImageToGraphProcessing(args, config)

    # Process a list of files or a single file
    if args.b_imglist:
        data_dir = config["datasets"][args.dataset]["data_dir"]

        imglistfn = os.path.join(data_dir, "imglist.txt")

        fh = open(imglistfn, "r")

        for idx, x in enumerate(tqdm(fh, ascii=True)):
            image_fn = x.rstrip()
            # print(image_fn)

            if args.dataset == "GIPS":
                img_fn = os.path.join(data_dir, "imgs", image_fn)
            elif args.dataset == "PrivacyAlert":
                img_fn = os.path.join(data_dir, "imgs", image_fn)
            elif args.dataset == "PicAlert":
                img_fn = os.path.join(data_dir, "imgs", image_fn)
            elif args.dataset == "VISPR":
                img_fn = os.path.join(data_dir, "imgs", image_fn)
            elif args.dataset == "IPD":
                print("Not yet implemented!")
                img_fn = os.path.join(data_dir, image_fn)

            img2graph.run(img_fn)

    else:
        # print(args.image_fn)
        img_fn = os.path.join(args.data_dir, args.image_fn)
        img2graph.run(img_fn)
