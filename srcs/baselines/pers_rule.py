#!/usr/bin/env python
#
# Rule-based method that simply discriminates the presence of the person category.
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2024/02/19
# Modified Date: 2024/02/19
#
# -----------------------------------------------------------------------------
# Libraries

# PyTorch libraries
import torch
import torch.nn as nn

from pdb import set_trace as bp


#############################################################################
class PersonRule(nn.Module):
    def __init__(
        self,
        config,
    ):
        """Constructor of the class

        Arguments:
            - n_layers: number of layers
            - n_features: number of features
            - n_features_layer: number of features per layer
            - n_out_classes: number of output classes (2 for binary classification)
            - b_batch_norm: boolean to use the batch normalization. Deafult: false
            - b_softmax: boolean to use softmax. Default: false
            - dropout_prob: probability for the droput layer. Default: 0
        """
        super(PersonRule, self).__init__()

        self.model_name = config["model_name"]

        # Hyperparameters
        net_params = config["net_params"]

        self.node_feat_size = net_params["node_feat_size"]

        self.n_classes = net_params["num_out_classes"]

        # We assume binary classification for this baseline
        try:
            if self.n_classes != 2:
                raise ValueError("The number of classes is not binary!")
        except (ValueError, IndexError):
            exit(
                "We assume binary classification for this baseline. We cannot complete the request!"
            )

        self.b_use_pers_card = net_params["use_pers_card"]
        self.person_card_th = net_params["person_card_th"]

        # We assume the person index is 2 for this baseline (HARD-CODED!!!!)
        self.person_mask = torch.zeros(
            1, net_params["n_graph_nodes"], self.node_feat_size
        )
        self.person_mask[:, 2, :] = 1.0

    def print_number_parameters(self):
        """Print classifier parameters to screen."""
        print("\n0 parameters!")

    def get_model_name(self):
        return self.model_name

    def set_batch_size(self, batch_size):
        """Dummy function based on other networks"""
        batch_sz = batch_size

    def person_presence_rule(self, x, y):
        """ """
        x_masked = x * self.person_mask

        # Here we assume that person is at index 2 (because 0 and 1 are public and private)
        z = x_masked[:, 2, :]
        z[z > 0] = 1

        zb = z[:, 0] + z[:, 1]

        # We assume the number of classes to be only 2 (binary classification)
        y[zb == 0, 0] = 100
        y[zb > 0, 1] = 100

        # This will be raised if feature normalization is enabled in the loading of the datasets
        assert y.sum([0, 1]) == 0

        return y

    def person_presence_cardinality_rule(self, x, y):
        """ """
        x_masked = x * self.person_mask

        # Here we assume that person is at index 2 (because 0 and 1 are public and private)
        z = x_masked[:, 2, :].clone()
        card = x_masked[:, 2, 0].clone()  # cardinality

        z[z > 0] = 1
        zb = z[:, 0] + z[:, 1]  # person presence

        # We assume the number of classes to be only 2 (binary classification)
        y[zb == 0, 0] = 100
        y[(zb > 0) * (card > self.person_card_th), 0] = 100
        y[(zb > 0) * (card <= self.person_card_th), 1] = 100

        # print(y.sum([0,1]))
        # This will be raised if feature normalization is enabled in the loading of the datasets
        assert y.sum([0, 1]) == 0

        return y

    def forward(self, x, adj_mat):
        """Forward pass.

        Arguments:
            - x: node features
            - adj_mat: adjacency matrix

        Return the output of the MLP before or after a sigmoid or a softmax function.
        """
        # Input data
        # if self.node_feat_size == 1:
        #     x = torch.squeeze(x, -1)
        # else:
        #     x = x.view(x.shape[0], x.shape[1] * x.shape[2])

        # Initialise output "logits" - here to simulate logits we use the numbers
        # 100 for very confident, and -100 for not confident
        #
        # PrivacyAlert (for the columns):
        #   0: public
        #   1: private
        self.y = torch.ones(x.shape[0], self.n_classes) * -100

        if self.b_use_pers_card:
            self.y = self.person_presence_cardinality_rule(x, self.y)
        else:
            self.y = self.person_presence_rule(x, self.y)

        return self.y
