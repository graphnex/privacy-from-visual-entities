#!/usr/bin/env python
#
# Library of image privacy models to load based on the model name provided in input.
# Each model receives a set of parameters in input.
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/21
# Modified Date: 2025/02/06
#
# -----------------------------------------------------------------------------

# Package modules
from srcs.baselines.pers_rule import PersonRule as PersRuleBaseline

from srcs.nets.MLP import MLP as MLPnet
from srcs.nets.MLP import iMLP as iMLPnet

from srcs.nets.ga_mlp import GraphAgnosticMLP as GraphAgnosticMLPnet

from srcs.nets.gip_img import GraphImagePrivacy as GIPnet
from srcs.nets.gpa_img import GraphPrivacyAdvisor as GPAnet

# Scene-to-privacy classifier variants
from srcs.nets.s2p import SceneToPrivacyClassifier as S2Pnet
from srcs.nets.s2p import SceneToPrivacyMLPClassifier as S2PMLPnet

from srcs.nets.resnet_ft import ResNetPlacesFineTuningPrivacy as RNP2FTPnet
from srcs.nets.resnet_svm import ResNetPlacesAndTags as RNP2SVMnet
from srcs.nets.resnet_svm import ConvNetAndTags as TAGSVMnet


# List of image models (predefined names)
IMAGE_MODELS = [
    "S2P",
    "GIP",
    "GPA",
    "iMLP",
    "S2PMLP",
    "RNP2FTP",
    "RNP2SVM",
    "TAGSVM",
]


def GraphAgnosticMLP(net_params):
    return GraphAgnosticMLPnet(net_params)


def GPA(net_params):
    return GPAnet(net_params)


def GIP(net_params):
    return GIPnet(net_params)


def MLP(net_params):
    return MLPnet(net_params)


def iMLP(net_params):
    return iMLPnet(net_params)


def S2P(net_params):
    return S2Pnet(net_params)


def S2PMLP(net_params):
    return S2PMLPnet(net_params)


def RNP2FTP(net_params):
    return RNP2FTPnet(net_params)


def RNP2SVM(net_params):
    return RNP2SVMnet(net_params)


def TAGSVM(net_params):
    return TAGSVMnet(net_params)


def PersonRule(net_params):
    """

    @param net_params:
    @return:
    """
    return PersRuleBaseline(net_params)


def gnn_model(model_name, net_params):
    models = dict(
        GPA=GPA,
        GIP=GIP,
        GraphAgnosticMLP=GraphAgnosticMLP,
        MLP=MLP,
        iMLP=iMLP,
        S2P=S2P,
        S2PMLP=S2PMLP,
        RNP2FTP=RNP2FTP,
        RNP2SVM=RNP2SVM,
        TAGSVM=TAGSVM,
        PersonRule=PersonRule,
    )

    return models[model_name](net_params)
