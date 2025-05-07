# Artifact Appendix

Paper title: **Learning Privacy from Visual Entities**

Artifacts HotCRP Id: **#14**

Requested Badge: **Functional and Available**


## Table of Contents
1. [Description](#description)
    1. [Security/Privacy Issues and Ethical Concerns](#security-ethical)
2. [Basic Requirements](#basic-requirements)
    1. [Hardware Requirements](#hardware-requirements)
    2. [Software Requirements](#software-requirements)
    3. [Curated image privacy data sets](#datasets)
    4. [Pre-computed visual entitities](#data-visual-entities)
    5. [Trained models](#trained-models)
    6. [Estimated Time and Storage Consumption](#time-storage)
3. [Environment](#environment)
    1. [Accessibility](#accessibility)
    2. [Set up the environment](#set-up-environment)
    3. [Testing the Environment](#testing-environment)
4. [Artifact Evaluation](#evaluation)
    1. [Main Results and Claims](#results-claims)
    2. [Experiments](#experiments)
5. [Limitations](#limitations)
6. [Notes on Reusability](#reusability)
7. [References](#references)

## Description <a name="description"></a>

This artifact contains the source code of the framework for training, testing, and evaluating the models for image privacy classification used in the article: A. Xompero and A. Cavallaro, "Learning Privacy from Visual Entities", Proceedings on Privacy Enhancing Technologies (PoPETs), Volume 2025, Issue 3, 2025.

### Security/Privacy Issues and Ethical Concerns <a name="security-ethical"></a>

The software enables the training and testing of various learning-based models for predicting an image as public or private. Some of the models learn to predict this label directly from the image as input. Other models have intermediate steps that use pre-trained model for recognising object types or scene types in the image and then these visual entities are used as input to another model trained for predicting the privacy label. The software includes the loading of exisiting public datasets for image privacy, such as PrivacyAlert and Image Privacy Dataset (IPD), that contain sensitive photos even if the original images were provided under Public Domain license by the users on Flickr. 

Potential ethical concerns of the trained models are directly related to the data used for training. Models achieve limited classification performance on the testing sets of the dataset and hence they should not be considered reliable for general use.

## Basic Requirements <a name="basic-requirements"></a>

### Hardware Requirements <a name="hardware-requirements"></a>

The software was run on a Linux-based machine using one NVIDIA GeForce GTX 1080 Ti GPU with 12 GB of RAM (CUDA 10.2). 

We expect the software to successfully run on a machine/workstation with similar or higher specifications. 

### Software Requirements <a name="software-requirements"></a>

The software was run on a Linux-based machine (CentOS Linux release 7.7.1908).

<details>
<summary> Show packages </summary><br>

* Python >3.9 
* [PyTorch](https://pytorch.org/) 1.13.1
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) (PyG)

Other libraries:
* Numpy 1.22
* scikit-learn 1.0.1
* Pandas 1.5.3
* Scipy
* Pandas
* [Networkx](https://networkx.org/) (Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks)
* json
* Matplotlib

Libraries for vision
* TorchVision 0.13.1
* OpenCV
* PIL
* [Ultralytics](https://github.com/ultralytics/ultralytics)

Additional libraries
* tqdm (for progress bar monitoring)
* [black](https://pypi.org/project/black/) (for automatic formatting according to PEP 8 Python Style Guide)
* [colorcet](https://pypi.org/project/colorcet/) (Collection of perceptually uniform colormaps)
* [six](https://pypi.org/project/six/) (a Python 2 and 3 compatibility library.)
</details>

### Curated image privacy data sets <a name="datasets"></a>

In the article, we trained and evaluated models on the Image Privacy Dataset (IPD) and the PrivacyAlert dataset. Both datasets refer to  images publicly available on Flickr. These images have a large variety of content, including _sensitive content, seminude people, vehicle plates, documents, private events_. Images were annotated with a binary label denoting if the content was deemed to be _public_ or _private_. As the images are publicly available, their label is mostly public. These datasets have therefore a high imbalance towards the public class. Note that IPD combines two other existing datasets, PicAlert and part of VISPR, to increase the number of private images already limited in PicAlert. 

List of datasets and their original source:
* [PicAlert](https://zenodo.org/record/4568971#.ZFz75tJBz0q) [Images occupy 2.4 GB]
* [VISPR](https://tribhuvanesh.github.io/vpa/) [Images occupy 49.7 GB]
* [PrivacyAlert](https://zenodo.org/record/6406870#.Y2KtsdLP3ow) [Images occupy 1 GB] 

For PicAlert and PrivacyAlert, only urls to the original locations in Flickr are available in the Zenodo record. 

**Disclaimer**: The datasets are originally provided by other sources and have been re-organised and curated for this work. Similar to the original datasets (PicAlert and PrivacyAlert), we provide the link to the images in the download scripts, however running the scripts can incur in the "429 Too Many Requests" status code. This makes the datasets hard to obtain from the original Flickr location, and thus impacting the testing of the reproducibility of the experiments. Moreover, owners of the photos on Flick could have removed the photos from the social media platform, resulting in less images than those used for training and testing the models in the article. This means that other researchers will need to privately request the current version of the datasets as used in the article to reproduce the results or make fair comparisons.

<details>
<summary> Show details and instructions </summary>

#### Data organisation <a name="data-organisation"></a>

The following folder structure using the provided folder names in a location at your own preference. 
```
.
|---image-privacy-datasets
|        |---curated_picalert
|        |        |---imgs
|        |---curated_ipd
|        |        
|        |---curated-privacy-alert-dataset
|        |        |---imgs
|        |---curated_vispr
|        |        |---imgs
```

After downloading all the images of a dataset, and independently of their initial organisation (e.g., already arranged into train, val, and test splits), all images should be arranged accordingly to a standard folder structure while making the parsing of the images easy, especially when the number is getting large.
```
.
|---image-privacy-datasets
|     |---curated_picalert
|           |-- annotations
|                 |---labels.csv
|                 |---labels_split.csv
|           |-- imgs
|                 |---batch00
|                         |--- ***.jpg
|                         |--- ***.jpg
|                         |--- ...
|                 |---batch01
|                         |--- ***.jpg
|                         |--- ***.jpg
|                         |--- ...
|                 |--- ...
|           |-- scripts
|                 |---run_imgs_to_batches.sh
|                 |--- ...
|           |-- imglist.txt
|           |-- README.md
```

#### Download <a name="data-download"></a>

The following instructions are for a Linux-based machine and use the shell terminal.

```bash
# 1. Download the zip file ``curated_imageprivacy_datasets.zip``
wget https://zenodo.org/records/15348506/files/curated_imageprivacy_datasets.zip

# 2. Extract the content of the zip file into a path of your choice. Recommendation: avoid storing the datasets under the same directory of this repository/artifact.
cd <YOUR_DATA_PATH>
unzip curated_imageprivacy_datasets.zip

# 3. Enter the dataset directory: ``cd <datapath>/curated_imageprivacy_datasets/<dataset_name>/``
# Replace <datapath> with the data path where you extract the zip file
# Replace <dataset_name> with one of curated_picalert, curated_vispr, or curated_privacyalert.
cd curated_privacyalert/

# 4. Run the script to download the images (For IPD there is no need to download the images)
source scripts/download_images.sh

# 5. Run the script to re-arrange the images in batches.
# This script assumes that the same images in the dataset are still available. 
# If there could be missing images, batches would contain different images than those organised in the curated data annotations.
source scripts/run_imgs_to_batches.sh

# 6. Go back to parent directory and repeat 3-5 steps for other datasets
cd ..

cd curated_picalert/
source scripts/download_images.sh
source scripts/run_imgs_to_batches.sh

cd ..

cd curated_vispr/
source scripts/download_images.sh
source scripts/run_imgs_to_batches.sh

cd ..
```

The script ``run_imgs_to_batches.sh`` automatically rearranges all the images in newly created subdirectories named batchX, where X is an incremental number, and each directory contains maximum 1,000 images to make the parsing easy when using a Graphical User Interface (e.g., Linux File Explorer) or the Linux shell command ``ls``, especially when the number of images is large (e.g., > 10,000 images in a single directory).

Note that the incremental number X has different number of digits depending on the number of images. For example, PrivacyAlert has only 6,800 images and therefore the number of digits for the batches is 1. For VISPR and PicAlert, the number of images is larger than 20,000 and therefore the number of digits is 2. IPD is a composition of PicAlert and VISPR, and the folder does not store any image, but we provide a way that can retrieve the image from the corresponding dataset assuming the datasets are all stored under the same parent directory.

Here is an example of the final expected organisation of the images:
```
.
|---picalert
|     |-- ...
|     |--imgs
|     |     |---batch00
|     |             |--- ***.jpg
|     |             |--- ***.jpg
|     |             |--- ...
|     |     |---batch01
|     |             |--- ***.jpg
|     |             |--- ***.jpg
|     |             |--- ...
|     |     |--- ...
|     |-- ...
```

</details>

Zenodo record storing the curated datasets (compliance with FAIR principles and Open Research): [https://doi.org/10.5281/zenodo.15348506](https://doi.org/10.5281/zenodo.15348506) 

### Pre-computed visual entitities <a name="data-visual-entities"></a>

Some of the models run their pipeline end-to-end with the images as input, whereas other models require different or additional inputs.
These inputs include the pre-computed visual entities (scene types and object types) represented in a graph format, e.g. for a Graph Neural Network.

For each image of each dataset, namely PrivacyAlert, PicAlert, and VISPR, we provide the predicted scene probabilities as a .csv file , the detected objects as a .json file in COCO data format, and the node features (visual entities already organised in graph format with their features) as a .json file. 
For consistency, all the files are already organised in batches following the structure of the images in the datasets folder.
For each dataset, we also provide the pre-computed adjacency matrix for the graph data. 

**Note**: IPD is based on PicAlert and VISPR and therefore IPD refers to the scene probabilities and object detections of the other two datasets. Both PicAlert and VISPR must be downloaded and prepared to use IPD for training and testing. 

The table below provides the link to the archive file for each dataset and each visual entity type. Links are temporarily stored at a personal public institutional web site and ZIP files will be uploaded in a Zenodo record for long-term preservation.

| Type | PicAlert | VISPR | PrivacyAlert | IPD |
|------|----------|-------|--------------|-----|
| Scenes | [link](https://zenodo.org/records/15348506/files/scenes_picalert.zip) (105 MB) | [link](https://zenodo.org/records/15348506/files/scenes_VISPR.zip) (82 MB) | [link](https://zenodo.org/records/15348506/files/scenes_privacyalert.zip) (26 MB) | N/A |
| Objects | [link](https://zenodo.org/records/15348506/files/objects_picalert.zip) (10 MB) | [link](https://zenodo.org/records/15348506/files/objects_VISPR.zip) (8 MB) | [link](https://zenodo.org/records/15348506/files/objects_privacyalert.zip) (3 MB) | N/A |
| Graph data | [link](https://zenodo.org/records/15348506/files/graphdata_picalert.zip) (9 MB) | [link](https://zenodo.org/records/15348506/files/graphdata_VISPR.zip) (7 MB) | [link](https://zenodo.org/records/15348506/files/graphdata_privacyalert.zip) (3 MB) | [link](https://zenodo.org/records/15348506/files/graphdata_IPD.zip) (22 KB) |

These files should be unzipped in the folder ``/resources/`` and follow the structure below.

Zenodo record storing the pre-computed visual entities (compliance with FAIR principles and Open Research): [https://doi.org/10.5281/zenodo.15348506](https://doi.org/10.5281/zenodo.15348506) 

<details>
<summary> Show structure </summary>

```
. # resources/
|--IPD
|     |--- graph_data
|           |---adj_mat
|                    |--- ***.csv
|
|--PicAlert
|     |---dets
|     |     |---batch00
|     |     |       |--- ***.json
|     |     |       |--- ***.json
|     |     |       |--- ...
|     |     |---batch01
|     |     |       |--- ***.json
|     |     |       |--- ***.json
|     |     |       |--- ...
|     |     |--- ...
|     |
|     |--- scenes
|     |     |---batch00
|     |     |       |--- ***.csv
|     |     |       |--- ***.csv
|     |     |       |--- ...
|     |     |---batch01
|     |     |       |--- ***.csv
|     |     |       |--- ***.csv
|     |     |       |--- ...
|     |     |--- ...
|     |
|     |--- graph_data
|           |---adj_mat
|           |        |--- ***.csv
|           |
|           |---node_feats
|                   |--- batch00
|                   |       |--- ***.json
|                   |       |--- ***.json
|                   |       |--- ...
|                   |--- batch01
|                   |       |--- ***.json
|                   |       |--- ***.json
|                   |       |--- ...
|                   |--- ...
|
|
|--PrivacyAlert
|     |---dets
|     |     |--- ...
|     |
|     |--- scenes
|     |     |--- ...
|     |
|     |--- graph_data
|           |---adj_mat
|           |        |--- ***.csv
|           |
|           |---node_feats
|                   |--- ...
|
|--VISPR
|     |---dets
|     |     |--- ...
|     |
|     |--- scenes
|     |     |--- ...
|     |
|     |--- graph_data
|           |---adj_mat
|           |        |--- ***.csv
|           |
|           |---node_feats
|                   |--- ...
|-- ...
```

</details>

Example of bash code to download the ZIP files of the visual entities and unzip them in the folder ``resources/``. 

You can run the bash script [scripts/download_precomputed_visual_entities.sh](scripts/download_precomputed_visual_entities.sh) from your terminal
```bash
source scripts/download_precomputed_visual_entities.sh
```
The script takes about 5 minutes to run a Linux-based server machine with a fast Internet connection or about 30 minutes with a laptop and home-based Internet connection via WiFi.


Alternatively, the following lines are expected to be run from the repository working directory.

```bash
# Download and extract the scene probabilities for each dataset
wget https://zenodo.org/records/15348506/files/scenes_picalert.zip
wget https://zenodo.org/records/15348506/files/scenes_VISPR.zip
wget https://zenodo.org/records/15348506/files/scenes_privacyalert.zip

unzip scenes_picalert.zip -d resources/
unzip scenes_VISPR.zip -d resources/
unzip scenes_privacyalert.zip -d resources/

# Download and extract the detected objects for each dataset
wget https://zenodo.org/records/15348506/files/objects_picalert.zip
wget https://zenodo.org/records/15348506/files/objects_VISPR.zip
wget https://zenodo.org/records/15348506/files/objects_privacyalert.zip

unzip objects_picalert.zip -d resources/
unzip objects_VISPR.zip -d resources/
unzip objects_privacyalert.zip -d resources/

# Download and extract the pre-computed graph data for each dataset
wget https://zenodo.org/records/15348506/files/graphdata_picalert.zip
wget https://zenodo.org/records/15348506/files/graphdata_VISPR.zip
wget https://zenodo.org/records/15348506/files/graphdata_privacyalert.zip
wget https://zenodo.org/records/15348506/files/graphdata_IPD.zip

unzip graphdata_picalert.zip -d resources/
unzip graphdata_VISPR.zip -d resources/
unzip graphdata_privacyalert.zip -d resources/
unzip graphdata_IPD.zip -d resources/
```

### Trained models <a name="trained-models"></a>

This section includes the link to a zip file related to each model trained in the article. 
Models can be extracted in the folder ``trained_models`` and then used within the testing pipeline to obtain the predictions on the testing sets of the datasets avoiding to re-train the model for scratch.

Zenodo record storing the trained models (compliance with FAIR principles and Open Research): [https://doi.org/10.5281/zenodo.15349470](https://doi.org/10.5281/zenodo.15349470) 

<details>
<summary> Show table of models and links for GIP and GPA analysis (Section 6.7, Tables 3 and 4) </summary>

| Model | Configuration | PrivacyAlert | IPD |
|-------|---------|--------------|-----|
| GIP   | 2.0     | [link](https://zenodo.org/records/15349470/files/gip_PrivacyAlert_v0.0.0.zip) (1.8 GB) | [link](https://zenodo.org/records/15349470/files/gip_IPD_v0.0.0.zip) (1.7 GB) |
| GIP   | 2.1     | [link](https://zenodo.org/records/15349470/files/gip_PrivacyAlert_v1.0.0.zip) (1.9 GB) | [link](https://zenodo.org/records/15349470/files/gip_IPD_v1.0.0.zip) (1.9 GB) |
| GIP   | 2.2     | [link](https://zenodo.org/records/15349470/files/gip_PrivacyAlert_v2.0.0.zip) (1.7 GB) | [link](https://zenodo.org/records/15349470/files/gip_IPD_v2.0.0.zip) (1.7 GB) |
| GIP   | 2.3     | [link](https://zenodo.org/records/15349470/files/gip_PrivacyAlert_v3.0.0.zip) (1.7 GB) | [link](https://zenodo.org/records/15349470/files/gip_IPD_v3.0.0.zip) (1.7 GB) |
| GIP   | 2.5     | [link](https://zenodo.org/records/15349470/files/gip_PrivacyAlert_v4.0.0.zip) (1.9 GB) | [link](https://zenodo.org/records/15349470/files/gip_IPD_v5.0.0.zip) (1.9 GB) |
| GPA   | 1.0     | [link](https://zenodo.org/records/15349470/files/gpa_PrivacyAlert_v1.0.0.zip) (90.5 MB) | [link](https://zenodo.org/records/15349470/files/gpa_IPD_v1.0.0.zip) (90.6 MB) |
| GPA   | 1.1     | [link](https://zenodo.org/records/15349470/files/gpa_PrivacyAlert_v1.1.0.zip) (90.5 MB) | [link](https://zenodo.org/records/15349470/files/gpa_IPD_v1.1.0.zip) (90.6 MB) |
| GPA   | 1.2     | [link](https://zenodo.org/records/15349470/files/gpa_PrivacyAlert_v1.2.0.zip) (90.5 MB) | [link](https://zenodo.org/records/15349470/files/gpa_IPD_v1.2.0.zip) (90.7 MB) |
| GPA   | 1.3     | [link](https://zenodo.org/records/15349470/files/gpa_PrivacyAlert_v1.3.0.zip) (90.5 MB) | [link](https://zenodo.org/records/15349470/files/gpa_IPD_v1.3.0.zip) (90.7 MB) |
| GPA   | 1.4     | [link](https://zenodo.org/records/15349470/files/gpa_PrivacyAlert_v1.4.0.zip) (90.5 MB) | [link](https://zenodo.org/records/15349470/files/gpa_IPD_v1.4.0.zip) (90.6 MB) |
| GPA   | 1.5     | [link](https://zenodo.org/records/15349470/files/gpa_PrivacyAlert_v1.5.0.zip) (90.5 MB) | [link](https://zenodo.org/records/15349470/files/gpa_IPD_v1.5.0.zip) (90.6 MB) |
| GPA   | 1.6     | [link](https://zenodo.org/records/15349470/files/gpa_PrivacyAlert_v1.6.0.zip) (35.4 kB) | [link](https://zenodo.org/records/15349470/files/gpa_IPD_v1.6.0.zip) (112.3 kB) |
| GPA   | 1.7     | [link](https://zenodo.org/records/15349470/files/gpa_PrivacyAlert_v1.7.0.zip) (34.2 kB) | [link](https://zenodo.org/records/15349470/files/gpa_IPD_v1.7.0.zip) (112.3 kB) |
| GPA   | 1.8     | [link](https://zenodo.org/records/15349470/files/gpa_PrivacyAlert_v1.8.0.zip) (90.5 MB) | [link](https://zenodo.org/records/15349470/files/gpa_IPD_v1.8.0.zip) (90.7 MB) |

</details>

<details>
<summary> Show table of models and links for comparative analysis (Section 6.9, Table 5) </summary>

| Model | Configuration | PrivacyAlert | IPD |
|-------|---------|--------------|-----|
| MLP-I   | 1.0     | [link](https://zenodo.org/records/15349470/files/imlp_PrivacyAlert_1.0.0.zip) (366.8 MB) | [link](https://zenodo.org/records/15349470/files/imlp_IPD_1.0.0.zip) (367.4 MB) |
| MLP   | 1.0     | [link](https://zenodo.org/records/15349470/files/mlp_PrivacyAlert_v0.0.0.zip) (167.7 kB) | [link](https://zenodo.org/records/15349470/files/mlp_IPD_v0.0.0.zip) (147.0 kB) |
| GA-MLP   | 1.4     | [link](https://zenodo.org/records/15349470/files/ga_mlp_PrivacyAlert_v1.4.0.zip) (42.7 kB) | [link](https://zenodo.org/records/15349470/files/ga_mlp_IPD_v1.4.0.zip) (158.5 kB) |
| GIP   | 2.2     | [link](https://zenodo.org/records/15349470/files/gip_PrivacyAlert_v2.0.0.zip) (1.7 GB) | [link](https://zenodo.org/records/15349470/files/gip_IPD_v2.0.0.zip) (1.7 GB) |
| GPA   | 1.4     | [link](https://zenodo.org/records/15349470/files/gpa_PrivacyAlert_v1.4.0.zip) (90.5 MB) | [link](https://zenodo.org/records/15349470/files/gpa_IPD_v1.4.0.zip) (90.6 MB) |
| S2P   | 1.0     | [link](https://zenodo.org/records/15349470/files/s2p_PrivacyAlert_v1.0.0.zip) (90.5 MB) | [link](https://zenodo.org/records/15349470/files/s2p_IPD_v1.0.0.zip) (90.6 MB) |

</details>

<details>
<summary> Show table of models and links for additional comparisons (Appendix C, Table 7) </summary>

| Model | Configuration | PrivacyAlert | IPD |
|-------|---------|--------------|-----|
| TAGSVM   | 1.0 | [link](https://zenodo.org/records/15349470/files/tagsvm_PrivacyAlert_v1.0.0.zip) (12.7 kB) | [link](https://zenodo.org/records/15349470/files/tagsvm_IPD_v1.0.0.zip) (12.9 kB) |
| RNP2SVM  | 1.0 | [link](https://zenodo.org/records/15349470/files/rnp2svm_PrivacyAlert_v1.0.0.zip) (9.0 kB) | [link](https://zenodo.org/records/15349470/files/rnp2svm_IPD_v1.0.0.zip) (5.3 kB) |
| RNP2SVM  | 1.2 | [link](https://zenodo.org/records/15349470/files/rnp2svm_PrivacyAlert_v1.2.0.zip) (5.4 kB) | [link](https://zenodo.org/records/15349470/files/rnp2svm_IPD_v1.2.0.zip) (5.3 kB) |
| RNP2FT   | 1.0 | [link](https://zenodo.org/records/15349470/files/rnp2ftp_PrivacyAlert_v1.0.0.zip) (87.6 MB) | [link](https://zenodo.org/records/15349470/files/rnp2ftp_IPD_v1.0.0.zip) (87.9 MB) |
| S2P      | 1.0 | [link](https://zenodo.org/records/15349470/files/s2p_PrivacyAlert_v1.0.0.zip) (90.5 MB) | [link](https://zenodo.org/records/15349470/files/s2p_IPD_v1.0.0.zip) (90.6 MB) |
| S2P_MLP  | 1.0 | [link](https://zenodo.org/records/15349470/files/s2p_mlp_PrivacyAlert_v1.0.0.zip) (90.7 MB) | [link](https://zenodo.org/records/15349470/files/s2p_mlp_IPD_v1.0.0.zip) (90.7 MB) |
| S2P_MLP  | 1.1 | [link](https://zenodo.org/records/15349470/files/s2p_mlp_PrivacyAlert_v1.1.0.zip) (90.8 MB) | [link](https://zenodo.org/records/15349470/files/s2p_mlp_IPD_v1.1.0.zip) (90.8 MB) |

</details>

<details>
<summary> Show table of models and links for the analysis of different design choices of MLP (Appendix D.1, Table 8) </summary>

| Model | Configuration | PrivacyAlert | IPD |
|-------|---------|--------------|-----|
| MLP   | 1.0     | [link](https://zenodo.org/records/15349470/files/mlp_PrivacyAlert_v0.0.0.zip) (167.7 kB) | [link](https://zenodo.org/records/15349470/files/mlp_IPD_v0.0.0.zip) (147.0 kB) |
| MLP   | 1.1     | [link](https://zenodo.org/records/15349470/files/mlp_PrivacyAlert_v0.1.0.zip) (176.6 kB) | [link](https://zenodo.org/records/15349470/files/mlp_IPD_v0.1.0.zip) (163.7 kB) |
| MLP   | 1.2     | [link](https://zenodo.org/records/15349470/files/mlp_PrivacyAlert_v0.2.0.zip) (171.1 kB) | [link](https://zenodo.org/records/15349470/files/mlp_IPD_v0.2.0.zip) (148.0 kB) |
| MLP   | 1.3     | [link](https://zenodo.org/records/15349470/files/mlp_PrivacyAlert_v0.3.0.zip) (179.6 kB) | [link](https://zenodo.org/records/15349470/files/mlp_IPD_v0.3.0.zip) (164.4 kB) |
| MLP   | 1.4     | [link](https://zenodo.org/records/15349470/files/mlp_PrivacyAlert_v0.4.0.zip) (167.5 kB) | [link](https://zenodo.org/records/15349470/files/mlp_IPD_v0.4.0.zip) (143.0 kB) |
| MLP   | 1.5     | [link](https://zenodo.org/records/15349470/files/mlp_PrivacyAlert_v0.5.0.zip) (177.0 kB) | [link](https://zenodo.org/records/15349470/files/mlp_IPD_v0.5.0.zip) (161.2 kB) |

Note that we provide only the trained models with feature normalisation based on the results obtained in Table 8.

</details>

<details>
<summary> Show table of models and links for hyper-parameter analysis (Appendix D.2, Figure 9) </summary>

| Model | PrivacyAlert | IPD |
|-------|--------------|-----|
| MLP   | [link](https://zenodo.org/records/15349470/files/mlp_PrivacyAlert_hyperparameter_analysis.zip) (15.9 MB) | [link](https://zenodo.org/records/15349470/files/mlp_IPD_hyperparameter_analysis.zip) (79.1 MB) |

Each archive has 37 .zip files, one for each configuration of the hyper-parameter analysis. 
These archives are meant only for reproducibility checks and are not expected to be used for future comparisons. 

</details>


### Estimated Time and Storage Consumption <a name="time-storage"></a>

Storing the datasets, the repository, the precomputed visual entities and graph data, and the trained models can approximately occupy 120 GB. 

As obtaining and reproducing the datasets from public urls is not trivial (see also [Limitations](#limitations)), providing an estimated time is also not straightforward.

Excluding the preparation of the datasets and the GIP model (largest deep learning model, whose archive file occupies approximately 3.8 GB) that runs with a batch size of maximum 2 in our machine (slow inference time), 
inference of other models on the testing sets of IPD and PrivacyAlert to reproduce the results of the article is expected to take approximately 5-6 hours.

## Environment <a name="environment"></a>
In the following, describe how to access our artifact and all related and necessary data and software components.
Afterward, describe how to set up everything and how to verify that everything is set up correctly.

### Accessibility <a name="accessibility"></a>

The repository is hosted at https://github.com/graphnex/privacy-from-visual-entities (use the most recent commit in the branch main).

### Set up the environment <a name="set-up-environment"></a>

The following commands allow to recreate the conda environment used to train and test the models for image privacy classification.

Before running any of the following commands, check your cuda version and install the compatible packages versions (e.g., PyTorch).

```bash
git clone https://github.com/graphnex/privacy-from-visual-entities

conda create -n image-privacy
conda activate image-privacy

conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 pyg -c pytorch -c pyg
conda install pandas tqdm scikit-learn scipy -c anaconda -c conda-forge

python -m pip install -U pip
python -m pip install -U matplotlib networkx black python-magic torch-summary

# Vision libraries
python -m pip install -U opencv-python
python -m pip install ultralytics
```

### Testing the Environment <a name="testing-environment"></a>

No specific tests currently available. We provide a demo to test one of the trained models with some images sampled from PrivacyAlert. We also provide bash scripts in the folder ``scripts/`` that other researchers can use to run the training and testing pipelines for each model, and evaluate the results. 

#### Demo with S2P and sample images from PrivacyAlert

The demo is located in the directory ``demo`` and containes a bash script [demo/run_demo.sh](demo/run_demo.sh) to run the demo provided in the Python file [demo/demo_s2p.py](demo/demo_s2p.py) with the S2P model, trained on PrivacyAlert, on some images of PrivacyAlert. 

The bash script of the demo can be run from a terminal and the repository directory as follows:
```bash
cd demo/

source run_demo.sh
```

<details>
<summary> Show preliminary steps (download of images and model)</summary>

The demo automatically downloads a list of images from the source location on Flickr in the demo directory. These images correspond to some of the images shown in the [paper](https://doi.org/10.48550/arXiv.2503.12464) (Sec.6.10, Fig.7, page 13).
```bash
# STEP 1: DOWNLOAD IMAGES FOR DEMO
#
# Download selected images from PrivacyAlert (as shown in the paper, Sec.6.10) in the current directory
#
# When PrivacyAlert was collected, these images were licensed under Public Domain. 
# Please verify if the user changed the licence when downloading these images.

wget https://live.staticflickr.com/65535/50039888078_88fb5d4722_c.jpg -O  50039888078.jpg
wget https://live.staticflickr.com/65535/50252178257_4e69f47eac_c.jpg -O  50252178257.jpg
wget https://live.staticflickr.com/65535/50789435118_4c012e37c1_c.jpg -O  50789435118.jpg
wget https://live.staticflickr.com/65535/50583669162_77829f45af_c.jpg -O  50583669162.jpg
wget https://live.staticflickr.com/65535/49196229738_3aa9ea88f8_c.jpg -O  49196229738.jpg
wget https://live.staticflickr.com/65535/47705667841_00724f3a60_c.jpg -O  47705667841.jpg
wget https://live.staticflickr.com/1887/43830339314_f317a8e1c9_c.jpg  -O  43830339314.jpg
wget https://live.staticflickr.com/4305/35886895852_c4f26ed6f9_c.jpg  -O  35886895852.jpg
wget https://live.staticflickr.com/65535/49834471323_2a4fd949b1_c.jpg -O  49834471323.jpg
```

The ZIP archive with the weights of the model S2P trained on PrivacyAlert is downloaded and extracted in the current directory along with the configuration file. 
```bash
# STEP 2: DOWNLOAD AND EXTRACT MODEL IN CURRENT DEMO FOLDER
#
wget https://zenodo.org/records/15349470/files/s2p_PrivacyAlert_v1.0.0.zip

# extract the trained model
unzip -j s2p_PrivacyAlert_v1.0.0.zip trained_models/privacyalert/2-class/s2p/best_acc_s2p-original.pth -d .
mv best_acc_s2p-original.pth s2p.pth # rename model for simplicity

# extract also the configuration file
unzip -j s2p_PrivacyAlert_v1.0.0.zip configs/s2p_v1.0.json -d .
```

</details>

<details>
<summary> Show details of the demo with multiple images in a custom folder</summary>

The script then executes the inference of the model on the downloaded images. 

```bash
# STEP 3: PREDICT IMAGES AS PRIVATE OR PUBLIC WITH MODEL
#
# Activate the conda environment. The command conda is the default one.
# Some OS might not work with the command conda, and the command source can be used as an alternative
conda activate image-privacy

# Set the GPU number to use in a multi-GPU machine. The variable 
# is not always needed, and depends on your machine/server
CUDA_DEVICE=0

CONFIG_FILE=s2p_v1.0.json

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python demo_s2p.py \
    s2p.pth \
    $CONFIG_FILE \
    "."

conda deactivate
```

The Python script receives three positional arguments as input strings:
1. the trained model (filename with extension .pth)
2. the configuration file
3. the directory where the list of images are stored (*or* the image filename).

The script currently supports the loading of .jpg and .png images, and automatically distinguishes between an input image and a directory.

Note that [ResNet-50 pre-trained on Places365](resnet50_places365.pth.tar) is automatically downloaded from [http://places2.csail.mit.edu/models_places365/](http://places2.csail.mit.edu/models_places365/) and placed into the directory ``resources/``. 

A custom data loader reads the images in the folder, and then the model predicts their privacy label (1: private, 0 public). 

Predictions are saved into a .csv file ``demo_predictions.csv``, whose columns are:
* image: filename of the image [str]
* probability: probability of the predicted class based on the sigmoid function [float]
* pred_class: predicted class (1: private, 0 public)  [float]

</details>

<details>
<summary> Show details of the demo with a single image</summary>

The script executes the inference of the model on an image provided as input. The following lines can be used directly on the command line of the terminal instead of running the bash script.  

```bash
# Activate the conda environment. The command conda is the default one.
# Some OS might not work with the command conda, and the command source can be used as an alternative
conda activate image-privacy

# Set the GPU number to use in a multi-GPU machine. The variable 
# is not always needed, and depends on your machine/server
CUDA_DEVICE=0

CONFIG_FILE=s2p_v1.0.json

IMAGENAME=50039888078.jpg

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python demo_s2p.py s2p.pth $CONFIG_FILE $IMAGENAME

conda deactivate
```

The Python script receives three positional arguments as input strings:
1. the trained model (filename with extension .pth)
2. the configuration file
3. the image filename (*or* the directory where the list of images are stored)

The script currently supports the loading of .jpg and .png images, and automatically distinguishes between an input image and a directory.

Note that [ResNet-50 pre-trained on Places365](resnet50_places365.pth.tar) is automatically downloaded from [http://places2.csail.mit.edu/models_places365/](http://places2.csail.mit.edu/models_places365/) and placed into the directory ``resources/``. 

A custom data loader reads the images in the folder, and then the model predicts their privacy label (1: private, 0 public). 

Predictions are saved into a .csv file ``demo_predictions.csv``, whose columns are:
* image: filename of the image [str]
* probability: probability of the predicted class based on the sigmoid function [float]
* pred_class: predicted class (1: private, 0 public)  [float]

</details>

**Reproducibility check**

We provide the predictions of the model on the sampled images in the table below. 

| image | probability | pred_class | 
|-------|-------------|------------|
| 35886895852.jpg | 0.5692839 | 1.0 |
| 43830339314.jpg | 0.5559994 | 1.0 |
| 47705667841.jpg | 0.7970571 | 1.0 |
| 49196229738.jpg | 0.746452 | 1.0 |
| 49834471323.jpg | 0.82996505 | 1.0 |
| 50039888078.jpg | 0.98218954 | 0.0 |
| 50252178257.jpg | 0.740997 | 0.0 |
| 50583669162.jpg | 0.982347 | 0.0 |
| 50789435118.jpg | 0.9788375 | 0.0 |

#### Scripts to run training and testing pipelines 

<details>
<summary> Show instructions to run script for S2P as example </summary>

Follow these instructions to run the script. 

1. Modify the bash script [scripts/run_s2p.sh](scripts/run_s2p.sh) with your own variables and settings:
    1. Set ``IMAGEPRIVACY_DIR`` with your directory to the image privacy datasets
    2. Choose the dataset for the variable ``DATASET`` (either PrivacyAlert or IPD)
    3. Set the variable ``TRAINING_MODE`` based on the chosen dataset (original for PrivacyAlert, crossval for IPD)
    4. Change the name of conda environment based on the one created. Check if the command conda or source needs to be used within your machine
    5. Comment/uncomment the pipelines (python scripts) to execute: training, testing of the best selected model, testing of the model saved at the last epoch, evaluation of the best model, evaluation of the model saved at the last epoch
2. Open the terminal in the parent working directory and run the bash script, for example: ``source scripts/run_s2p.sh``

The script will automatically execute the pipelines and save a backup ZIP archive in the directory ``backups/``. The ZIP file archives all the files related to the trained model (e.g., .pth file with the weights, logs of the training loss, plots of the training curves, etc.), the config file with the settings of the model and training, the prediction results, and the computed performance scores. 

Segment for the training pipeline:
```bash
# Make sure to run the pipeline within the created conda environment
conda activate graphnex-gnn

# Run the training pipeline
python srcs/main.py                       \  #
   --root_dir              $ROOT_DIR      \  # The parent folder of this repository
   --dataset               $DATASET       \  # Name of the dataset (e.g., IPD, PrivacyAlert)
   --config                $CONFIG_FILE   \  # Path to the config file and filename (contains the model name)
   --training_mode         $TRAINING_MODE \  # The selected training mode based on the dataset
   --mode                  "training"       # For training the model this is fixed
   # --weight_loss  # To enable the use of the weighted cross-entropy loss (Default: disabled)
   # --use_bce      # To enable the use of the binary cross-entropy loss (Default:disabled)
   # --resume       # To enable the resuming of the model training from last saved epoch (Default: disabled)

conda deactivate
```

Instead of using the bash script, the training pipeline can be run with the following command lines from the terminal (arguments are used as an example):
```bash
conda activate graphnex-gnn

python srcs/main.py --root_dir "." --dataset "PrivacyAlert" --config "configs/s2p_v1.0.0.json" --mode "training" --training_mode "original"

conda deactivate
```


Segment for the testing pipeline:
```bash
# Make sure to run the pipeline within the created conda environment
conda activate graphnex-gnn

# Run the training pipeline
python srcs/main.py                       \  #
   --root_dir              $ROOT_DIR      \  # The parent folder of this repository
   --dataset               $DATASET       \  # Name of the dataset (e.g., IPD, PrivacyAlert)
   --config                $CONFIG_FILE   \  # Path to the config file and filename (contains the model name)
   --training_mode         $TRAINING_MODE \  # The selected training mode based on the dataset
   --mode                  "testing"      \  # For testing the model this is fixed
   --split_mode            "test"         \  # What data split to test (e.g., train, val, test)
   --model_mode            "best"            # What mode to use for testing: best model based on early stop and validation split (best), or the model saved at the last epoch (last)


conda deactivate
```

Instead of using the bash script, the testing pipeline can be run with the following command lines from the terminal (arguments are used as an example):
```bash
conda activate graphnex-gnn

python srcs/main.py --root_dir "." --dataset "PrivacyAlert" --config "configs/s2p_v1.0.0.json" --mode "testing" --training_mode "original" --split_mode "test" --model_mode "best"

conda deactivate
```

</details>

## Artifact Evaluation <a name="evaluation"></a> 

### Main Results and Claims <a name="results-claims"></a> 

References to sections, tables, and figures correspond to those in the revised and accepted manuscript (uploaded for review as Revision_v1.0). 
These references will be updated to the corresponding ones in the final version after the camera-ready is submitted and published.

#### Main Result 1: Using transfer learning from a pre-trained scene classifier is sufficient for achieving best classification performance in image privacy

We perform an in-depth comprative analysis of the classification performance across various models, including GIP, GPA, MLP-I, MLP, GA-MLP, and S2P, and other baselines, on both the IPD and PrivacyAlert datasets. For both datasets, our re-trained GPA and S2P achieve the highest classification results in terms of overall accuracy (>83% on IPD, and >80% in PrivacyAlert) and balanced accuracy (>80% on IPD and >75% in PrivacyAlert). The similar performance between GPA and S2P indicates that using transfer learning from the pre-trained scene classifier is sufficient for achieving such a performance and the impact of graph processing on the results is minimal. 

The claim is mentioned across the paper, including abstract, introduction, Sec.6.11 (Discussion), and Sec.7 (Conclusion). These results are discussed in Sec.6.9, Table 5, Figure 5, anf Figure 6. 

#### Main Result 2: Relative impacts of individual model components on image privacy

We analysed the classification performance of both GIP and GPA models when re-trained with different training strategies and different design choices to better understand the relative contribution of their individual components. 
The analysis shows that the GIP is highly affected by the presence of the deep features in the privacy nodes and only using the deep features from the objects might be sufficient, and using the node type in the feature vectors is relevant for the model.
GPA degenerates to predict only the public class when pre-training and fixing the parameters of the scene-to-privacy layer, and when correcting the implementation of the adjacency matrix wrongly initialised in the original implementation of GPA. Best performance are achieved by the GPA variant with the bipartite prior graph, pre-trained re-shape layer, and use of the reshape layer. 

These results are discussed in Sec.6.7 (Relative impacts on image privacy) and presented in Table 3 and Table 4 of the article.

#### Main Result 3: Using SVM as a classifier is still a good alternative in these small datasets

We compare the classification results of S2P with those of four related works using transfer learning, CNNs, and other classifiers (e.g., Support Vector Machine or SVM): Image Tags + SVM, Top-10 Scene Tags + SVM, a scene classifier coupled with SVM (ResNet50 + SVM), and a finetuning of the scene classifier (ResNet50-FT).
ResNet50 + SVM achieves the best recall on the private class (81.42\% on IPD and 79.78\% on PrivacyAlert) and the best balanced accuracy (83.08\% on IPD and 78.48\% on PrivacyAlert). Given the small size of the datasets and the features extracted by a pre-trained CNN, the good performance of SVM are expected. However, scaling to larger dataset size is a known drawback of this model to consider.

These results are discussed in Appendix C and presented in Table 7.

#### Main Result 4: Using a 1-layer MLP is better than using a single fully connected layer as privacy classifier coupled with a pre-trained scene classifier

We compared S2P with 2 variants that replace the fully connected layer of S2P with a 1-layer MLP and 2-layer MLP, respectively, where 1-layer means the number of hidden layers.
The increasing number of parameters optimised for privacy by the variants of S2P allows the model to improve the classification performance compared to S2P, especially in terms of recall of the private class and balanced accuracy on both datasets, and achieve performance more comparable to ResNet-50 with SVM. However, increasing from 1 to 2 hidden layers provides a minimal overall improvement and an increase in false positives towards predicting images as private.

These results are discussed in Appendix C and presented in Table 7.

### Experiments <a name="experiments"></a>

These experiments simply reproduce the results reported in the article using the already trained models. For each experiment, we provide one bash script that can be run to reproduce the results. We do not include details on how to reproduce the training of the models and reviewers are not asked to run the training pipeline of the models.

The following experiments do not reproduce the analysis of the MLP variants, the anaylsis of the MLP hyperparameters, and the analysis of GA-MLP variants, presented in the Appendix of the article. Users can refer to the scripts for each model in the folder ``scripts``, the instructions provided in [Testing the Environment](#testing-environment), and the corresponding [Trained Models](#trained-models) to reproduce the results.

**Reminder**: these experiments are currently not reproducible due to the limitations in obtaining the datasets and the not-yet-available links to the trained models. Because of this, the esimated time to run each script might not be provided.

#### Experiment 1: Evaluation of various design choices for the GIP model

This experiment reproduces the results presented in Table 3 of the article, Sec.6.7 (Relative impacts on image privacy), using the various GIP models (see corresponding table in [Trained Models](#trained-models) for downloading each model). We provide a bash script that unzips the archive of each GIP model and runs each model on the testing sets of both PrivacyAlert and IPD datasets. The predictions of each model are saved into ``results/<dataset>/<model-name>.csv``. Classification performance are also computed and saved into ``results/<dataset>/res_experiment1.csv``. The latter file allows to verify the results of the experiment as reported in Table 3, except for the rows of results taken from the GIP paper.

Running this script takes approximately 15 minutes. The predictions and classification performance .csv files occupy less than 1 MB. The largest model, stored in the ``/trained_models/`` folder after unzipping, occupies 200 MB. Each model is unzipped from its corresponding archive into the directory ``trained_models/<dataset_name>/2-class/<model_name>``. This directory is removed after running each model. 

Running instructions:
1. Modify the variable ``IMAGEPRIVACY_DIR`` in the file [scripts/run_experiment1.sh](scripts/run_experiment1.sh) by placing the path to the folder where you downloaded the datasets.  
2. Place the path to the folder where you downloaded the datasets in the field ``data_dir`` of the file [configs/datasets.json](configs/datasets.json).
3. Open the terminal in the working directory of the repository.
4. Run: ``source scripts/run_experiment1.sh``

Note that the prediction file is overwritten after each model as the model name is the same.

This experiment supports the claims in the Main Results 2.

#### Experiment 2: Evaluation of various design choices for the GPA model

This experiment reproduces the results presented in Table 4 of the article, Sec.6.7 (Relative impacts on image privacy), using the various GIP models (see corresponding table in [Trained Models](#trained-models) for downloading each model). We provide a bash script that unzips the archive of each GIP model and runs each model on the testing sets of both PrivacyAlert and IPD datasets. The predictions of each model are saved into ``results/<dataset>/<model-name>.csv``. Classification performance are also computed and saved into ``results/<dataset>/res_experiment2.csv``. The latter file allows to verify the results of the experiment as reported in Table 4, except for the rows of results taken from the GIP paper [1].

The predictions and classification performance .csv files occupy less than 1 MB. The largest model, stored in the ``/trained_models/`` folder after unzipping, occupies 200 MB. Each model is unzipped from its corresponding archive into the directory ``trained_models/<dataset_name>/2-class/<model_name>``. This directory is removed after running each model. 

Running instructions:
1. Modify the variable ``IMAGEPRIVACY_DIR`` in the file [scripts/run_experiment2.sh](scripts/run_experiment2.sh) by placing the path to the folder where you downloaded the datasets.  
2. Place the path to the folder where you downloaded the datasets in the field ``data_dir`` of the file [configs/datasets.json](configs/datasets.json).
3. Open the terminal in the working directory of the repository.
4. Run: ``source scripts/run_experiment2.sh``

Note that the prediction file is overwritten after each model as the model name is the same.

This experiment supports the claims in the Main Results 2.

#### Experiment 3: Comparative analysis of methods for image privacy classification 

This experiment reproduces the results presented in Table 5 of the article, Sec.6.9 (Comparative analysis). We provide a bash script that unzips the archive of each model listed in the table and runs these models on the testing sets of both PrivacyAlert and IPD datasets. The predictions of each model are saved into ``results/<dataset>/<model-name>.csv``. Classification performance are also computed and saved into ``results/<dataset>/<model-name>.csv``. The latter file allows to verify the results of the experiment as reported in Table 5.

The predictions and classification performance .csv files occupy less than 1 MB.  Each model is unzipped from its corresponding archive into the directory ``trained_models/<dataset_name>/2-class/<model_name>``. This directory is removed after running each model. 

Running instructions:
1. Modify the variable ``IMAGEPRIVACY_DIR`` in the file [scripts/run_experiment3.sh](scripts/run_experiment3.sh) by placing the path to the folder where you downloaded the datasets.  
2. Place the path to the folder where you downloaded the datasets in the field ``data_dir`` of the file [configs/datasets.json](configs/datasets.json).
3. Open the terminal in the working directory of the repository.
4. Run: ``source scripts/run_experiment3.sh``

Note that the prediction file is overwritten after each model as the model name is the same.

This experiment supports the claims in the Main Results 1.

#### Experiment 4: Comparative analysis of additional methods for image privacy classification

This experiment reproduces the results presented in Table 7 of the article (Appendix C). We provide a bash script that download the already trained models listed in the table and runs these models on the testing sets of both PrivacyAlert and IPD datasets. The predictions of each model are saved into ``results/<dataset>/<model-name>.csv``. Classification performance are also computed and saved into ``results/<dataset>/res_experiment4.csv``. The latter file allows to verify the results of the experiment as reported in Table 7, except the two rows whose results are taken from Zhao et al.'s evaluation on PrivacyAlert [2].

Running this script takes approximately 15 minutes. The predictions and classification performance .csv files occupy less than 1 MB. The largest model, stored in the ``/trained_models/`` folder after unzipping, occupies 200 MB. Each model is unzipped from its corresponding archive into the directory ``trained_models/<dataset_name>/2-class/<model_name>``. This directory is removed after running each model. 

Running instructions:
1. In the file [scripts/run_experiment4.sh](scripts/run_experiment4.sh), modify the variable ``IMAGEPRIVACY_DIR`` by placing the path to the folder where you downloaded the datasets.
2. In the file [configs/datasets.json](configs/datasets.json), alss place the path to the folder where you downloaded the datasets in the field ``data_dir``.
3. Open the terminal in the working directory of the repository.
4. Run: ``source scripts/run_experiment4.sh``

Note that some of the models are run with different configurations and therefore their predictions file are overwritten with the most recent configuration. 

This experiment supports the claims in the Main Results 3 and Main Results 4.

## Limitations <a name="limitations"></a>

See disclaimer in [Curated image privacy data sets](#datasets) related to the problem of obtaining the datasets ("429 Too Many Requests" status code when downloading images from Flickr, and missing images), making the results of the artifact hard to reproduce and requiring other researchers to privately request the images.

The software was designed and developed to also favour reproducibility of the training pipeline of the various models. However, also including the reproducibility of training of the various models is time-consuming during the review process (especially for large models such as GIP [1]). 

## Notes on Reusability <a name="reusability"></a>

This artifact (source code) is a general framework that contains:
* pipelines for training and testing models on publicly available datasets for image privacy (pipelines are modular and depend on the input information, e.g., only image or graph data); 
* module that loads image privacy datasets with a unified format based on our curation and adapted for either type of input information;
* toolkit to evaluate the model predictions with respect to the datasets annotations as a binary classification task;
* module loading multiple models in an agnostic way to the pipelines.

Other researchers can:
* reuse the full framework to re-train and evaluate the already provided models based on our configurations (see ``configs/*``);
* train and evaluate the models with new configurations for comparison and optimisation by creating customised config files;
* add, train, and test new models to the framework (see ``srcs/nets/*.py`` and [srcs/load_net.py](srcs/load_net.py));
* add new datasets following the format of the curated datasets and corresponding loading modules (see for example [srcs/datasets/imageprivacy.py](srcs/datasets/imageprivacy.py), [srcs/datasets/privacyalert_graph.py](srcs/datasets/privacyalert_graph.py), [srcs/datasets/wrapper_imgs.py](srcs/datasets/wrapper_imgs.py), [srcs/datasets/wrapper.py](srcs/datasets/wrapper.py));
* extend the framework to multi-class classification and evaluation.

Overall, the framework can enable a common and standard benchmark for image privacy classification. 

We might include further documentation on how to add new datasets, models, and components upon community requests.

## References <a name="references"></a>

[1] G. Yang, J. Cao, Z. Chen, J. Guo, and J. Li., "_Graph-based neural networks for explainable image privacy inference_", Pattern Recognition, 2020 [[link](https://doi.org/10.1016/j.patcog.2020.107360)]

[2] C. Zhao, J. Mangat, S. Koujalgi, A. Squicciarini, and C. Caragea, "_PrivacyAlert: A Dataset for Image Privacy Prediction_", Int. AAAI Conf. Web and Social Media, 2022 [[link](https://doi.org/10.1609/icwsm.v16i1.19387)]
