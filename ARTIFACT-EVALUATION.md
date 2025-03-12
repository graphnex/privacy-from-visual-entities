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

## Description <a name="description"></a>

This artifact contains the source code of the framework for training, testing, and evaluating the models for image privacy classification used in the article: A. Xompero and A. Cavallaro, "Learning Privacy from Visual Entities", Proceedings on Privacy Enhancing Technologies (PoPETs), Volume 2025, Issue 3, 2025 (to appear).

### Security/Privacy Issues and Ethical Concerns <a name="security-ethical"></a>

The software enables the training and testing of various learning-based models for predicting an image as public or private. Some of the models learn to predict this label directly from the image as input. Other models have intermediate steps that use pre-trained model for recognising object types or scene types in the image and then these visual entities are used as input to another model trained for predicting the privacy label. The software includes the loading of exisiting public datasets for image privacy, such as PrivacyAlert and Image Privacy Dataset (IPD), that contain sensitive photos even if the original images were provided under Public Domain license by the users on Flickr. 

Potential ethical concerns of the trained model are direclty related to the data used for training. Models achieve limited classification performance on the testing sets of the dataset and hence they should not be considered reliable for general use.

## Basic Requirements <a name="basic-requirements"></a>

### Hardware Requirements <a name="hardware-requirements"></a>

The software was run on a Linux-based machine using one NVIDIA GeForce GTX 1080 Ti GPU with 12 GB of RAM (CUDA 10.2). 

We expect the software to successfully run on a machine/workstation with similar or higher requirements. 

### Software Requirements <a name="software-requirements"></a>

The software was run on a Linux-based machine (CentOS 6).

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

**Note**: Following links to the curated image privacy data sets will be updated with the corresponding Zenodo links once the archives are uploaded in a Zenodo repository/record to comply with FAIR principles and Open Research.

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
wget http://www.eecs.qmul.ac.uk/~ax300/privacy-from-visual-entities/curated_imageprivacy_datasets.zip

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

### Pre-computed visual entitities <a name="data-visual-entities"></a>

Some of the models run their pipeline end-to-end with the images as input, whereas other models require different or additional inputs.
These inputs include the pre-computed visual entities (scene types and object types) represented in a graph format, e.g. for a Graph Neural Network.

For each image of each dataset, namely PrivacyAlert, PicAlert, and VISPR, we provide the predicted scene probabilities as a .csv file , the detected objects as a .json file in COCO data format, and the node features (visual entities already organised in graph format with their features) as a .json file. 
For consistency, all the files are already organised in batches following the structure of the images in the datasets folder.
For each dataset, we also provide the pre-computed adjacency matrix for the graph data. 

**Note**: IPD is based on PicAlert and VISPR and therefore IPD refers to the scene probabilities and object detections of the other two datasets. Both PicAlert and VISPR must be downloaded and prepared to use IPD for training and testing. 

The table below provides the link to the archive file for each dataset and each visual entity type.

| Type | PicAlert | VISPR | PrivacyAlert | IPD |
|------|----------|-------|--------------|-----|
| Scenes | [link]() | [link]() | [link]() | N/A |
| Objects | [link]() | [link]() | [link]() | N/A |
| Graph data | [link]() | [link]() | [link]() | [link]() |

These files should be unzipped in the folder ``/resources/`` and follow this structure:
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

### Trained models <a name="trained-models"></a>

This section includes the link to a zip file related to each model trained in the article. 
Models can be extracted in the folder ``trained_models`` and then used within the testing pipeline to obtain the predictions on the testing sets of the datasets avoiding to re-train the model for scratch.

**Disclaimer** Trained models are not yet available (empty links) and under preparation for release on a Zenodo record to comply with FAIR principles and Open Research.

<details>
<summary> Show table of models and links for GIP and GPA analysis (Section 6.7, Tables 3 and 4) </summary>

| Model | Configuration | PrivacyAlert | IPD |
|-------|---------|--------------|-----|
| GIP   | 1.0     | [link]()     | [link]() |
| GIP   | 2.0     | [link]()     | [link]() |
| GIP   | 2.1     | [link]()     | [link]() |
| GIP   | 2.2     | [link]()     | [link]() |
| GIP   | 2.3     | [link]()     | [link]() |
| GIP   | 2.4     | [link]()     | [link]() |
| GIP   | 2.5     | [link]()     | [link]() |
| GPA   | 1.0     | [link]()     | [link]() |
| GPA   | 1.1     | [link]()     | [link]() |
| GPA   | 1.2     | [link]()     | [link]() |
| GPA   | 1.3     | [link]()     | [link]() |
| GPA   | 1.4     | [link]()     | [link]() |
| GPA   | 1.5     | [link]()     | [link]() |
| GPA   | 1.6     | [link]()     | [link]() |
| GPA   | 1.7     | [link]()     | [link]() |
| GPA   | 1.8     | [link]()     | [link]() |

</details>

<details>
<summary> Show table of models and links for comparative analysis (Section 6.9, Table 5) </summary>

| Model | Configuration | PrivacyAlert | IPD |
|-------|---------|--------------|-----|
| MLP-I   | 1.0     | [link]()     | [link]() |
| MLP   | 1.0     | [link]()     | [link]() |
| GA-MLP   | 1.0     | [link]()     | [link]() |
| GIP   | 2.2     | [link]()     | [link]() |
| GPA   | 1.4     | [link]()     | [link]() |
| S2P   | 1.0     | [link]()     | [link]() |

</details>

<details>
<summary> Show table of models and links for additional comparisons (Appendix C, Table 7) </summary>


| Model | Configuration | PrivacyAlert | IPD |
|-------|---------|--------------|-----|
| TAGSVM   | 1.0     | [link]()     | [link]() |
| RNP2SVM   | 1.0     | [link]()     | [link]() |
| RNP2SVM   | 1.2     | [link]()     | [link]() |
| RNP2FT   | 1.0     | [link]()     | [link]() |
| S2P   | 1.0     | [link]()     | [link]() |
| S2P_MLP   | 1.0     | [link]()     | [link]() |
| S2P_MLP   | 1.1     | [link]()     | [link]() |

</details>

<details>
<summary> Show table of models and links for the analysis of different design choices of MLP (Appendix D.1, Table 8) </summary>

| Model | Configuration | PrivacyAlert | IPD |
|-------|---------|--------------|-----|
| MLP   | 1.0     | [link]()     | [link]() |
| MLP   | 1.1     | [link]()     | [link]() |
| MLP   | 1.2     | [link]()     | [link]() |
| MLP   | 1.3     | [link]()     | [link]() |
| MLP   | 1.4     | [link]()     | [link]() |
| MLP   | 1.5     | [link]()     | [link]() |

</details>

<details>
<summary> Show table of models and links for hyperparameter analysis (Appendix D.2, Figure 9) </summary>

| Model | Configuration | PrivacyAlert | IPD |
|-------|---------|--------------|-----|
| MLP   | 2.0     | [link]()     | [link]() |
| MLP   | 2.1     | [link]()     | [link]() |
| MLP   | 2.2     | [link]()     | [link]() |
| MLP   | 2.3     | [link]()     | [link]() |
| MLP   | 2.4     | [link]()     | [link]() |
| MLP   | 2.5     | [link]()     | [link]() |
| MLP   | 2.6     | [link]()     | [link]() |
| MLP   | 2.7     | [link]()     | [link]() |
| MLP   | 2.8     | [link]()     | [link]() |
| MLP   | 2.9     | [link]()     | [link]() |
| MLP   | 2.10     | [link]()     | [link]() |
| MLP   | 2.11     | [link]()     | [link]() |
| MLP   | 2.12     | [link]()     | [link]() |
| MLP   | 2.13     | [link]()     | [link]() |
| MLP   | 2.14     | [link]()     | [link]() |
| MLP   | 2.15     | [link]()     | [link]() |
| MLP   | 2.16     | [link]()     | [link]() |
| MLP   | 2.17     | [link]()     | [link]() |
| MLP   | 2.18     | [link]()     | [link]() |
| MLP   | 2.19     | [link]()     | [link]() |
| MLP   | 2.20     | [link]()     | [link]() |
| MLP   | 2.21     | [link]()     | [link]() |
| MLP   | 2.22     | [link]()     | [link]() |
| MLP   | 2.23     | [link]()     | [link]() |
| MLP   | 2.24     | [link]()     | [link]() |
| MLP   | 2.25     | [link]()     | [link]() |
| MLP   | 2.26     | [link]()     | [link]() |
| MLP   | 2.27     | [link]()     | [link]() |
| MLP   | 2.28     | [link]()     | [link]() |
| MLP   | 2.29     | [link]()     | [link]() |
| MLP   | 2.30     | [link]()     | [link]() |
| MLP   | 2.31     | [link]()     | [link]() |
| MLP   | 2.32     | [link]()     | [link]() |
| MLP   | 2.33     | [link]()     | [link]() |
| MLP   | 2.34     | [link]()     | [link]() |
| MLP   | 2.35     | [link]()     | [link]() |
| MLP   | 2.36     | [link]()     | [link]() |

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

The repository is hosted in GitHub at https://github.com/graphnex/privacy-from-visual-entities (use the most recent commit in the branch main).

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

No available tests in the current status of the repository.

## Artifact Evaluation <a name="evaluation"></a> 

### Main Results and Claims <a name="results-claims"></a> 
List all your paper's results and claims that are supported by your submitted artifacts.

#### Main Result 1: Name
Describe the results in 1 to 3 sentences.
Refer to the related sections in your paper and reference the experiments that support this result/claim.

#### Main Result 2: Name
...

### Experiments <a name="experiments"></a>
List each experiment the reviewer has to execute. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes and how much space it consumes on disk. (approximately)
 - Which claim and results does it support, and how.

These experiments simply reproduce the results reported in the article using the already trained models. 
We do not include details on how to reproduce the training of the models and reviewers are not asked to run the training pipeline of the models.

#### Experiment 1: Evaluation of various design choices for the GIP model
Provide a short explanation of the experiment and expected results.
Describe thoroughly the steps to perform the experiment and to collect and organize the results as expected from your paper.
Use code segments to support the reviewers, e.g.,
```bash
python experiment_1.py
```

 

#### Experiment 2: Evaluation of various design choices for the GPA model
...

#### Experiment 3: Comparative analysis of methods for image privacy classification 

This experiment reproduces the results presented in Table 5 of the article. We provide a bash script that downloads the already trained models listed in the table and runs these models on the testing sets of both PrivacyAlert and IPD datasets. The predictions of each model are saved into ``results/<dataset>/<model-name>.csv``. Classification performance are also computed and saved into ``results/<dataset>/<model-name>.csv``. The latter file allows to verify the results of the experiment as reported in Table 5.

Running this script takes XXX. The predictions and classification performance .csv files occupy less than 1 MB. The models, stored in ``/trained_models/`` folder, occupy at maximum 200 MB.

#### Experiment 4: Comparative analysis of additional methods for image privacy classification

This experiment reproduces the results presented in Table 9 of the article (Appendix B.3). We provide a bash script that download the already trained models listed in the table and runs these models on the testing sets of both PrivacyAlert and IPD datasets. The predictions of each model are saved into ``results/<dataset>/<model-name>.csv``. Classification performance are also computed and saved into ``results/<dataset>/res_experiment4.csv``. The latter file allows to verify the results of the experiment as reported in Table 9, except the two rows whose results are taken from Zhao et al.'s evaluation on PrivacyAlert [62].

Running this script takes approximately 15 minutes. The predictions and classification performance .csv files occupy less than 1 MB. The largest model, stored in the ``/trained_models/`` folder after unzipping, occupies 200 MB. Each model is unzipped from its corresponding archive into the directory ``trained_models/<dataset_name>/2-class/<model_name>``. This directory is removed after running each model. 

Running instructions:
1. In the file ``source scripts/run_experiment4.sh``, modify the variable ``IMAGEPRIVACY_DIR`` by placing the path to the folder where you downloaded the datasets.
2. In the file ``configs/datasets.json``, alos place the path to the folder where you downloaded the datasets in the field ``data_dir``.
3. Open the terminal in the working directory of the repository.
4. Run: ``source scripts/run_experiment4.sh``

Note that some of the models are run with different configurations and therefore their predictions file are overwritten with the most recent configuration. 

#### Experiment 5: MLP variants
...

#### Experiment 6: MLP hyper-parameters analysis
...


#### Experiment 7: GA-MLP variants
...

## Limitations <a name="limitations"></a>

The datasets are originally provided by other sources and have been re-organised and curated for this work. Similar to the original datasets (PicAlert and PrivacyAlert), we provide the link to the images in the download scripts, however running the scripts can incur in the "429 Too Many Requests" status code. This makes the datasets hard to obtain from the original Flickr locations, and thus impacting the testing of the reproducibility of the experiments. Moreover, owners of the photos on Flick could have removed the photos from the social media platform, resulting in less images than those used for training and testing the models in the article. This means that other researchers will need to privately request the current version of the datasets as used in the article to reproduce the results or make fair comparisons.

The software was designed and developed to also favour reproducibility of the training pipeline of the various models. However, also including the reproducibility of training of the various models is time-consuming during the review process (especially for large models such as GIP). 

## Notes on Reusability <a name="reusability"></a>

This artifact (source code) is a general framework that contains:
* pipelines for training and testing models on publicly available datasets for image privacy (pipelines are modular and depend on the input information, e.g., only image or graph data); 
* module that loads image privacy datasets with a unified format based on our curation and adapted for either type of input information;
* toolkit to evaluate the model predictions with respect to the datasets annotations as a binary classification task;
* module loading multiple models in an agnostic way to the pipelines.

Other researchers can:
* reuse the full framework to re-train and evaluate the already provided models based on our configurations (see ``configs/*``);
* train and evaluate the models with new configurations for comparison and optimisation by creating customised config files;
* add, train, and test new models to the framework (see ``srcs/nets/*.py`` and ``srcs/load_net.py``);
* add new datasets following the format of the curated datasets and corresponding loading modules (see for example ``srcs/datasets/imageprivacy.py``, ``srcs/datasets/privacyalert_graph.py``, ``srcs/datasets/wrapper_imgs.py``, ``srcs/datasets/wrapper.py``);
* extend the framework to multi-class classification and evaluation.

Overall, the framework can enable a common and standard benchmark for image privacy classification. 

We might include further documentation on how to add new datasets, models, and components upon community requests.