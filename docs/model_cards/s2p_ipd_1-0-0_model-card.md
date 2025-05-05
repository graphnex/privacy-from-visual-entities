# Model Card for S2P-IPD-1.0.0

Scene-to-privacy (S2P) is a deep learning model for image privacy classification. The model is based on a convolutional neural network that takes an RGB image as input and outputs a binary variable to classify an image as either private or public. This model is trained and tested on the Image Privacy Dataset (IPD).

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Dr. Alessio Xompero
- **Funded by:** the CHIST-ERA programme through the project GraphNEx, under UK EPSRC grant EP/V062107/1
- **Shared by:** 
    * Dr. Alessio Xompero (Queen Mary University of London)
    * Prof. Andrea Cavallaro (Idiap Research Institute and École Polytechnique Fédérale de Lausanne)
- **Model type:** Learning method
- **License:** [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) (CC-BY-NC 4.0)
- **Version:** 1.0.0 

*Note on model versioning*

In a similar way to semantic versioning for software, the semantic versioning used in this card follows 3 components:
- *Major*: the model had a change in its architecture (but not so subtantial for being called in a different way).
- *Config*: index of the configuration mode. The model is re-trained with different values for the hyper-parameters and stored in a corresponding configuration file (e.g. ``configs/s2p_1.0.0.json``)
- *Run*: index of the run to train the model with the same configuration file (e.g. for reproducbility checks)

### Model Sources

- **Repository:** https://github.com/graphnex/privacy-from-visual-entities
- **Paper:** https://doi.org/10.48550/arXiv.2503.12464 
- **Demo:** https://github.com/graphnex/privacy-from-visual-entities/demo

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The primary intended users of this model are AI academic researchers, scholars, and practitioners working on computer vision, machine learning, deep learning, and visual privacy. The primary intended uses of S2P are:
* Research on designing and developing models for recognising images as private (image privacy classification)
* Benchmarking and comparing models for image privacy classification on the Image Privacy Dataset

### Downstream Use 

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The model could be plugged into an app to alert a user of disclosing or sharing an image that can be potentially private (see also out-of-scope use and limitations). The model could be further fine-tuned on user's data for personalisation, subject to annotations of the images by the users, to provide a layer of privacy detection before sharing or uploading images online. 

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->


## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

S2P is influenced by the predicted scenes in the backbone component and therefore is not biased towards the presence of people to determine if an image is private. The model is also not biased towards only outdoor images or indoor images as both scene types are predicted as either public or private. However, there are indoor environments that the model learned to predict as private (e.g. bathroom). There are scenes, such as religious environments/ceremonies and escape rooms, that
are ambiguously predicted as either public or private, denoting that the model also rely on the features related to other scene types to make the decision. Office-like scenes (e.g., person in the classroom, person presenting next to a laptop and screen) are predicted as public, but these images may also denote a private setting. 

S2P fails to recognise private images of an individual in various contexts (beach, gym). These are places that can commonly be related to a public context and, based on the predictions, the model learned to associate these scenes to the public label. However, the model cannot distinguish when this scene type is public or private due to the lack of additional information, such as the person’s presence. Simply relying on the scene information is insufficient in various cases. 

Further details and examples of misclassifications by S2P are reported in the [paper](https://arxiv.org/pdf/2503.12464) (Section 6.10, pages 12-13).

The model is also highly biased by the annotations in the dataset is trained on. These annotations are often subjective, affected by instructions provided to the annotations, and therefore potentially incorrect (see Appendix A, pages 15-16, in the paper). 

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Accuracy performance of the model on the testing set of the same dataset where the model is trained on is also limited. We therefore recommend avoiding its use as the model as it is can provide misleading predictions. Personalisation has not yet been verified with this model and therefore unwanted or not satisfactory behaviours should be expected. 

## How to Get Started with the Model

Use the code below to get started with the model.

See [ARTIFACT-EVALUATION](https://github.com/graphnex/privacy-from-visual-entities/blob/main/ARTIFACT-EVALUATION.md#testing-environment) in the repo for more information.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

*Image Privacy Dataset (IPD)* 

The datasets refers to images publicly available on Flickr. 
These images have a large variety of content, including sensitive content, semi-nude people, vehicle plates, documents, private events. 
Images were annotated with a binary label denoting if the content was deemed to be public or private. 
As the images are publicly available, their label is mostly public; however, there is a ∼33% of the images labelled as private. 
The dataset has a high imbalance towards the public class. 

IPD combines two other existing datasets, PicAlert and part of VISPR, to increase the number of private images already limited in PicAlert.

IPD (or works using the dataset) does not provide reproducible information on how data was previously split into training, validation, and testing sets. 
The data was therefore randomly split into training, validation, and testing sets using a K-Fold stratified strategy (with K=3).
Only the first fold was considered to train and evaluate the model. 

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

The ResNet-50 is pre-trained on Places365. 
The parameters of the trainable fully connected layer are initialised with Xavier uniform distribution and zero bias.
The input image is resized to a resolution of 448×448 pixels and values are normalised with respect to the statistics computed on ImageNet. 


#### Training Hyperparameters

- **Seed for reproducibility**: 789
- **Maximum number of epochs**: 1000
- **Batch size (training)**: 100
- **Batch size (validation)**: 100
- **Optimizer** : Adam
- **Initial learning rate**: 0.001
- **Reducing factor for the learning rate**: 0.5
- **Patience for scheduling learning rate**: 10
- **Minimum learning rate**: 1e-5
- **Maximum learning rate**: 0.1
- **Weight decay**: 0.0
- **Momentum** : 0.9
- **Maximum traininig time before stopping**: 12 h
- **Training mode** : crossval (pre-computed stratified K-Fold splits of the dataset)
- **Fold ID** : 0
- **Performance measure monitored for early stopping**: balanced_accuracy 
- **use_bce** : false (use of the binary cross-entropy instead of the cross-entropy with 2 classes)

The maximum number of epochs is set to 1,000, but the training stopped early if the learning rate is reduced to a value lower than 0.00001 or the training time lasted longer than 12 h. 

See also the corresponding configuration file: https://github.com/graphnex/privacy-from-visual-entities/configs/s2p_v1.0.json

Additional details on the training procedure can be found in in the [paper](https://arxiv.org/pdf/2503.12464) (Sec.6.5, page 8).

#### Speeds, Sizes, Times 

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

* Number of trainable parameters: 24,256,649
* Number of optimised parameters (for privacy): 732
* Storage size (checkpoint): ~97 MB
* Number of epochs the model was trained for before early stopping: 117


## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

*Image Privacy Dataset (IPD)*

The testing set of IPD has 6,912 images: 2,304 images are labelled as private and 4,608 images are labelled as public.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

The model is evaluated using per-class precision, per-class recall, per-class F1-score, overall precision, overall recall (or balanced accuracy), and accuracy. 

First, the number of true positives (TP), false positives (FP), and false negatives (FN) are computed for each class 𝑦.

For a given class, precision is the number of images correctly classified as class 𝑦 over the total number of images predicted as the class $y$: 
$P_y = TP_y /(TP_y + FP_y )$.

Recall is the number of images correctly classified as class 𝑦 over the total number of images annotated as class $y$: 
$R_y = TP_y /(TP_y + FN_y )$.

Accuracy is the total number of images that are correctly classified as either public or private over the total number of images that are correctly classified, wrongly predicted ($FP_y$ ) and missed to be predicted ($FN_y$) with respect to the annotated class:
$𝐴𝐶𝐶 = \sum_𝑦 𝑇𝑃_𝑦 /( \sum_y (𝑇𝑃_𝑦 + 𝐹𝑃_𝑦 + 𝐹𝑁_𝑦))$

Balanced accuracy is the main performance measure to better assess the class imbalance of the dataset, and is the average between the recall of the two classes. Similarly, overall precision is the average between the precision of the two classes. 

Given the semantics of the task, particular emphasis is given to the recall for the private class.

### Results

| Class | Precision | Recall | F1-score | Accuracy |
|-------|-----------|--------|----------|----------|
| Private | 75.83   |  72.44 | 74.10    |          |
| Public  | 86.52   |  88.45 | 87.48    |          |
| Overall | 81.18   |  80.45 |          |   83.12  |

Note that for Overall, Recall and Balanced Accuracy are the same. 

Results and metrics computed with scikit-learn library.

#### Summary

Using transfer learning with a CNN suffices to achieve the highest classification performance on Image Privacy Dataset. By adding a fully connected layer while keeping all the parameters of the CNN fixed and pre-trained on a different source dataset, this simple model only optimises a small number of parameters for the image privacy task (732 instead of 14,000 or 500 millions). On the contrary, end-to-end training of graph-based methods can mask the contribution of individual components to the classification performance. Performance gains of graph-based models are only marginally due to the graph component and mostly due to the fine-tuning of the CNNs. 

Further details on the results and discussion can be found in the [paper](https://arxiv.org/pdf/2503.12464).

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

- **Hardware Type:** GTX 1090 Ti
- **Hours used:** ~8
- **Cloud Provider:** Private infrastructure
- **Compute Region:** United Kingdom
- **Carbon Emitted:** 0.62

Carbon emissions estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

Note: we use the factor ((kgCO2e per kWh)) 0.3072 for UK in the source https://www.carbonfootprint.com/docs/2018_8_electricity_factors_august_2018_-_online_sources.pdf

## Technical Specifications

### Model Architecture and Objective

**Objective**: recognising private images with a learning-based model trained on a set of images that are publicly available and manually annotated, while overcoming the problems of limited training data and class imbalance. To this end, we consider a pre-trained Convolutional Neural Network and apply transfer learning by keeping the parameters fixed to recognise visual entities (e.g., scenes). 

**Model architecture**: A ResNet-50 is used as a Convolutional Neural Network pre-trained on Places365 for scene recognition. Instead of applying the sigmoid function to the logits outputted by the model for all pre-defined scenes, we add a trainable fully  connected layer (scene-to-privacy layer, or S2P) that transforms the scene logits to two logits, followed by softmax to obtain the probability distribution of private and public classes. The parameters of this layer (732) are randomly initialised and then optimised during the end-to-end training of the model.

### Compute Infrastructure

Linux-based compute servers available by the School of Electronic Engineering and Computer Science at Queen Mary University of London (United Kingdom). 

#### Hardware

Linux-based machine using CentOS Linux release 7.7.1908, and one NVIDIA GeForce GTX 1080 Ti GPU with 12 GB of RAM (CUDA 10.2).

#### Software

Model was implemented in Python and PyTorch. 

* Python >3.9 
* [PyTorch](https://pytorch.org/) 1.13.1

Other libraries:
* Numpy 1.22
* scikit-learn 1.0.1
* Pandas 1.5.3
* Scipy
* Pandas
* [Networkx](https://networkx.org/)
* json
* Matplotlib
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) (PyG)

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

## Citation

**BibTeX:**
```
@Article{Xompero2025PoPETs,
    title = {Learning Privacy from Visual Entities},
    author = {Xompero, A. and Cavallaro, A.},
    journal = {Proceedings on Privacy Enhancing Technologies},
    volume = {2025},
    number = {3},
    pages={1--21},
    month = {Mar},
    year = {2025},
}
```

**APA:**
```
A Xompero, A. Cavallaro, "Learning Privacy from Visual Entities", Proceedings on Privacy Enhancing Technologies (PoPETs), vol. 2025, n.3, March 2025
```

## Model Card Authors

* Dr. Alessio Xompero (Queen Mary University of London)
* Prof. Andrea Cavallaro (Idiap Research Institute and École Polytechnique Fédérale de Lausanne)

## Model Card Contact

Please raise an issue at https://github.com/graphnex/privacy-from-visual-entities/issues
