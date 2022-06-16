# zebrafish-classifier
The repository contains the code used to train, evaluate and visualize the results of zebrafish embryo classifiers from "Identifying zebrafish segmentation phenotype features using multiple instance learning".

We trained the classifiers on zebrafish embryos which are either WT (nonperturbed) or have a disruption in the segmentation clock (her1;her7, tbx6_fss and DAPT classes). The specific type of the disruption is hard to be discerned based on the in situ hibridization images of the zebrafish embryo. For example:

![DAPT](https://github.com/glig95/zebrafish-classifier/examples_of_images/DAPT/2_DAPT50ISC_2.png)

The three main Jupyter notebooks in the repository are used for training three different classifiers:
1. the baseline, shallow convolutional classifier
2. the transfer-learning-, and MobileNetV2-based, classifier
3. the attention-based multiple-instance-learning classifier.

Each of the notebooks contains the code to train and evaluate the classifier (construct confusion matrix and calculate the accuracy of classification).
The evaluation of the second classifier is followed by a code to generate class-activation maps and quantify the contribution of fish parts to it.
Similarly, in the case of attention-based multiple instance learning, we calculate normalized attention given to each of the fish parts.

The sets used for training and evaluation of the classifiers, which were the same for all classifiers, can be downloaded from Zenodo: https://zenodo.org/record/6651752. Since the number of images in the training set was different from class to class, we use weights normalization during training. Validation sets are balanced and consist of 20 images from each class.

Lastly, the "Code_for_plotting_figures" folder contains the code for plotting the three main figures of the manuscript.

The code is written in Python (version 3.6.5) using Jupyter notebooks as the primary working environment. Neural networks were implemented in the Tensorflow library (version 2.3.0).
