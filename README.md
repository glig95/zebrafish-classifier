# zebrafish-classifier
The repository contains the code used to train, evaluate and visualize the results of zebrafish embryo classifiers from "Identifying zebrafish segmentation phenotype features using multiple instance learning".

We trained the classifiers on zebrafish embryos which are either WT (nonperturbed) or have a disruption in the segmentation clock (her1;her7, tbx6_fss and DAPT classes). The specific type of the disruption is hard to discern by a non-trained eye based on the in situ hibridization images of the zebrafish embryo. For example:

1. DAPT class

![2_DAPT50ISC_2](https://user-images.githubusercontent.com/20626185/174072959-7f0fe306-e1f8-4a50-9ec6-6b7ab8b0953d.png)

2. her1-;her7- class

![20190412_DMSO_7hpf_C1_gui3](https://user-images.githubusercontent.com/20626185/174072964-1a3ae32b-827b-46b0-8b2e-ad0a8117f9b6.png)

3. tbx6-fss class

![20190427_fss_fss18](https://user-images.githubusercontent.com/20626185/174072954-90e55190-1ba6-468a-9eee-1e3911cd12c9.png)

4. WT class

![2_FW_1](https://user-images.githubusercontent.com/20626185/174072962-44aa7b8d-d21a-464a-b322-c572a23c42ff.png)

We have thus trained three different classifiers that map the phenotype of the embryo into one of the four classes. The three main Jupyter notebooks in the repository are used for training three different classifiers:
1. the baseline, shallow convolutional classifier
2. the transfer-learning-, and MobileNetV2-based, classifier
3. the attention-based multiple-instance-learning classifier.

Each of the notebooks contains the code to train and evaluate the classifier (construct confusion matrix and calculate the accuracy of classification).
To identify the features of the fish used to make classification decision, the second classifier notebook contains the code to generate class-activation maps (CAM) that point to discriminatory features of the fish. Similarly, in the case of third classifier notebook, we evaluate the attention given to different fish parts. To quantify the importance of fish parts for classification of different embryo classes in case of the classifiers 2. and 3., we manually generated masks which define boundaries of major fish parts (head, trunk, tail, yolk and yolk extension).

The sets used for training and evaluation of the classifiers were kept the same for all classifiers. Since the number of images in the training set was different from class to class, we use class-weights normalization during training. Validation set is balanced and consist of 20 images from each class.

Lastly, the "code_for_plotting_figures" folder contains the code for plotting the three main figures of the manuscript.

The code is written in Python (version 3.6.5) using Jupyter notebooks as the primary working environment. Neural networks were implemented in the Tensorflow library (version 2.3.0).

To get started: 
1. Download the dataset for training and evaluation from Zenodo (https://zenodo.org/record/6651752),
2. Extract the files from 'data' folder and place 'training', 'validation', 'test' and 'fish_part_labels' folders in the same folder as the Jupyter notebooks,
4. In the command line, create a virtual environment with python 3.6.5 with the command "conda create -n ZF_classification python=3.6.5",
5. Activate the environment using "conda activate ZF_classification"
5. Install the packages required for running the notebooks that can be found in 'requirements.txt' by executing "pip install -r requirements.txt" (make sure to have pip installed),
6. Run the notebook by executing the cells from top to bottom.
