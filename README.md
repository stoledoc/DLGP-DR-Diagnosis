# DLGP for Diabetic Retinopathy Diagnosis and Uncertainty Quantification

![dlgp](https://github.com/stoledoc/Resources/blob/master/dlgp/hybrid_model.png)

Implementation of the Deep Learning Gaussian Process for Diabetic Retinopathy Diagnosis:

* Santiago Toledo-Cortés, Melissa de la Pava, Oscar Perdomo, and Fabio A. González. (2020) "Hybrid Deep Learning Gaussian Process for Diabetic Retinopathy Diagnosis and Uncertainty Quantification". Acepted at the 7th MICCAI Workshop on Ophthalmic Medical Image Analysis - OMIA7. arXiv preprint: http://arxiv.org/abs/2007.14994

## Abstract

Diabetic Retinopathy (DR) is one of the microvascular complications of Diabetes Mellitus, which remains as one of the leading causes of blindness worldwide. Computational models based on Convolutional Neural Networks represent the state of the art for the automatic detection of DR using eye fundus images. Most of the current work address this problem as a binary classification task. However, including the grade estimation and quantification of predictions uncertainty can potentially increase the robustness of the model. In this paper, a hybrid Deep Learning-Gaussian process method for DR diagnosis and uncertainty quantification is presented. This method combines the representational power of deep learning, with the ability to generalize from small datasets of Gaussian process models. The results show that uncertainty quantification in the predictions improves the interpretability of the method as a diagnostic support tool.

## Requirements

Python requirements:

- Python >= 3.5
- Tensorflow >= 2.0
- Pillow
- h5py
- xlrd
- scikit-learn >= 0.23.1
- matplotlib >= 2.1

## Preprocessing EyePACS and Messidor-2

Download EyePACS zip files from https://www.kaggle.com/c/diabetic-retinopathy-detection/data into `./data/eyepacs`. Run `$ ./eyepacs.sh` to decompress and preprocess the EyePACS data set, and redistribute it into a training and test set with gradable images

Run `$ ./messidor2.sh` to download, unpack, and preprocess the Messidor-2 data set.

## Training

Details and procedure for Inception-V3 fine-tuning with EyePACS dataset are in `./DLGP/InceptionV3_fine_tuning.ipynb`. After fine tuning, the global average pooling layer is defined as output for the feature extraction model. Extracted features are saved in an `.h5` file, and used as input for the Gaussian process. For Gaussian process training, run `$ ./DLGP/Gaussian_Process_training.py`. The model is saved in a `.joblib` file.

## Evaluation

Evaluation of the final DLGP model on EyePACS test partition and Messidor-2 is performed in `./DLGP/Evaluation.ipynb`.

## Usage

Example on how to use the DLGP to diagnose diabetic retinopathy in a single eye fundus image is provided in `./DLGP/Usage.ipynb`.