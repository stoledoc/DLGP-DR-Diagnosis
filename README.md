# DLGP for Diabetic Retinopathy Diagnosis and Uncertainty Quantification

Implementation of the Deep Learning Gaussian Process for Diabetic Retinopathy Diagnosis:

* Santiago Toledo-Cortés, Melissa De La Pava, Oscar Perdómo, and Fabio A. González. ["Hybrid Deep Learning Gaussian Process for Diabetic Retinopathy Diagnosis and Uncertainty Quantification"].

## Abstract

Diabetic Retinopathy (DR) is one of the microvascular complications of Diabetes Mellitus, which remains as one of the leading causes of blindness worldwide. Computational models based on Convolutional Neural Networks represent the state of the art for the automatic detection of DR using eye fundus images. Most of the current work address this problem as a binary classification task. However, including the grade estimation and quantification of predictions uncertainty can potentially increase the robustness of the model. In this paper, a hybrid Deep Learning-Gaussian process method for DR diagnosis and uncertainty quantification is presented. This method combines the representational power of deep learning, with the ability to generalize from small datasets of Gaussian process models. The results show that uncertainty quantification in the predictions improves the interpretability of the method as a diagnostic support tool.

## Requirements

Python requirements:

- Python >= 3.6
- Tensorflow >= 2.0
- Pillow
- h5py
- xlrd
- scikit-learn >= 0.23.1
- matplotlib >= 2.1

## Preprocessing EyePACS and Messidor-2

Download EyePACS zip files from https://www.kaggle.com/c/diabetic-retinopathy-detection/dataRun into `./data/eyepacs`. `$ ./eyepacs.sh` to decompress and preprocess the _Kaggle_ EyePACS data set, and redistribute this set into a training and test set.

Run `$ ./messidor2.sh` to download, unpack, and preprocess the Messidor-2 data set. This data set is downloaded from the Datasets and Algorithms' section on Michael D. Abramoff's page [here](https://medicine.uiowa.edu/eye/abramoff).

## Training

## Evaluation