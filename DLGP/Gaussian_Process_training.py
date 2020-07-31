#!/usr/bin/env python
# coding: utf-8

import pickle
import gzip
import numpy as np
import scipy
import pandas as pd
import tensorflow as tf
import keras
import keras.layers as layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Dense, Lambda, Conv1D, Conv2D, AveragePooling2D, AveragePooling1D, Flatten, MaxPooling2D, MaxPooling1D, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.applications import imagenet_utils, mobilenet_v2
from keras import backend as K
from time import time
from keras import losses
from sklearn.metrics import  roc_curve, roc_auc_score, classification_report, confusion_matrix
import glob
from PIL import Image
import h5py
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import sklearn.gaussian_process as gp
import pandas as pd
import pickle
from joblib import dump, load


with h5py.File("/home/stoledoc/work/datanfs/stoledoc/jama16-retina-replication-master/inceptionv3_features_kaggle_train.hdf5", "r") as datafile:
    features_kaggle_train = datafile["features"][:]
    retinopathy_kaggle_train = datafile["Retinopathy"][:]

df_train = pd.read_csv('/home/stoledoc/work/datanfs/stoledoc/jama16-retina-replication-master/kaggle_train_levels.csv', header = None)

levels_kaggle_train = np.asarray(df_train[1])

features_kaggle_train_0 = features_kaggle_train[np.where(levels_kaggle_train == 0)]
features_kaggle_train_1 = features_kaggle_train[np.where(levels_kaggle_train == 1)]
features_kaggle_train_2 = features_kaggle_train[np.where(levels_kaggle_train == 2)]
features_kaggle_train_3 = features_kaggle_train[np.where(levels_kaggle_train == 3)]
features_kaggle_train_4 = features_kaggle_train[np.where(levels_kaggle_train == 4)]

features_kaggle_train_0_selected = features_kaggle_train_0[np.random.choice(features_kaggle_train_0.shape[0], size = 12660, replace = False),:]
features_kaggle_train_1_selected = features_kaggle_train_1
features_kaggle_train_2_selected = features_kaggle_train_2
features_kaggle_train_3_selected = features_kaggle_train_3
features_kaggle_train_4_selected = features_kaggle_train_4

X_train = np.concatenate([features_kaggle_train_0_selected, features_kaggle_train_1_selected, features_kaggle_train_2_selected, features_kaggle_train_3_selected, features_kaggle_train_4_selected], axis = 0)

levels_kaggle_train_0 = np.zeros(12660)
levels_kaggle_train_1 = np.ones(3479)
levels_kaggle_train_2 = np.ones(12873)*2
levels_kaggle_train_3 = np.ones(2046)*3
levels_kaggle_train_4 = np.ones(1220)*4

y_train_levels = np.concatenate([levels_kaggle_train_0, levels_kaggle_train_1, levels_kaggle_train_2, levels_kaggle_train_3, levels_kaggle_train_4], axis = 0)

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))+ WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))

gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y = True, n_restarts_optimizer = 5)

gpr.fit(X_train, y_train_levels)

dump(gpr, '/home/stoledoc/work/datanfs/stoledoc/work/Kaggle_DR_3/gpr_kaggle_train_voets_inceptionv3_features_alldatawithnoise_paper.joblib')