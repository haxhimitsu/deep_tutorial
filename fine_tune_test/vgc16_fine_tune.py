##################################################################
# memo
# it1915 Hachimine Takumi
#ディレクトリ構成  
#
##################################################################
import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras.callbacks
from keras.models import Sequential, model_from_json
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input
import numpy as np
import cv2
import os
import csv
import copy
import random

#################setting GPU useage#####################

config = tf.ConfigProto(
      gpu_options=tf.GPUOptions(
          per_process_gpu_memory_fraction=0.8, # 最大値の80%まで
          allow_growth=True # True->必要になったら確保, False->全部
      )
    )
sess = sess = tf.Session(config=config)

#####################################################
EPOCHS = 100

h5 = 'CNN_weight_2_4.h5' #使用する重みファイル名(存在しないファイル名を書くと学習を開始し，その名前で重みファイルを保存)
TrainIMG = []
TrainLABEL = []
ValIMG = []
ValLABEL = []
TestIMG = []
TestLABEL = []

img_dirs = ['edge','fuchaku','haikei','kuro']#データセットを保存しているフォルダ名 可変 
img_dirs2=['test2']
img_dirs3=['testy']
filename='result_grindstone2-test.csv'
label = 0

result_count = []
all_count = 0

img_rows, img_cols = 150, 150
#####################################################




################モデル作成############################
# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=(img_rows, img_cols, 3))
conv_base = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
# vgg16.summary()

    # FC層を構築
    # Flattenへの入力指定はバッチ数を除く
model = models.Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(4,activation='softmax'))

# 学習済みのFC層の重みをロード
# top_model.load_weights(os.path.join(result_dir, 'bottleneck_fc_model.h5'))

    # VGG16とFCを接続
#model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # 最後のconv層の直前までの層をfreeze

conv_base.trainable = False


# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


model.summary()