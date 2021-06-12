import os
import sys
import time
import math
import glob
import resnet
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

parser = argparse.ArgumentParser()
parser.add_argument('--version', default=1)
parser.add_argument('--n', default=3)
parser.add_argument('--id', default=0)
parser.add_argument('--data', default='data')

args = vars(parser.parse_args())
version = int(args['version'])
n = int(args['n'])
identifier = int(args['id'])
data = args['data']

num_classes = 2

if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

model_type = 'ResNet%dv%d_%d_%s' % (depth, version, identifier, data)
save_dir = os.path.join(os.getcwd(), 'saved_models')

x_train = np.load(data + '/x_train.npy')
x_test = np.load(data + '/x_test.npy')
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

model = resnet.resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=resnet.lr_schedule(0, 1e-3/3.)),
              metrics=['acc', 'mse'])
model.summary()

files = glob.glob(save_dir + "/" + model_type + "/" + "model_" + model_type + ".*.h5")
files = sorted(files)
checkpoint = files[-1]
checkpoint_epoch = int(checkpoint[-6:-3])

model.load_weights(checkpoint)
model = Model(inputs=model.input, outputs=model.get_layer('dense').output)
np.save(data + "/hhat_test", model.predict(x_test))
np.save(data + "/hhat_train", model.predict(x_train))
