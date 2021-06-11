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
parser.add_argument('--batch_size', default=32)
parser.add_argument('--epochs', default=200)
parser.add_argument('--version', default=1)
parser.add_argument('--n', default=3)
parser.add_argument('--id', default=0)
parser.add_argument('--data', default=0)
args = vars(parser.parse_args())

batch_size = int(args['batch_size'])
epochs = int(args['epochs'])
version = int(args['version'])
n = int(args['n'])
identifier = int(args['id'])
data = int(args['data'])

num_classes = 2

if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

model_type = 'ResNet%dv%d_%d_data%d_sanity_check' % (depth, version, identifier, data)
save_dir = os.path.join(os.getcwd(), 'saved_models')
os.mkdir('saved_models/' + model_type)

old_stdout = sys.stdout
log_file = open(save_dir + "/" + model_type + "/log.txt", "w")
sys.stdout = log_file

x_train = np.load('data_sanity_check/x_train.npy')
x_test = np.load('data_sanity_check/x_test.npy')
y_train = np.load('data_sanity_check/y_train.npy')
y_test = np.load('data_sanity_check/y_test.npy')

input_shape = x_train.shape[1:]

if data == 0:
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

if version == 2:
    model = resnet.resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
else:
    model = resnet.resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=resnet.lr_schedule(0)),
              metrics=['acc', 'mse'])
model.summary()

print("Model:", model_type)

model_name = 'cifar10_%s_model_sanity_check.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=save_dir + "/" + model_type + "/" + model_name,
                             monitor='val_mse',
                             verbose=1,
                             save_best_only=False)

lr_scheduler = LearningRateScheduler(resnet.lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)


class PlottingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        preds = self.model.predict(x_test)
        plt.clf()
        plt.hist(preds[:, 0])
        plt.title("Test Set Distribution, Epoch " + str(epoch) + ", MSE " + str(round(logs['mse'], 2)) + ", VAL MSE " + str(round(logs['val_mse'], 2)))
        plt.savefig(save_dir + "/" + model_type + "/" + "test_dist_" + str(epoch))       
        plt.clf()
plotting_callback = PlottingCallback()

callbacks = [checkpoint, lr_reducer, lr_scheduler, plotting_callback]

print('Using real-time data augmentation.')
# this will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

datagen.fit(x_train)

t1 = time.time()

steps_per_epoch =  math.ceil(len(x_train) / batch_size)
history = model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),
          verbose=1,
          epochs=epochs,
          validation_data=(x_test, y_test),
          steps_per_epoch=steps_per_epoch,
          callbacks=callbacks)

t2 = time.time()
print("*******Train time" + str((t2 - t1)/60.))

with open(save_dir + "/" + model_type  + "/history" + '.pkl', 'wb') as f:
    pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)


# score trained model
scores = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

