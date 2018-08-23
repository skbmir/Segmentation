import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, concatenate, MaxPool2D, UpSampling2D, Dropout, Input, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import matplotlib.pyplot as plt

IMAGE_LIB = 'data/membrane/train/image/'
MASK_LIB = 'data/membrane/train/label/'
IMG_HEIGHT, IMG_WIDTH = 256, 256
SEED = 42

all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.png']

x_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(IMAGE_LIB + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_data[i] = im

y_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(MASK_LIB + name, cv2.IMREAD_UNCHANGED).astype('float32') / 255.
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    y_data[i] = im

x_data = x_data[:, :, :, np.newaxis]
y_data = y_data[:, :, :, np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.5)

# another simple example
# input_layer = Input(shape=x_train.shape[1:])
# c1 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
# l = MaxPool2D(strides=(2, 2))(c1)
# c2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(l)
# l = MaxPool2D(strides=(2, 2))(c2)
# c3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(l)
# l = MaxPool2D(strides=(2, 2))(c3)
# c4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(l)
# l = concatenate([UpSampling2D(size=(2, 2))(c4), c3], axis=-1)
# l = Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same')(l)
# l = concatenate([UpSampling2D(size=(2, 2))(l), c2], axis=-1)
# l = Conv2D(filters=24, kernel_size=(2, 2), activation='relu', padding='same')(l)
# l = concatenate([UpSampling2D(size=(2, 2))(l), c1], axis=-1)
# l = Conv2D(filters=16, kernel_size=(2, 2), activation='relu', padding='same')(l)
# l = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(l)
# l = Dropout(0.5)(l)
# output_layer = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(l)
#
# model = Model(input_layer, output_layer)

inputs = Input(shape=x_train.shape[1:])
conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
merge6 = concatenate([drop4, up6], axis=3)
conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)

up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

model = Model(input=inputs, output=conv10)


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=['accuracy'])
# weight_saver = ModelCheckpoint('lung.h5', save_best_only=True, save_weights_only=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
hist = model.fit_generator(my_generator(x_train, y_train, 8),
                           steps_per_epoch=200,
                           validation_data=(x_val, y_val),
                           epochs=10, verbose=2,
                           callbacks=[annealer])

model_json = model.to_json()
# Записываем модель в файл
json_file = open("model1.json", "w")
json_file.write(model_json)
json_file.close()
print('Модель сохранена.')

model.save_weights("weights1.h5")
print('Веса модели сохранены')