import numpy as np
import argparse
import h5py as hp
from keras.layers import Conv2D, concatenate, MaxPool2D, UpSampling2D, Dropout, Input, MaxPooling2D, BatchNormalization
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K

IMG_HEIGHT, IMG_WIDTH = 128, 128
SEED = 42
smooth = 1e-12

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path to dataset')
parser.add_argument('--name', help='Dataset name')
args = parser.parse_args()


def make_unet():
    input_layer = Input(shape=(128, 128, 3))

    conv0 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv0 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(pool0)
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(conv9)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv0], axis=3)
    conv10 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(conv10)

    conv11 = Conv2D(filters=2, kernel_size=(3, 3), activation='relu', padding='same')(conv10)
    conv11 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv11)

    model = Model(input=input_layer, output=conv11)

    return model


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
        width_shift_range=0.15,
        height_shift_range=0.15,
        rotation_range=30,
        zoom_range=0.15,
        rescale=1. / 255).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
        width_shift_range=0.15,
        height_shift_range=0.15,
        rotation_range=30,
        zoom_range=0.15,
        rescale=1. / 255).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


if __name__ == '__main__':
    ds = hp.File(args.path, 'r')
    x_data = ds['images'][:]
    y_data = ds['labels'][:]
    # x_test = np.load('testimages.npy')
    # y_test = np.load('testlabels.npy')
    print('found ' + str(len(x_data)) + ' images..')
    print('found ' + str(len(y_data)) + ' labels..')
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.25)
    x_val = x_val / 255.
    y_val = y_val / 255.
    model = make_unet()
    print('make unet..')
    model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=['accuracy', jaccard_coef])
    model.summary()
    hist = model.fit_generator(my_generator(x_train, y_train, 2),
                               steps_per_epoch=500,
                               validation_data=(x_val, y_val),
                               epochs=15, verbose=1)
    # if need test
    # x_test = x_test / 255.
    # y_test = y_test / 255.
    # scores = model.evaluate(x_test, y_test, verbose=0)
    # print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))

    model_json = model.to_json()
    json_file = open(args.name + ".json", "w")
    json_file.write(model_json)
    json_file.close()
    print('Модель сохранена.')

    model.save_weights(args.name + ".h5")
    print('Веса модели сохранены')
