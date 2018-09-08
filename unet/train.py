import numpy as np
from keras.layers import Conv2D, concatenate, MaxPool2D, UpSampling2D, Dropout, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

IMG_HEIGHT, IMG_WIDTH = 128, 128
SEED = 42


def make_unet():
    input_layer = Input(shape=(128, 128, 3))
    c1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    l = MaxPool2D(strides=(2, 2))(c1)
    c2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(l)
    l = MaxPool2D(strides=(2, 2))(c2)
    c3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(l)
    l = MaxPool2D(strides=(2, 2))(c3)
    c4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(l)
    l = concatenate([UpSampling2D(size=(2, 2))(c4), c3], axis=-1)
    l = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(l)
    l = concatenate([UpSampling2D(size=(2, 2))(l), c2], axis=-1)
    l = Conv2D(filters=96, kernel_size=(3, 3), activation='relu', padding='same')(l)
    l = concatenate([UpSampling2D(size=(2, 2))(l), c1], axis=-1)
    l = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(l)
    l = Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(l)
    l = Dropout(0.5)(l)
    output_layer = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid')(l)

    model = Model(input_layer, output_layer)

    return model


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1,
        rescale=1. / 255).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1,
        rescale=1. / 255).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


if __name__ == '__main__':
    x_data = np.load('images.npy')
    y_data = np.load('labels.npy')
    x_test = np.load('testimages.npy')
    y_test = np.load('testlabels.npy')
    print('found ' + str(len(x_data)) + ' images..')
    print('found ' + str(len(y_data)) + ' labels..')
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.10)
    x_val = x_val / 255.
    y_val = y_val / 255.
    model = make_unet()
    print('make unet..')
    model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    hist = model.fit_generator(my_generator(x_train, y_train, 4),
                               steps_per_epoch=200,
                               validation_data=(x_val, y_val),
                               epochs=15, verbose=2)
    # if need test
    x_test = x_test / 255.
    y_test = y_test / 255.
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))

    model_json = model.to_json()
    json_file = open("unet_on_bdd.json", "w")
    json_file.write(model_json)
    json_file.close()
    print('Модель сохранена.')

    model.save_weights("unet_on_bdd.h5")
    print('Веса модели сохранены')

