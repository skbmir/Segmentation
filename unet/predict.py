import cv2
import os
import numpy as np
import argparse
import math
from keras.models import model_from_json
from keras.optimizers import Adam
import matplotlib.pyplot as plt

IMG_HEIGHT, IMG_WIDTH = 128, 128
SEED = 42


def make_data(path):
    data = np.empty((50, IMG_HEIGHT, IMG_WIDTH, 3), dtype='float32')
    image = cv2.imread(path)
    image = image[80:720, 0:1280]
    height, width, depth = image.shape
    steps_h = round((height / IMG_HEIGHT))
    minus_h = round((height - steps_h * IMG_HEIGHT) / steps_h)
    steps_w = math.floor((width / IMG_WIDTH))
    minus_w = round((width - steps_w * IMG_WIDTH) / steps_w)
    y = 0
    x = 0
    cnt = 0

    for i in range(steps_h):
        for j in range(steps_w):
            crop = image[y:y + IMG_HEIGHT, x:x + IMG_WIDTH]
            x = x + IMG_WIDTH - minus_w
            im = np.array(crop) / 255.
            data[cnt] = im
            cnt += 1
        y = y + IMG_HEIGHT - minus_h
        x = 0
    return data


def argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', help='Path for dir with color labels')
    ap.add_argument('--model', help='Path for model JSON')
    ap.add_argument('--weight', help='Path for weights')
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    exargs = argparser()

    im = cv2.imread(exargs.path, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    cv2.imshow('test', im)

    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

    im = (im - np.min(im)) / (np.max(im) - np.min(im))

    original = cv2.imread(exargs.path)
    cv2.imshow('original', original)
    x_data = make_data(exargs.path)
    modelJson = open(exargs.model, 'r')
    loadedModelJson = modelJson.read()
    modelJson.close()
    loadedModelJson = model_from_json(loadedModelJson)
    loadedModelJson.load_weights(exargs.weight)
    loadedModelJson.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=['accuracy'])
    das = np.empty((5, 128, 1280), dtype='uint8')
    cnts = 0
    for i, image in enumerate(x_data):
        result = loadedModelJson.predict(image.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3))
        res = result[0, :, :, 0]
        res = (res - np.min(res)) / (np.max(res) - np.min(res))
        if i == 0 or i == 10 or i == 20 or i == 30 or i == 40 or i == 50:
            old = res
        elif i == 9 or i == 19 or i == 29 or i == 39 or i == 49 or i == 59:
            data = np.hstack((old, res))
            # print(data.shape)
            # data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
            das[cnts] = data
            cv2.imshow(str(i), data)
            cnts += 1
            old = res
        else:
            data = np.hstack((old, res))
            old = data
    cv2.waitKey(0)
