from keras.models import model_from_json
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGE_LIB = 'test/'
IMG_HEIGHT, IMG_WIDTH = 256, 256
modelJson = open('model.json', 'r')
loadedModelJson = modelJson.read()
modelJson.close()
loadedModelJson = model_from_json(loadedModelJson)
loadedModelJson.load_weights('weights.h5')
loadedModelJson.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=['accuracy'])
all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.png']
for i, name in enumerate(all_images):
    original = cv2.imread(IMAGE_LIB + name, cv2.IMREAD_UNCHANGED)
    # im = original.astype("int16").astype('float32')
    im = original[0:256, 0:256]
    im = im / 255.
    # im = (im - np.min(im)) / (np.max(im) - np.min(im))
    result = loadedModelJson.predict(im.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1))
    cv2.imshow('predict', result[0, :, :, 0])
    cv2.imshow('original', original)
    cv2.waitKey(0)
