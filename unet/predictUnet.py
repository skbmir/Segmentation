from keras.models import model_from_json
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGE_LIB = 'data/membrane/test/'
IMG_HEIGHT, IMG_WIDTH = 256, 256
modelJson = open('model.json', 'r')
loadedModelJson = modelJson.read()
modelJson.close()
loadedModelJson = model_from_json(loadedModelJson)
loadedModelJson.load_weights('weights.h5')
loadedModelJson.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=['accuracy'])
all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.png']
x_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(IMAGE_LIB + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_data[i] = im
x_data = x_data[:, :, :, np.newaxis]
result = loadedModelJson.predict(x_data[0].reshape(1, IMG_HEIGHT, IMG_WIDTH, 1))
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(result[0, :, :, 0], cmap='gray')
plt.show()
