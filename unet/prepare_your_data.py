import numpy as np
import cv2
import os
import argparse
import math

IMG_WIDTH = 128
IMG_HEIGHT = 128

parser = argparse.ArgumentParser()
parser.add_argument('--images', help='Path to images')
parser.add_argument('--labels', help='Path to labels')
args = parser.parse_args()


def load_ans_make(list, is_label=False):
    data = []

    for i, img in enumerate(list):
        if is_label is True:
            print('preparing ' + str(i) + ' label')
            full_size_image = cv2.imread(os.path.join(args.labels, img))
            # 8, 8, 8 for trees, for example
            full_size_image = cv2.inRange(full_size_image, (8, 8, 8), (8, 8, 8))
            full_size_image = cv2.cvtColor(full_size_image, cv2.COLOR_GRAY2BGR)
            full_size_image = np.invert(full_size_image)
        else:
            print('preparing ' + str(i) + ' image')
            full_size_image = cv2.imread(os.path.join(args.images, img))
        height, width, depth = full_size_image.shape
        steps_h = math.floor((height / IMG_HEIGHT))
        minus_h = round((height - steps_h * IMG_HEIGHT) / steps_h)
        steps_w = math.floor((width / IMG_WIDTH))
        minus_w = round((width - steps_w * IMG_WIDTH) / steps_w)

        y = 0
        x = 0

        for i in range(steps_h):
            for j in range(steps_w):
                crop = full_size_image[y:y + IMG_HEIGHT, x:x + IMG_WIDTH]
                x = x + IMG_WIDTH - minus_w
                data.append(crop)
            y = y + IMG_HEIGHT - minus_h
            x = 0
    return data


if __name__ == '__main__':
    images_list = os.listdir(args.images)
    labels_list = os.listdir(args.labels)
    # use custom images range * 50
    x = load_ans_make(images_list[:100])
    y = load_ans_make(labels_list[:100], True)
    np.save('images', x)
    np.save('labels', y)

    # y_data = np.load('images.npy')
    # for img in y_data:
    #     cv2.imshow('window', img)
    #     cv2.waitKey(0)
