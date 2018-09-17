import numpy as np
import cv2
import os
import argparse
import math
import h5py as hp

IMG_WIDTH = 128
IMG_HEIGHT = 128

parser = argparse.ArgumentParser()
parser.add_argument('--images', help='Path to images')
parser.add_argument('--labels', help='Path to labels')
parser.add_argument('--count', help='Counter of samples')
parser.add_argument('--ds', help='DS name')
args = parser.parse_args()


def load_ans_make(list_of_labels):
    dataset = hp.File(args.ds + '_dataset.hdf5', 'w')

    data_labels = dataset.create_dataset('labels', (int(args.count), 128, 128, 1), dtype='uint8')
    data_images = dataset.create_dataset('images', (int(args.count), 128, 128, 3), dtype='uint8')
    cnt = 0

    for i, img in enumerate(list_of_labels):
        print('preparing ' + str(i) + ' image')
        full_size_label = cv2.imread(os.path.join(args.labels, img))
        full_size_label = full_size_label[80:720, 0:1280]
        # 8, 8, 8 for trees, for example
        full_size_label = cv2.inRange(full_size_label, (142, 0, 0), (142, 0, 0))
        full_size_label = np.invert(full_size_label)
        full_size_label = full_size_label[::np.newaxis]
        full_size_label = full_size_label.reshape(full_size_label.shape[0], full_size_label.shape[1], 1)
        full_size_image = cv2.imread(os.path.join(args.images, img.split('_mask_color')[0] + '.png'))
        # full_size_image = cv2.imread(os.path.join(args.images, img))
        full_size_image = full_size_image[80:720, 0:1280]
        height, width, depth = full_size_image.shape
        steps_h = math.floor((height / IMG_HEIGHT))
        minus_h = round((height - steps_h * IMG_HEIGHT) / steps_h)
        steps_w = math.floor((width / IMG_WIDTH))
        minus_w = round((width - steps_w * IMG_WIDTH) / steps_w)

        y = 0
        x = 0

        for i in range(steps_h):
            for j in range(steps_w):
                crop_label = full_size_label[y:y + IMG_HEIGHT, x:x + IMG_WIDTH]
                crop_image = full_size_image[y:y + IMG_HEIGHT, x:x + IMG_WIDTH]
                x = x + IMG_WIDTH - minus_w
                if np.sum(crop_label == 0) > 0:
                    data_labels[cnt] = crop_label
                    data_images[cnt] = crop_image
                    cnt += 1
                    if cnt == int(args.count):
                        print(str(args.count) + ' images loaded')
                        dataset.close()
                        return
            y = y + IMG_HEIGHT - minus_h
            x = 0


if __name__ == '__main__':
    labels_list = os.listdir(args.labels)
    # use custom images range * 50
    load_ans_make(labels_list)
    f = hp.File(args.ds + '_dataset.hdf5', 'r')
    data_x = f['images'][:]
    data_y = f['labels'][:]
    f.close()
    cv2.imshow('image', data_x[10])
    cv2.imshow('label', data_y[10])
    cv2.waitKey(0)
