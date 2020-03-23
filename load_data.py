import os

import numpy as np
from skimage.io import imread
from skimage.transform import resize

TRAIN_DIR = 'data/stage1_train'
TEST_DIR = ['data/stage1_test', 'data/stage2_test_final']
IMAGES_DIR_NAME = 'images'
MASKS_DIR_NAME = 'masks'


def load_data(image_height, image_width, image_channels):
    data_length = len(next(os.walk(TRAIN_DIR))[1])
    X = np.zeros((data_length, image_height, image_width, image_channels), dtype=np.uint8)
    y = np.zeros((data_length, image_height, image_width, 1), dtype=np.bool)

    log_per = data_length // 20

    for i, dir_name in enumerate(next(os.walk(TRAIN_DIR))[1]):

        img_dir = os.path.join(TRAIN_DIR, dir_name, IMAGES_DIR_NAME)
        mask_dir = os.path.join(TRAIN_DIR, dir_name, MASKS_DIR_NAME)

        img_path = os.path.join(img_dir, next(os.walk(img_dir))[2][0])

        image = imread(img_path)[:, :, :image_channels]
        image = resize(image, (image_height, image_width), mode='constant', preserve_range=True)

        X[i] = image

        mask = np.zeros((image_height, image_width, 1), dtype=np.bool)
        for mask_file in next(os.walk(mask_dir))[2]:
            mask_tmp = imread(os.path.join(mask_dir, mask_file))
            mask_tmp = np.expand_dims(resize(mask_tmp, (image_height, image_width), mode='constant',
                                             preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_tmp)
        y[i] = mask

        if not i % log_per:
            print(f"Data loading: {i}/{data_length}")

    return X, y
