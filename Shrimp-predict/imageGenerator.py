import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2

HOME_DIR = "C:/Users/pornk/PycharmProjects/Project/testmo/train/y/"
AFTER_DIR = "C:/Users/pornk/PycharmProjects/Project/testmo/train/y/"

if __name__ == '__main__':
ั
    # result
    result_dir = "{0}".format(AFTER_DIR)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    input_dir = '{0}'.format(HOME_DIR)
    filenames = os.listdir(input_dir)
    datagen = ImageDataGenerator(vertical_flip=True)
    datagen = ImageDataGenerator(horizontal_flip=True)
    datagen = ImageDataGenerator(rotation_range=60)
    datagen = ImageDataGenerator(shear_range=0.25)
    datagen = ImageDataGenerator(zoom_range=0.25)
    datagen = ImageDataGenerator(width_shift_range=0.2)
    datagen = ImageDataGenerator(height_shift_range=0.2)

    # pi/4=0.78
    # sample only disease shear 0.7

    for filename in filenames:
        print (filename)


        img = load_img("{0}/{1}".format(input_dir, filename))


        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)


        gen = datagen.flow(x, batch_size=1, save_to_dir=result_dir, save_prefix='img', save_format='jpg')
        gen.next()