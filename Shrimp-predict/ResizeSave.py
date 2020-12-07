from PIL import Image
from PIL import ImageFile
import os, sys

path = "C:/Users/pornk/PycharmProjects/Project/NEW/train/n/"
save = "C:/Users/pornk/PycharmProjects/Project/New/train/n360/"
dirs = os.listdir(path)

def resize():

    for item in dirs:
        if os.path.isfile(path+item):


            im = Image.open(path+item)
            f, e = os.path.splitext(save+item)
            imResize = im.resize((360, 360), Image.ANTIALIAS)
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            imResize.save(f + '.jpg', 'JPEG', quality=90)

resize()