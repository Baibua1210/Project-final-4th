import numpy as np
import cv2
import os
path = 'C:/Users/pornk/PycharmProjects/Project/examples/test/y/2.jfif'
img = cv2.imread(path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



cv2.imwrite(str(path) + 'Grey.jpg', gray)
# imwrite( "../../images/Gray_Image.jpg", gray_image );

# img = cv2.imread('1.jpg', 1)
#
# cv2.imwrite(os.path.join(path , 'waka.jpg'), img)
# cv2.imwrite(str(path) + 'waka.jpg', img)
# cv2.imshow('color_image', img)
# cv2.imshow('gray_image', gray)
cv2.waitKey(0)
# cv2.destroyAllWindows()