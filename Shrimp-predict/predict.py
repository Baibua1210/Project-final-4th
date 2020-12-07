from keras.models import load_model
import cv2
import numpy as np

model = load_model('test_final.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

img = cv2.imread('C:/Users/pornk/PycharmProjects/Project/predict/test/11.jpg')
img = cv2.resize(img,(224,224))
img = np.reshape(img,[1,224,224,3])

classes = model.predict(img)
print (classes)
#
# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
#
# Test_Dir = 'D:/TestData/test/'
#
# New_Model = tf.keras.models.load_model('VGG16_final.h5')
# New_Model.summary()
#
# Image_Path = os.path.join(Test_Dir, '1General1102resized.jpg')
# Img = image.load_img(Image_Path, target_size = (150,150))
# Img_Array = image.img_to_array(Img)
# Img_Array = Img_Array/255.0
# Img_Array = tf.reshape(Img_Array, (-1,150,150,3))
#
# Predictions = New_Model.predict(Img_Array)
#
# Label = tf.argmax(Predictions)
#
# Label.numpy()[0]