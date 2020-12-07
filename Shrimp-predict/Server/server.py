
from flask import Flask
from flask_cors import cross_origin, CORS
from flask_restful import Resource, Api, reqparse
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
parser = reqparse.RequestParser()
import urllib.request
import os
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = load_model('test_final.h5')

# Downloaded image
def downloader_img():
    URL = "https://firebasestorage.googleapis.com/v0/b/shimp-detection.appspot.com/o/Shimp%2F6.jpg?alt=media&token=5b8917a0-df71-495d-aeab-576a9e7c658e"
    path = os.path.join('./image/Shrimp.jpg')
    urllib.request.urlretrieve(URL,path) #คำสั่งโหลดไฟล์จากเน็ต
    print('Downloaded:' + path)
    return path
def predict():
    print("Start")
    img = cv2.imread('./image/Shrimp.jpg')
    img = cv2.resize(img, (224, 224))
    #
    img = np.reshape(img, [1, 224, 224, 3])
    #
    classes = model.predict(img)
    #
    index_max = np.argmax(classes)
    if (index_max == 0):
        result = "normal"

    elif (index_max == 1):
        result = "taura"

    elif (index_max == 2):
        result = "virus"

    elif (index_max == 3):
        result = "whitespot"
    else:
        result = "yellow"

    print(result)
    return result

def preprocess():
    print("color")
    path = os.path.join('./image/Shrimp.jpg')
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path,grey)
    # img.save('./image/Shrimp.jpg',grey)


# SERVER
app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'flask'}

class Downloaded(Resource):
    def get(self):
        downloader_img()
        preprocess()
        result = predict()
        return {'result': result}


api.add_resource(HelloWorld, '/')
api.add_resource(Downloaded, '/predict')

if __name__ == '__main__':
    CORS(app)
    app.run(debug=False, host='0.0.0.0',threaded=False)