from flask import Flask, request
<<<<<<< HEAD
from flask import send_file
=======
>>>>>>> refs/remotes/origin/master
import boto3
import json
import numpy as np
import sys
from flask import send_file

from handwritten_digit_recognition.resnet164 import ResNet164
from fashion_mnist.predict import predict as fm_predict
<<<<<<< HEAD
from mask_rcnn.samples import object_recognition
=======
from Mask_RCNN.samples import object_recognition
>>>>>>> refs/remotes/origin/master

app = Flask(__name__)

properties = json.load(open('properties.json', 'r'))
s3_bucket_name = properties['s3_bucket_name']

s3 = boto3.resource('s3')
bucket = s3.Bucket(s3_bucket_name)
# bucket.download_file('test_file.txt', './tmp/test_file.txt')

@app.route('/classification', methods=['POST'])
def classification_handler():
    data = json.loads(request.data)
    x_test = np.array(data['x'])
<<<<<<< HEAD
    if len(x_test.shape) < 4:
        x_test = np.expand_dims(x_test, axis=3)
=======
>>>>>>> refs/remotes/origin/master
    model = ResNet164()
    model.load_weights('./handwritten_digit_recognition/models/ResNet164.h5')
    predictions = model.predict(x_test)
    return app.response_class(
        response=json.dumps({'predictions' : predictions.tolist()}),
        status=200
    )

@app.route('/fm-classification', methods=['POST'])
def fashion_mnist_classification_handler():
    data = json.loads(request.data)
    X_test = np.array(data['x'])
    predictions = fm_predict(X_test)
    return app.response_class(
        response=json.dumps({'predictions' : predictions}),
        status=200
    )

@app.route('/objectdetection', methods = ['POST'])
def objectdetection_handler():
<<<<<<< HEAD
    image2 = request.files.get('file', '')
    image = object_recognition.object_detection(image2)

    bucket.upload_file('./mask_rcnn/samples/foo.png', 'object_detection/foo.png')

    return app.response_class(
        response='https://s3.us-east-2.amazonaws.com/dl-portal-bucket/object_detection/foo.png',
=======
    image2=request.files.get('image', '')
    print(type(image2))
    image = object_recognition.object_detection(image2)

    #if image
    return send_file("foo.png", mimetype='image/png')

    return app.response_class(
        response=json.dumps({'predictions' : 'lol'}),
>>>>>>> refs/remotes/origin/master
        status=200
    )
