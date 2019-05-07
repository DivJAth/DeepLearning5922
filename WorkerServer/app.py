from flask import Flask, request
import boto3
import json
import numpy as np
import sys

from handwritten_digit_recognition.resnet164 import ResNet164
from fashion_mnist.predict import predict as fm_predict

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
    print(x_test)
    if len(x_test.shape) == 3:
        x_test = np.expand_dims(x_test, axis=3)
    model = ResNet164()
    model.load_weights('./handwritten_digit_recognition/models/ResNet164.h5')
    output = model.predict(x_test)
    predictions = np.array([x.tolist().index(max(x)) for x in output])
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
    pass