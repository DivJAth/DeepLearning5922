from flask import Flask, request
import boto3
import json
import numpy as np
import sys

from handwritten_digit_recognition.resnet164 import ResNet164

app = Flask(__name__)

properties = json.load(open('properties.json', 'r'))
s3_bucket_name = properties['s3_bucket_name']

s3 = boto3.resource('s3')
bucket = s3.Bucket(s3_bucket_name)
# bucket.download_file('test_file.txt', './tmp/test_file.txt')

# Example function for classification task
@app.route('/classification', methods=['POST'])
def classification_handler():
    data = json.loads(request.data)
    x_test = np.array(data['x'])
    model = ResNet164()
    model.load_weights('./handwritten_digit_recognition/models/ResNet164.h5')
    predictions = model.predict(x_test)
    return app.response_class(
        response=json.dumps({'predictions' : predictions.tolist()}),
        status=200
    )