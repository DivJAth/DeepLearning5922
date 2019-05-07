from flask import Flask, request
import boto3
import json
import numpy as np
import sys

from handwritten_digit_recognition.resnet164 import ResNet164


## OR code start
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import sys
print(sys.path)

ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))

import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


## OR code end


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

@app.route('/objectdetection', methods = ['POST'])
def objectdetection_handler():
    object_detection()
