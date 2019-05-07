from flask import Flask, render_template, request
from flask import send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import requests
import json
import sys
import shutil
import os

from utils import *

properties = json.load(open('properties.json', 'r'))

ip = properties['ml_server']['ip'] if len(sys.argv) > 2 else properties['local_server']['ip']
port = properties['ml_server']['port'] if len(sys.argv) > 2 else properties['local_server']['port']

app = Flask(__name__)
CORS(app)
app.jinja_env.auto_reload = True
app.config['IMG_UPLOAD_DIR'] = './imgs'

@app.route('/')
def get_index_page():
    return render_template('index.html', title='Deep Learning Portal')

@app.route('/ml/classification', methods=['POST'])
def classification_req_handler():
    files = request.files.to_dict(flat=False)['classification-input-files']
    saved_files = save_imgs_to_path(files, app.config['IMG_UPLOAD_DIR'])
    data = load_imgs_from_path(saved_files)
    response = requests.post('http://%s:%s/classification' % (ip, port), data=json.dumps({'x' : data}))
    clear_imgs_from_path(app.config['IMG_UPLOAD_DIR'])
    return app.response_class(
        response=response.text,
        status=200
    )

@app.route('/ml/fm-classification', methods=['POST'])
def fm_classification_handler():
    files = request.files.to_dict(flat=False)['fm-classification-input-files']
    saved_files = save_imgs_to_path(files, app.config['IMG_UPLOAD_DIR'])
    data = load_imgs_from_path(saved_files)
    response = requests.post('http://%s:%s/fm-classification' % (ip, port), data=json.dumps({'x' : data}))
    clear_imgs_from_path(app.config['IMG_UPLOAD_DIR'])
    return app.response_class(
        response=response.text,
        status=200
    )

@app.route('/ml/objectdetection', methods=['POST'])
def objectdetection_handler():
    file = request.files.to_dict(flat=False)['object-detection-input-files']
    saved_files = save_imgs_to_path(files, app.config['IMG_UPLOAD_DIR'])
    response = request.post('http://%s:%s/objectdetection' % (ip, port), files={'file' : open(saved_files[0], 'r')})
    clear_imgs_from_path(app.config['IMG_UPLOAD_DIR'])
    return app.response_class(
        response=response.text,
        status=200
    )


@app.route('/ml/progressive-gan-generation', methods=['GET'])
def progressive_gan_generation_handler():
    number = request.args.get('number')
    return app.response_class(
        response="https://s3.us-east-2.amazonaws.com/dl-portal-bucket/progressive_gans/img%s.png" % number,
        status=200
    )

@app.route('/ml/semantic_segmentation', methods=['POST'])
def vgg16_semantic_segmentation_handler():
    pass