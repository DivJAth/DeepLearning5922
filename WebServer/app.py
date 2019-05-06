from flask import Flask, render_template, request
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
ip = properties['ml_server']['ip']
port = properties['ml_server']['port']

app = Flask(__name__)
CORS(app)
app.jinja_env.auto_reload = True
app.config['IMG_UPLOAD_DIR'] = './imgs'

@app.route('/')
def get_index_page():
    return render_template('index.html', title='Deep Learning Portal')

# Example routing function for ML classification call
@app.route('/ml/classification', methods=['POST'])
def classification_req_handler():
    files = request.files.to_dict(flat=False)['classification-input-files']
    saved_files = save_imgs_to_path(files, app.config['IMG_UPLOAD_DIR'])
    data = load_imgs_from_path(saved_files)
    print(data)
    # response = requests.post('http://%s:%s/classification' % (ip, port), data=request.data)
    clear_imgs_from_path(app.config['IMG_UPLOAD_DIR'])
    return app.response_class(
        response='response.text',
        status=200
    )

@app.route('/ml/semantic_segmentation', methods=['POST'])
def vgg16_semantic_segmentation_handler():
    pass