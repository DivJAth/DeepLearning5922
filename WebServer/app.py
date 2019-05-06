from flask import Flask, render_template, request
from flask_cors import CORS
from PIL import Image
import requests
import json
import sys
import os

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
    for file in files:
        file.save(os.path.join(app.config['IMG_UPLOAD_DIR'], file.filename))
    # response = requests.post('http://%s:%s/classification' % (ip, port), data=request.data)
    return app.response_class(
        response='response.text',
        status=200
    )

@app.route('/ml/semantic_segmentation', methods=['POST'])
def vgg16_semantic_segmentation_handler():
    pass