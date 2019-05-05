from flask import Flask, render_template, request
from flask_cors import CORS
import requests
import json
import sys

properties = json.load(open('properties.json', 'r'))
ip = properties['ml_server']['ip']
port = properties['ml_server']['port']

app = Flask(__name__)
CORS(app)
app.jinja_env.auto_reload = True

@app.route('/')
def get_index_page():
    return render_template('index.html', title='Deep Learning Portal')

# Example routing function for ML classification call
@app.route('/ml/classification', methods=['POST'])
def classification_req_handler():
    response = requests.post('http://%s:%s/classification' % (ip, port), data=request.data)
    return app.response_class(
        response=response.text,
        status=200
    )