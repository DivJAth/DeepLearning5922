from flask import Flask, render_template, request
import requests
import json

properties = json.load(open('properties.json', 'r'))

ml_server_ip = properties['ml_server_ip']
app = Flask(__name__)


@app.route('/')
def get_index_page():
    return render_template('index.html', title='Deep Learning Portal')

# Example routing function for ML classification call
@app.route('/ml/classification', methods=['POST'])
def classification_req_handler():
    input_sample = request.data
    response_data = requests.post('http://%s/classification' % ml_server_ip, data=input_sample)
    return response_data