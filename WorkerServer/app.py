from flask import Flask
import boto3
import json

app = Flask(__name__)

properties = json.load(open('properties.json', 'r'))
s3_bucket_name = properties['s3_bucket_name']

s3 = boto3.resource('s3')
bucket = s3.Bucket(s3_bucket_name)



# Example function for classification task
@app.route('/classification', methods=['POST'])
def classification_handler():
    # Extract POST data
    # Download pre-trained model from S3
    bucket.download_file('test_file.txt', './tmp/test_file.txt')
    # Call classification function with model
    return None