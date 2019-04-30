import requests
import json
import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist

HOST_URL = 'ec2-3-16-143-139.us-east-2.compute.amazonaws.com:5000'

def normalize_images(images):
    H, W = 28, 28
    images = np.reshape(images, (-1, H * W))
    numerator = images - np.expand_dims(np.mean(images, 1), 1)
    denominator = np.expand_dims(np.std(images, 1), 1)
    return np.reshape(numerator / (denominator + 1e-7), (-1, H, W))

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = np_utils.to_categorical(y_train) # encode one-hot vector
    y_test = np_utils.to_categorical(y_test)

    num_of_test_data = 50000
    x_val = x_train[num_of_test_data:]
    y_val = y_train[num_of_test_data:]
    x_train = x_train[:num_of_test_data]
    y_train = y_train[:num_of_test_data]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    x_test_batch = x_test[:5, :, :, :]

    url = 'http://%s/ml/classification' % HOST_URL
    payload = { 'x' : x_test_batch.tolist() }
    print('Making POST request to server at %s...' % url)
    r = requests.post(url, data=json.dumps(payload),
        headers={'content-type' : 'application/json'})
    print('Done')
    print(r.text)