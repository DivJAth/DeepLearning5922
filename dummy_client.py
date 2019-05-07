import requests
import json
import numpy as np
import keras
import sys

HOST_URL, port = sys.argv[1], sys.argv[2]

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
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    url = 'http://%s:%s/fashion-mnist-classification' % (HOST_URL, port)
    payload = { 'x' : test_images.tolist() }
    print('Making POST request to server at %s...' % url)
    r = requests.post(url, data=json.dumps(payload),
        headers={'content-type' : 'application/json'})
    data = json.loads(r.text)
    data = np.array(data['x'])
    print(np.sum(data == test_labels)/len(test_labels))