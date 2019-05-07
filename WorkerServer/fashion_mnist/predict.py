import tensorflow as tf
import keras
from keras.models import model_from_json

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
        
    model.load_weights('./fashion_mnist/models/convnet.h5')
    print('Loaded model weights')
    return model

def predict(X_test):
    model = load_model()
    y_hat = model.predict(X_test)
    predictions = [class_names[y.tolist().index(max(y))] for y in y_hat]
    return predictions