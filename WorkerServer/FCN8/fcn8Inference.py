
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keras
import keras.applications
from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import random

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd

from keras import optimizers
warnings.filterwarnings("ignore")

from keras.applications import vgg16
from keras import optimizers
from sklearn.utils import shuffle 


def getImageArr( path , width , height ):
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
    return img

def predictfcn8(imagefile):
	print(imagefile)
	model=load_model('FCN8.h5')
	model.summary()
	sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
	input_width,input_height=224,224

	X_test=[getImageArr(imagefile, input_width , input_height )]
	im = X_test
	crpim = im
	preds = model.predict(crpim)
	imclass = np.argmax(preds, axis=3)

	plt.figure(figsize = (15, 7))
	plt.subplot(1,3,1)
	plt.imshow( np.asarray(crpim) )
	plt.subplot(1,3,2)
	plt.imshow( imclass)
	plt.subplot(1,3,3)
	plt.imshow( np.asarray(crpim) )
	masked_imclass = np.ma.masked_where(imclass == 0, imclass)
	plt.imshow( masked_imclass, alpha=0.5 )
	plt.savefig('fcn8.png')
	return

dir_data = r"/home/ubuntu/WorkerServer/fcn8/dataset1"
dir_img = dir_data + r"/images_prepped_train/"

img=dir_img+r'0016E5_08001.png'
print(img)
predictfcn8(img)

