import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.backend.tensorflow_backend import set_session
from keras.applications import vgg16
from keras.applications import inception_v3
from keras.applications import resnet50
from keras.applications import mobilenet


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
#config.gpu_options.visible_device_list = "2" 
set_session(tf.Session(config=config)) 

#Load the VGG16 model
vgg_model = vgg16.VGG16(weights='imagenet') 

#Load the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet') 

#Load the ResNet50 model
resnet_model = resnet50.ResNet50(weights='imagenet') 

#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet') 

image_path = '/home/ubuntu/WorkerServer/imagenetpretrained'
filename = "CHAS1066.jpg"

# load an image in PIL format
original_image = load_img(image_path+filename, target_size=(224, 224)) 
# convert the PIL image (width, height) to a NumPy array (height, width, channel)
numpy_image = img_to_array(original_image) 
# Convert the image into 4D Tensor (samples, height, width, channels) by adding an extra dimension to the axis 0.
input_image = np.expand_dims(numpy_image, axis=0) 
image1 = input_image.reshape((input_image.shape[0], input_image.shape[1], input_image.shape[2],3))
print('PIL image size = ', original_image.size)
print('NumPy image size = ', numpy_image.shape)
print('Input image size = ', input_image.shape)
plt.imshow(np.uint8(input_image[0])) 

	
##Normalization
#preprocess for vgg16
processed_image_vgg16 = vgg16.preprocess_input(input_image.copy()) 
#preprocess for inception_v3
processed_image_inception_v3 = inception_v3.preprocess_input(input_image.copy()) 
#preprocess for resnet50
processed_image_resnet50 = resnet50.preprocess_input(input_image.copy()) 
#preprocess for mobilenet
processed_image_mobilenet = mobilenet.preprocess_input(input_image.copy()) 

# vgg16
predictions_vgg16 = vgg_model.predict(processed_image_vgg16)
label_vgg16 = decode_predictions(predictions_vgg16)
print ('label_vgg16 = ', label_vgg16) 

# inception_v3
predictions_inception_v3 = inception_model.predict(processed_image_inception_v3)
label_inception_v3 = decode_predictions(predictions_inception_v3)
print ('label_inception_v3 = ', label_inception_v3) 

# resnet50
predictions_resnet50 = resnet_model.predict(processed_image_resnet50)
label_resnet50 = decode_predictions(predictions_resnet50)
print ('label_resnet50 = ', label_resnet50) 

# mobilenet
predictions_mobilenet = mobilenet_model.predict(processed_image_mobilenet)
label_mobilenet = decode_predictions(predictions_mobilenet)
print ('label_mobilenet = ', label_mobilenet) 

