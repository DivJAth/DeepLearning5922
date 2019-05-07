import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys
import os
import re

#print(sys.path)

ROOT_DIR = os.path.abspath("../")

#sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))
#print('+'*100)
#print(ROOT_DIR)
print(os.getcwd())
#print('+'*100)


<<<<<<< HEAD
from mask_rcnn.samples.mrcnn import utils
from mask_rcnn.samples.mrcnn import model as modellib
from mask_rcnn.samples.mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))

from mask_rcnn.samples.coco import coco
=======
from Mask_RCNN.samples.mrcnn import utils
from Mask_RCNN.samples.mrcnn import model as modellib
from Mask_RCNN.samples.mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))

from Mask_RCNN.samples.coco import coco
>>>>>>> refs/remotes/origin/master


from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from django.http import HttpResponse


MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH1 = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = "mask_rcnn_coco.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = "Mask_RCNN/images"
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
# Create model object in inference mode.


def object_detection(image2):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # print(os.getcwd())
    # print(COCO_MODEL_PATH1)
    model.load_weights(COCO_MODEL_PATH, by_name=True)


    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']


    file_names1 = next(os.walk(IMAGE_DIR))[2]

    #print(file_names)
    r = re.compile("^.*\.jpg")
    file_names = list(filter(r.match, file_names1))
    # file_names = filter(lambda s: re.match("*.png", s), file_names1)
    print(file_names)

    print( os.path.join(IMAGE_DIR, random.choice(file_names)))
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    #print(type(image))
    #np.savetxt("input.txt",image)
    #print(type(image))
    #print(image)
    # Run detection
    image = skimage.io.imread(image2)
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.return_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    #print(image.shape)
    #return image


#         # .............
#
#         canvas = FigureCanvasAgg(fig)
#         response = HttpResponse(content_type = 'image/png')
#         canvas.print_png(response)
#         return response
#
#


if __name__ == '__main__':
    print(object_detection())
