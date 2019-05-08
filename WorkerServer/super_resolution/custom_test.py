import numpy as np
from PIL import Image
from ISR.models import RDN

# Initialize the correct model and load pre-trained weights
rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('./super_resolution/weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
print("Successfully loaded pre-trained weights")

def predict(X):
    sr_img = rdn.predict(X)
    return sr_img
