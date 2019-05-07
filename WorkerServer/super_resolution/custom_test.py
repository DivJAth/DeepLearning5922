import numpy as np
from PIL import Image
from ISR.models import RDN

# Initialize the correct model and load pre-trained weights
rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/sample_weights/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
print("Successfully loaded pre-trained weights")

# Test Image
img = Image.open('data/input/sample/sandal.jpg')
lr_img = np.array(img)

# Run Prediction
sr_img = rdn.predict(lr_img)
print("Got the prediction!")

# Save Image
img = Image.fromarray(sr_img)
img.save("out.jpg", "JPEG", quality=80, optimize=True, progressive=True)
print("Done")
