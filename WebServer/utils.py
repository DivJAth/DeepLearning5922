import os
import shutil
from PIL import Image
import numpy as np

def save_imgs_to_path(files, folderpath):
    saved_files = []
    for file in files:
        path = os.path.join(folderpath, file.filename)
        file.save(path)
        saved_files.append(path)
    return saved_files

def clear_imgs_from_path(folderpath):
    shutil.rmtree(folderpath)
    os.mkdir(folderpath)

def load_imgs_from_path(imgpaths):
    data = []
    for imgpath in imgpaths:
        data.append(np.array(Image.open(imgpath)))
    data = np.array(data)
    return data