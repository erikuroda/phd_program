"""Utility functions to read / write images."""
from PIL import Image
import numpy as np

def read_image(filename):
    img = np.asarray(Image.open(filename))
    img = img.astype(np.float32)
    img /= 255
    return img

def write_image(img, filename):
    img *= 255
    img = img.astype(np.uint8)
    image = Image.fromarray(img)
    image.save(filename)

    
