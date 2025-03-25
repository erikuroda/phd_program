import numpy as np
import struct
from PIL import Image

def output_mnist_img_png( source_mnist_img_file ):
    f = open("./mnist/t10k-images-idx3-ubyte", 'rb')
    magic_number      = struct.unpack('>i', f.read(4))[0]
    number_of_images  = struct.unpack('>i', f.read(4))[0]
    number_of_rows    = struct.unpack('>i', f.read(4))[0]
    number_of_columns = struct.unpack('>i', f.read(4))[0]
    bytes_per_image = number_of_rows * number_of_columns
    format = '%dB' % bytes_per_image
    for i in range(0, number_of_images):
        raw_img = f.read(bytes_per_image)
        lin_img = struct.unpack(format, raw_img)
        np_ary  = np.asarray(lin_img).astype('uint8')
        np_ary  = np.reshape(np_ary, (28,28),order='C')
        pil_img = Image.fromarray(np_ary)
        fpath   = '%05d.png' % i
        pil_img.save(fpath)
