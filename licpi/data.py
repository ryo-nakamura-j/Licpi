import numpy as np
import matplotlib.pyplot as plt
import struct
import gzip

def parse_mnist(file_read):
    with gzip.open(file_read, "rb") as f:
        magic_number, num_images, rows, columns = struct.unpack('>4i', f.read(16))
        image_data = f.read(rows * columns * num_images)
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, rows, columns)
        return images
    
def visualize_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()
