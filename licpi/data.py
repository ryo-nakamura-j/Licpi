import numpy as np
import matplotlib.pyplot as plt
import struct
import gzip

def parse_mnist_to_images(file_read):
    with gzip.open(file_read, "rb") as f:
        magic_number, num_images, rows, columns = struct.unpack('>4i', f.read(16))
        image_data = f.read(rows * columns * num_images)
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, rows, columns)
        return images

def parse_mnist(image_filesname, label_filename):
    with gzip.open(image_filesname, "rb") as img_file:
        magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
        assert(magic_num == 2051)
        tot_pixels = row * col
        X = np.vstack([np.array(struct.unpack(f"{tot_pixels}B", img_file.read(tot_pixels)), dtype=np.float32) for _ in range(img_num)])
        X -= np.min(X)
        X /= np.max(X)

    with gzip.open(label_filename, "rb") as label_file:
        magic_num, label_num = struct.unpack(">2i", label_file.read(8))
        assert(magic_num == 2049)
        y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)

    return X, y
    
def visualize_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()
