""" Small piece of code to see the pictures of the training dataset"""

import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser("Display image of dataset")
parser.add_argument('image_number', type=int)
parser.add_argument('-f', '--file', type=str, default='datasets/training_2.h5')
parser.add_argument('-n', '--no_info', action='store_false')
args = parser.parse_args()

filename = args.file
f = h5py.File(filename, 'r')

obj = list(f.keys())[0]
# Get the data
data = f[obj]
images = data['img']


if args.no_info:
    print('\n [·] Data about the dataset selected: \n\
                * Number images    : {}\n\
                * Image dimensions : {}'.format(len(images), np.shape(images[args.image_number])))

plt.imshow(np.moveaxis(images[args.image_number], 0, -1))
if 'kp_2D' in data:
    labels = data['kp_2D']
    plt.scatter(labels[args.image_number, :, 0], labels[args.image_number, :, 1])
else:
    print('(Dataset without joint points.)')
plt.show()
