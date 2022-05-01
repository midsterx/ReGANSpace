"""
Figure 3, Effect of largest principal components and constraining variations to a subset of layers
"""

import sys
sys.path.append('../models/stylegan2')
sys.path.append('..')

import os

import matplotlib.pyplot as plt
import numpy as np

from decomposition import load_network, pca
from utils import centre_strip_stylegan

SAVE_PATH = './results'
os.makedirs(SAVE_PATH, exist_ok=True)

num_samples = 1_000_000
batch_size = 20
truncation_psi = 1.0
seed = 366745668
out_class = 'ffhq'

Gs, Gs_kwargs = load_network(out_class)
Gs_kwargs.truncation_psi = truncation_psi
dump_path = pca(Gs, 2, out_class, batch_size=batch_size, num_samples=num_samples)

with np.load(dump_path) as data:
    lat_comp = data['lat_comp']
    lat_mean = data['lat_mean']
    lat_std = data['lat_stdev']

input_shape = Gs.input_shape[1]
num_layers = Gs.components.mapping.output_shape[1]

rnd = np.random.RandomState(seed)
z = rnd.randn(1, *Gs.input_shape[1:])


# Figure 3, normal centered PCs
n_pcs = 14

strips = []

for i in range(n_pcs):
    batch_frames = centre_strip_stylegan(Gs, Gs_kwargs, z, lat_comp, lat_mean, lat_std, i, 2.0, 7, 0, 18)
    strips.append(np.hstack(batch_frames))

grid = np.vstack(strips)

plt.figure(figsize=(20, 40))
plt.imshow(grid, interpolation='bilinear')
plt.axis('off')
plt.savefig(os.path.join(SAVE_PATH, 'figure3_normal_centered_pcs.png'), bbox_inches='tight')


# Figure 3, hand-tuned layer ranges for some directions
hand_tuned = [
 ( 0, (1,  7), 2.0),  # gender, keep age
 ( 1, (0,  3), 2.0),  # rotate, keep gender
 ( 2, (3,  8), 2.0),  # gender, keep geometry
 ( 3, (2,  8), 2.0),  # age, keep lighting, no hat
 ( 4, (5, 18), 2.0),  # background, keep geometry
 ( 5, (0,  4), 2.0),  # hat, keep lighting and age
 ( 6, (7, 18), 2.0),  # just lighting
 ( 7, (5,  9), 2.0),  # just lighting
 ( 8, (1,  7), 2.0),  # age, keep lighting
 ( 9, (0,  5), 2.0),  # keep lighting
 (10, (7,  9), 2.0),  # hair color, keep geom
 (11, (0,  5), 2.0),  # hair length, keep color
 (12, (8,  9), 2.0),  # light dir lr
 (13, (0,  6), 2.0),  # about the same
]   

strips = []

for i, (s, e), sigma in hand_tuned:
    batch_frames = centre_strip_stylegan(Gs, Gs_kwargs, z, lat_comp, lat_mean, lat_std, i, sigma, 7, s, e)
    strips.append(np.hstack(batch_frames))

grid = np.vstack(strips)

plt.figure(figsize=(20, 40))
plt.imshow(grid, interpolation='bilinear')
plt.axis('off')
plt.savefig(os.path.join(SAVE_PATH, 'figure3_hand_tuned.png'), bbox_inches='tight')
