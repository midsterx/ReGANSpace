"""
Figure 1, Sequences of image edits performed using control discovered with our method, applied to
three different GANs. 
"""

import sys
sys.path.append('../models/stylegan2')
sys.path.append('..')

import os

import matplotlib.pyplot as plt
import numpy as np

from decomposition import load_network, pca
from figure_configs import configs_fig1 as configs

SAVE_PATH = './results'
os.makedirs(SAVE_PATH, exist_ok=True)


def apply_pca_fig1(Gs, edits, seed, dump_path):
    with np.load(dump_path) as data:
        lat_comp = data['lat_comp']
        lat_mean = data['lat_mean']
        lat_std = data['lat_stdev']

    input_shape = Gs.input_shape[1]
    num_layers = Gs.components.mapping.output_shape[1]
    
    rnd = np.random.RandomState(seed)
    z = rnd.standard_normal(input_shape * 1).reshape(1, input_shape)
    w = Gs.components.mapping.run(z, None)
    w = w.reshape((num_layers, 1, input_shape))

    pca_applied_ws = []
    pca_applied_ws.append(w.copy())
    for edit in edits:
        (idx, start, end, strength, invert) = configs[edit]

        # Find out coordinate of w along PC
        w_centered = w[0] - lat_mean
        w_coord = np.sum(w_centered.reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]

        # Invert property if desired (e.g. flip rotation)
        # Otherwise reinforce existing
        if invert:
            sign = w_coord / np.abs(w_coord)
            target = -sign*strength  # opposite side of mean
        else:
            target = strength

        delta = target - w_coord  # offset vector

        for l in range(start, end):
            w[l] = w[l] + lat_comp[idx]*lat_std[idx]*delta
        pca_applied_ws.append(w.copy())

    for i in range(len(pca_applied_ws)):
        pca_applied_ws[i] = pca_applied_ws[i].reshape((1, num_layers, input_shape))

    return pca_applied_ws


num_samples = 1_000_000
batch_size = 200


# StyleGAN2 - cars
# Figure 1, row 1
truncation_psi = 0.6
seed = 440749230
out_class = 'cars'

Gs, Gs_kwargs = load_network(out_class)
dump_path = pca(Gs, 2, out_class, batch_size=batch_size, num_samples=num_samples)

edits = ['Redness', 'Add grass', 'Horizontal flip', 'Blocky shape']
ws = apply_pca_fig1(Gs, edits, seed, dump_path)

w_avg = Gs.get_var('dlatent_avg')

imgs = []
for idx, w in enumerate(ws):
    w = w_avg + (w - w_avg) * truncation_psi
    images = Gs.components.synthesis.run(w, **Gs_kwargs)
    imgs.append(images[0])

crop = [64, 64, 1, 1]
imgs = [img[crop[0]:-crop[1], crop[2]:-crop[3], :] for img in imgs]
strip = np.hstack(imgs)
plt.figure(figsize=(30, 5))
plt.imshow(strip, interpolation='bilinear')
plt.axis('off')
plt.title(' -> '.join(['Initial'] + edits), fontsize=30)
plt.savefig(os.path.join(SAVE_PATH, "figure1_StyleGAN2_{}.png".format(out_class)), bbox_inches='tight')

# StyleGAN2 - ffhq
# Figure 1, row 2
truncation_psi = 0.7
seed = 6293435
out_class = 'ffhq'

Gs, Gs_kwargs = load_network(out_class)
dump_path = pca(Gs, 2, out_class, batch_size=batch_size, num_samples=num_samples)

edits = ['wrinkles', 'white_hair', 'in_awe', 'overexposed']
ws = apply_pca_fig1(Gs, edits, seed, dump_path)

w_avg = Gs.get_var('dlatent_avg')

imgs = []
for idx, w in enumerate(ws):
    w = w_avg + (w - w_avg) * truncation_psi
    images = Gs.components.synthesis.run(w, **Gs_kwargs)
    imgs.append(images[0])

strip = np.hstack(imgs)
plt.figure(figsize=(30, 5))
plt.imshow(strip, interpolation='bilinear')
plt.axis('off')
plt.title(' -> '.join(['Initial'] + edits), fontsize=30)
plt.savefig(os.path.join(SAVE_PATH, "figure1_StyleGAN2_{}.png".format(out_class)), bbox_inches='tight')
