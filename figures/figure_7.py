"""
Figure 7, A selection of interpretable edits discovered by selective application of
latent edits across the layers of several pretrained GAN models.
"""

import sys
sys.path.append('../models/stylegan2')
sys.path.append('..')

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_configs import configs_fig7 as configs
from decomposition import load_network, pca
from utils import centre_strip_stylegan

SAVE_PATH = './results'
os.makedirs(SAVE_PATH, exist_ok=True)


def apply_pca_fig7(Gs, Gs_kwargs, edit, dump_path):
    out_root = Path('out/directions')
    os.makedirs(out_root, exist_ok=True)
    B = 5

    num_imgs_per_example = 1

    input_shape = Gs.input_shape[1]
    num_layers = Gs.components.mapping.output_shape[1]

    grid = []
    strips = []

    configurations = []
    for c in configs:
        if c[9] == edit:
            configurations = c
            break

    print("Configs:", configurations)

    model_name, layer, mode, latent_space, l_start, l_end, classname, sigma, idx, title, s = configurations
    
    print(f'{model_name}, {layer}, {title}')

    with np.load(dump_path) as data:
        X_comp = data['act_comp']
        X_global_mean = data['act_mean']
        X_stdev = data['act_stdev']
        Z_comp = data['lat_comp']
        Z_global_mean = data['lat_mean']
        Z_stdev = data['lat_stdev']

    feat_shape = X_comp[0].shape
    sample_dims = np.prod(feat_shape)

    # Range is exclusive, in contrast to notation in paper
    edit_start = l_start
    edit_end = num_layers if l_end == -1 else l_end

    print("Seeds:", s)
    rnd = np.random.RandomState(s[0])
    z = rnd.randn(1, *Gs.input_shape[1:])

    batch_frames = centre_strip_stylegan(Gs, Gs_kwargs, z, Z_comp, Z_global_mean, Z_stdev, idx, sigma, 5, edit_start, edit_end)
    strips.append(np.hstack(batch_frames))

    # Show first
    grid = np.vstack(strips)
    plt.figure(figsize=(15, 15))
    plt.imshow(grid, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(SAVE_PATH, 'figure7_{}_{}_{}.png'.format(model_name, classname, title.replace(' ', '-'))), bbox_inches='tight')


num_samples = 1_000_000
batch_size = 20


# StyleGAN1 - wikiart
truncation_psi = 1.0
out_class = 'wikiart'

Gs, Gs_kwargs = load_network(out_class, 1)
Gs_kwargs.truncation_psi = truncation_psi
dump_path = pca(Gs, 1, out_class, batch_size=batch_size, num_samples=num_samples)

edits_stylegan1 = 'Head rotation'
apply_pca_fig7(Gs, Gs_kwargs, edits_stylegan1, dump_path)

edits_stylegan1 = 'Simple strokes'
apply_pca_fig7(Gs, Gs_kwargs, edits_stylegan1, dump_path)


# StyleGAN2 - cars
truncation_psi = 0.7
out_class = 'cars'

Gs, Gs_kwargs = load_network(out_class)
Gs_kwargs.truncation_psi = truncation_psi
dump_path = pca(Gs, 2, out_class, batch_size=batch_size, num_samples=num_samples)

edits_stylegan2 = 'Reflections'
apply_pca_fig7(Gs, Gs_kwargs, edits_stylegan2, dump_path)


# StyleGAN2 - horse
truncation_psi = 0.7
out_class = 'horse'

Gs, Gs_kwargs = load_network(out_class)
Gs_kwargs.truncation_psi = truncation_psi
dump_path = pca(Gs, 2, out_class, batch_size=batch_size, num_samples=num_samples)

edits_stylegan2 = 'Add rider'
apply_pca_fig7(Gs, Gs_kwargs, edits_stylegan2, dump_path)


# StyleGAN2 - cats
truncation_psi = 0.7
out_class = 'cats'

Gs, Gs_kwargs = load_network(out_class)
Gs_kwargs.truncation_psi = truncation_psi
dump_path = pca(Gs, 2, out_class, batch_size=batch_size, num_samples=num_samples)

edits_stylegan2 = 'Fluffiness'
apply_pca_fig7(Gs, Gs_kwargs, edits_stylegan2, dump_path)


# StyleGAN2 - ffhq
truncation_psi = 0.7
out_class = 'ffhq'

Gs, Gs_kwargs = load_network(out_class)
Gs_kwargs.truncation_psi = truncation_psi
dump_path = pca(Gs, 2, out_class, batch_size=batch_size, num_samples=num_samples)

edits_stylegan2 = 'Makeup'
apply_pca_fig7(Gs, Gs_kwargs, edits_stylegan2, dump_path)
