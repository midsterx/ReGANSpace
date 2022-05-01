"""
Figure 5, comparison of edit directions found through PCA to those found in previous work
using supervised methods.
"""

import sys
sys.path.append('../models/stylegan2')
sys.path.append('..')

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decomposition import load_network, pca
from utils import pad_frames

SAVE_PATH = './results'
os.makedirs(SAVE_PATH, exist_ok=True)

SEED_SAMPLING = 1


def apply_pca_fig5(Gs, Gs_kwargs, use_w, lat_mean, prefix, imgclass, seeds, d_ours, l_start, l_end, scale_ours, d_sup, scale_sup, center=True):
    out_root = Path('out/figures/steerability_comp')
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(out_root / imgclass, exist_ok=True)

    normalize = lambda t: t / np.sqrt(np.sum(t.reshape(-1)**2))

    w_avg = Gs.get_var('dlatent_avg')

    input_shape = Gs.input_shape[1]
    num_layers = Gs.components.mapping.output_shape[1]

    for seed in seeds:
        print("Seed:", seed)
        deltas = [d_ours, d_sup]
        scales = [scale_ours, scale_sup]
        
        ranges = [(l_start, l_end), (0, num_layers)]
        names = ['GANSpace', 'Supervised']

        for delta, name, scale, l_range in zip(deltas, names, scales, ranges):
            np.random.seed(seed or SEED_SAMPLING)
            lat_base = np.random.randn(1, *Gs.input_shape[1:])
            if use_w:
                w = Gs.components.mapping.run(lat_base, None, dlatent_broadcast=None)
                lat_base = w

            # Shift latent to lie on mean along given direction
            if center:
                y = normalize(d_sup)  # assume ground truth
                dotp = np.sum((lat_base - lat_mean) * y, axis=-1, keepdims=True)
                lat_base = lat_base - dotp * y

            # Convert single delta to per-layer delta (to support Steerability StyleGAN)
            if delta.shape[0] > 1:
                *d_per_layer, = delta  # might have per-layer scales, don't normalize
            else:
                d_per_layer = [normalize(delta)]*num_layers

            frames = []
            n_frames = 5
            for a in np.linspace(-1.0, 1.0, n_frames):

                w = [lat_base]*num_layers
                for l in range(l_range[0], l_range[1]):
                    w[l] = w[l] + a*d_per_layer[l]*scale

                w = np.array(w)
                w = w_avg + (w - w_avg) * Gs_kwargs.truncation_psi
                imgs = Gs.components.synthesis.run(w.reshape((1, num_layers, input_shape)), **Gs_kwargs)
                frames.append(imgs[0])


            strip = np.hstack(pad_frames(frames, 64))
            plt.figure(figsize=(12, 12))
            plt.imshow(strip)
            plt.axis('off')
            plt.tight_layout()
            plt.title(f'{prefix} - {name}', fontsize=20)
            plt.savefig(os.path.join(SAVE_PATH, f'figure5_{prefix}-{name}_scale={scale}.png'), bbox_inches='tight')


num_samples = 1_000_000
batch_size = 20


# StyleGAN1 - ffhq (InterfaceGAN) - Figure 5: c, d
truncation_psi = 1.0
out_class = 'ffhq'

Gs, Gs_kwargs = load_network(out_class, model=1)
Gs_kwargs.truncation_psi = truncation_psi
dump_path = pca(Gs, 2, out_class, batch_size=batch_size, num_components=128, num_samples=num_samples)

with np.load(dump_path) as data:
    lat_comp = data['lat_comp']
    lat_mean = data['lat_mean']

d_ffhq_pose = np.load('./data/interfacegan/stylegan_ffhq_pose_w_boundary.npy').astype(np.float32)
d_ffhq_smile = np.load('./data/interfacegan/stylegan_ffhq_smile_w_boundary.npy').astype(np.float32)

# Indices determined by visual inspection
d_ours_pose = lat_comp[9]
d_ours_smile = lat_comp[44]

apply_pca_fig5(Gs, Gs_kwargs, True, lat_mean, 'Pose', 'ffhq', [129888612], d_ours_pose, 0, 7, -1.0, d_ffhq_pose, 1.0)
apply_pca_fig5(Gs, Gs_kwargs, True, lat_mean, 'Smile', 'ffhq', [70163682], d_ours_smile, 3, 4, -8.5, d_ffhq_smile, 1.0)

# StyleGAN1 - celebahq (InterfaceGAN) - Figure 5: e, f
truncation_psi = 1.0
out_class = 'celebahq'

Gs, Gs_kwargs = load_network(out_class, model=1)
Gs_kwargs.truncation_psi = truncation_psi
dump_path = pca(Gs, 2, out_class, batch_size=batch_size, num_components=128, num_samples=num_samples)

with np.load(str(dump_path)) as data:
    lat_comp = data['lat_comp']
    lat_mean = data['lat_mean']

# SG-ffhq-w, non-conditional
d_celebahq_gender = np.load('./data/interfacegan/stylegan_celebahq_gender_w_boundary.npy').astype(np.float32)
d_celebahq_glasses = np.load('./data/interfacegan/stylegan_celebahq_eyeglasses_w_boundary.npy').astype(np.float32)

# Indices determined by visual inspection
d_ours_gender = lat_comp[1]
d_ours_glasses = lat_comp[5]

apply_pca_fig5(Gs, Gs_kwargs, True, lat_mean, 'Gender', 'celebahq', [264878205], d_ours_gender, 0, 2, -3.2, d_celebahq_gender, 1.2)
apply_pca_fig5(Gs, Gs_kwargs, True, lat_mean, 'Glasses', 'celebahq', [1919124025], d_ours_glasses, 0, 1, -10.0, d_celebahq_glasses, 2.0)
