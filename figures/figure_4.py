"""
Figure 4, Illustration of the significance of the principal components as compared
to random directions in the intermediate latent space W of StyleGAN2.
"""

import sys
sys.path.append('../models/stylegan2')
sys.path.append('..')

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import special_ortho_group

from decomposition import load_network, pca

SAVE_PATH = './results'
os.makedirs(SAVE_PATH, exist_ok=True)


num_samples = 1_000_000
batch_size = 500
num_components = 512
truncation_psi = 0.55
seed = 1866827965
class_name = 'cats'

Gs, Gs_kwargs = load_network(class_name)
Gs_kwargs.truncation_psi = truncation_psi
dump_path = pca(Gs, 2, class_name, batch_size=batch_size, num_components=num_components, num_samples=num_samples)

N = 8
use_w = True
use_random_basis = True

model_name = 'StyleGAN2'

outdir = Path('out/figures/random_baseline')
os.makedirs(outdir, exist_ok=True)

w_avg = Gs.get_var('dlatent_avg')

num_layers = Gs.components.mapping.output_shape[1]

input_shape = Gs.input_shape[1]
K = np.prod(input_shape)

with np.load(dump_path) as data:
    lat_comp = data['lat_comp']
    lat_mean = data['lat_mean']
    lat_std = data['lat_stdev']

B = 6
if seed is None:
    seed = np.random.randint(np.iinfo(np.int32).max - B)

print(f'Seeds: {seed} - {seed+B}')

# Resampling test
rnd = np.random.RandomState(seed+B)
w_base = rnd.randn(1, *Gs.input_shape[1:])
if use_w:
    w_base = Gs.components.mapping.run(w_base, None, dlatent_broadcast=None)
w_base_img = w_avg + (w_base - w_avg) * Gs_kwargs.truncation_psi
imgs = Gs.components.synthesis.run(np.array([w_base_img]*num_layers).reshape((1, num_layers, input_shape)), **Gs_kwargs)
plt.imshow(imgs[0])
plt.axis('off')
plt.title('Original', fontsize=20)
plt.savefig(os.path.join(SAVE_PATH, 'figure4_original.png'), bbox_inches='tight')


# Project tensor 'X' onto orthonormal basis 'comp', return coordinates
def project_ortho(X, comp):
    N = comp.shape[0]
    coords = (comp.reshape(N, -1) * X.reshape(-1)).sum(axis=1)
    return coords.reshape([N]+[1]*X.ndim)


# Resample some components
def get_batch(indices, basis):
    w_batch = np.zeros((B, K))
    coord_base = project_ortho(w_base - lat_mean, basis)

    for i in range(B):
        rnd = np.random.RandomState(seed+i)
        w = rnd.randn(1, *Gs.input_shape[1:])
        if use_w:
            w = Gs.components.mapping.run(w, None, dlatent_broadcast=None)
        coords = coord_base.copy()
        coords_resampled = project_ortho(w - lat_mean, basis)
        coords[indices, :, :] = coords_resampled[indices, :, :]
        w_batch[i, :] = lat_mean + np.sum(coords * basis, axis=0)

    return w_batch


def show_grid(w, title):
    w = np.expand_dims(w, axis=1)
    w = np.repeat(w, num_layers, axis=1)
    w = w_avg + (w - w_avg) * Gs_kwargs.truncation_psi
    out = Gs.components.synthesis.run(w, **Gs_kwargs)
    out = out[:, :, 18:-8, :]
    
    grid_np = np.hstack(out)
    plt.axis('off')
    plt.tight_layout()
    plt.title(title, fontsize=15)
    plt.imshow(grid_np, interpolation='bilinear')
    plt.savefig(os.path.join(SAVE_PATH, 'figure4_{}.png'.format(title.lower()[3:].replace(' -> ', '-').replace(' ', '_'))), bbox_inches='tight')


def save_imgs(w, prefix):
    w = np.expand_dims(w, axis=1)
    w = np.repeat(w, num_layers, axis=1)
    w = w_avg + (w - w_avg) * Gs_kwargs.truncation_psi
    imgs = Gs.components.synthesis.run(w, **Gs_kwargs)
    for i, img in enumerate(imgs):
        img = img[18:-8, :, :]
        outpath = outdir / f'{model_name}_{class_name}' / f'{prefix}_{i}.png'
        os.makedirs(outpath.parent, exist_ok=True)
        Image.fromarray(np.uint8(img * 255)).save(outpath)


def orthogonalize_rows(V):
    Q, R = np.linalg.qr(V.T)
    return Q.T


def assert_orthonormal(V):
    M = np.dot(V, V.T)
    det = np.linalg.det(M)
    assert np.allclose(M, np.identity(M.shape[0]), atol=1e-5), f'Basis is not orthonormal (det={det})'


plt.figure(figsize=((12, 6.5) if class_name in ['cars', 'cats'] else (12, 8)))

# First N fixed
ind_rand = np.array(range(N, K))  # N -> K rerandomized
b1 = get_batch(ind_rand, lat_comp)
plt.subplot(2, 2, 1)
show_grid(b1, f'a) Keep {N} first pca -> Consistent pose')
save_imgs(b1, f'keep_{N}_first_{seed}')

# First N randomized
ind_rand = np.array(range(0, N))  # 0 -> N rerandomized
b2 = get_batch(ind_rand, lat_comp)
plt.subplot(2, 2, 2)
show_grid(b2, f'b) Randomize {N} first pca -> Consistent style')
save_imgs(b2, f'randomize_{N}_first_{seed}')

if use_random_basis:
    # Random orthonormal basis drawn from p(w)
    # Highly shaped by W, sort of a noisy pseudo-PCA
    # V = (model.sample_latent(K, seed=seed + B + 1) - lat_mean).cpu().numpy()
    # V = V / np.sqrt(np.sum(V*V, axis=-1, keepdims=True)) # normalize rows
    # V = orthogonalize_rows(V)

    # Isotropic random basis
    V = special_ortho_group.rvs(K)
    assert_orthonormal(V)

    rand_basis = np.reshape(V, lat_comp.shape)
    assert rand_basis.shape == lat_comp.shape, f'Shape mismatch: {rand_basis.shape} != {lat_comp.shape}'

    ind_perm = range(K)
else:
    # Just use shuffled PCA basis
    rng = np.random.RandomState(seed=seed)
    perm = rng.permutation(range(K))
    rand_basis = lat_comp[perm, :]

basis_type_str = 'random' if use_random_basis else 'pca_shfl'

# First N random fixed
ind_rand = np.array(range(N, K))  # N -> K rerandomized
b3 = get_batch(ind_rand, rand_basis)
plt.subplot(2, 2, 3)
show_grid(b3, f'c) Keep {N} first {basis_type_str} -> Little consistency')
save_imgs(b3, f'keep_{N}_first_{basis_type_str}_{seed}')

# First N random rerandomized
ind_rand = np.array(range(0, N))  # 0 -> N rerandomized
b4 = get_batch(ind_rand, rand_basis)
plt.subplot(2, 2, 4)
show_grid(b4, f'd) Randomize {N} first {basis_type_str} -> Little variation')
save_imgs(b4, f'randomize_{N}_first_{basis_type_str}_{seed}')
