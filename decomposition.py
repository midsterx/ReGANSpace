# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

# Patch for broken CTRL+C handler
# https://github.com/ContinuumIO/anaconda-issues/issues/905

import copy
import datetime
import os
import sys
from pathlib import Path

sys.path.append('./models/stylegan2')

import dnnlib
import dnnlib.tflib as tflib
import matplotlib.pyplot as plt
import pretrained_networks
from PIL import Image
from scipy.stats import special_ortho_group
from tqdm import trange

from estimators import get_estimator
from utils import *

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

SEED_SAMPLING = 1
DEFAULT_BATCH_SIZE = 20
SEED_RANDOM_DIRS = 2

B = 20


def get_random_dirs(components, dimensions):
    gen = np.random.RandomState(seed=SEED_RANDOM_DIRS)
    dirs = gen.normal(size=(components, dimensions))
    dirs /= np.sqrt(np.sum(dirs**2, axis=1, keepdims=True))
    return dirs.astype(np.float32)


def load_network(out_class, model=2):
    network = out_classes[model][out_class]
    _G, _D, Gs = pretrained_networks.load_networks(network)

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    rnd = np.random.RandomState(0)
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]

    return Gs, Gs_kwargs


def pca(Gs, stylegan_version, out_class, estimator='ipca', batch_size=20, num_components=80, num_samples=1_000_000, use_w=True, force_recompute=False, seed_compute=None):
    dump_name = "{}-{}_{}_c{}_n{}{}{}.npz".format(
        f'stylegan{stylegan_version}',
        out_class.replace(' ', '_'),
        estimator.lower(),
        num_components,
        num_samples,
        '_w' if use_w else '',
        f'_seed{seed_compute}' if seed_compute else ''
    )
    dump_path = Path(f'./cache/components/{dump_name}')
    if not dump_path.is_file() or force_recompute:
        os.makedirs(dump_path.parent, exist_ok=True)
        compute_pca(Gs, estimator, batch_size, num_components, num_samples, use_w, seed_compute, dump_path)

    return dump_path


def compute_pca(Gs, estimator, batch_size, num_components, num_samples, use_w, seed, dump_path):
    global B

    timestamp = lambda : datetime.datetime.now().strftime("%d.%m %H:%M")
    print(f'[{timestamp()}] Computing', dump_path.name)

    # Ensure reproducibility
    np.random.seed(0)

    # Regress back to w space
    if use_w:
        print('Using W latent space')

    sample_shape = Gs.components.mapping.run(np.random.randn(1, *Gs.input_shape[1:]), None, dlatent_broadcast=None).shape
    sample_dims = np.prod(sample_shape)
    print("Feature shape: ", sample_shape)
    print("Feature dims: ", sample_dims)

    input_shape = (1, *Gs.input_shape[1:])
    input_dims = np.prod(input_shape)

    components = min(num_components, sample_dims)
    transformer = get_estimator(estimator, components, 1.0)

    X = None
    X_global_mean = None

    # Figure out batch size if not provided
    B = batch_size or DEFAULT_BATCH_SIZE

    # Divisible by B (ignored in output name)
    N = num_samples // B * B

    w_avg = Gs.get_var('dlatent_avg')

    # Compute maximum batch size based on RAM + pagefile budget
    target_bytes = 20 * 1_000_000_000 # GB
    feat_size_bytes = sample_dims * np.dtype('float64').itemsize
    N_limit_RAM = np.floor_divide(target_bytes, feat_size_bytes)
    if not transformer.batch_support and N > N_limit_RAM:
        print('WARNING: estimator does not support batching, ' \
              'given config will use {:.1f} GB memory.'.format(feat_size_bytes / 1_000_000_000 * N))

    print('B={}, N={}, dims={}, N/dims={:.1f}'.format(B, N, sample_dims, N/sample_dims), flush=True)

    # Must not depend on chosen batch size (reproducibility)
    NB = max(B, max(2_000, 3*components))  # ipca: as large as possible!

    samples = None
    if not transformer.batch_support:
        samples = np.zeros((N + NB, sample_dims), dtype=np.float32)

    np.random.seed(seed or SEED_SAMPLING)

    # Use exactly the same latents regardless of batch size
    # Store in main memory, since N might be huge (1M+)
    # Run in batches, since sample_latent() might perform Z -> W mapping
    n_lat = ((N + NB - 1) // B + 1) * B
    latents = np.zeros((n_lat, *input_shape[1:]), dtype=np.float32)
    for i in trange(n_lat // B, desc='Sampling latents'):
        seed_global = np.random.randint(np.iinfo(np.int32).max) # use (reproducible) global rand state
        rng = np.random.RandomState(seed_global)
        # z = np.random.randn(B, *input_shape[1:])
        z = rng.standard_normal(512 * B).reshape(B, 512)
        if use_w:
            w = Gs.components.mapping.run(z, None, dlatent_broadcast=None)
            latents[i*B:(i+1)*B] = w
        else:
            latents[i*B:(i+1)*B] = z

    # Decomposition on non-Gaussian latent space
    samples_are_latents = use_w

    canceled = False
    try:
        X = np.ones((NB, sample_dims), dtype=np.float32)
        action = 'Fitting' if transformer.batch_support else 'Collecting'
        for gi in trange(0, N, NB, desc=f'{action} batches (NB={NB})', ascii=True):
            for mb in range(0, NB, B):
                z = latents[gi+mb:gi+mb+B]

                batch = z.reshape((B, -1))

                space_left = min(B, NB - mb)
                X[mb:mb+space_left] = batch[:space_left]

            if transformer.batch_support:
                if not transformer.fit_partial(X.reshape(-1, sample_dims)):
                    break
            else:
                samples[gi:gi+NB, :] = X.copy()
    except KeyboardInterrupt:
        if not transformer.batch_support:
            sys.exit(1)  # no progress yet

        dump_name = dump_path.parent / dump_path.name.replace(f'n{N}', f'n{gi}')
        print(f'Saving current state to "{dump_name.name}" before exiting')
        canceled = True

    if not transformer.batch_support:
        X = samples  # Use all samples
        X_global_mean = X.mean(axis=0, keepdims=True, dtype=np.float32)
        X -= X_global_mean

        print(f'[{timestamp()}] Fitting whole batch')
        t_start_fit = datetime.datetime.now()

        transformer.fit(X)

        print(f'[{timestamp()}] Done in {datetime.datetime.now() - t_start_fit}')
        assert np.all(transformer.transformer.mean_ < 1e-3), 'Mean of normalized data should be zero'
    else:
        X_global_mean = transformer.transformer.mean_.reshape((1, sample_dims))
        X = X.reshape(-1, sample_dims)
        X -= X_global_mean

    X_comp, X_stdev, X_var_ratio = transformer.get_components()

    assert X_comp.shape[1] == sample_dims \
        and X_comp.shape[0] == components \
        and X_global_mean.shape[1] == sample_dims \
        and X_stdev.shape[0] == components, 'Invalid shape'

    Z_comp = X_comp
    Z_global_mean = X_global_mean

    # Normalize
    Z_comp /= np.linalg.norm(Z_comp, axis=-1, keepdims=True)

    # Random projections
    # We expect these to explain much less of the variance
    random_dirs = get_random_dirs(components, np.prod(sample_shape))
    n_rand_samples = min(5000, X.shape[0])
    X_view = X[:n_rand_samples, :].T
    assert np.shares_memory(X_view, X), "Error: slice produced copy"
    X_stdev_random = np.dot(random_dirs, X_view).std(axis=1)

    # Inflate back to proper shapes (for easier broadcasting)
    X_comp = X_comp.reshape(-1, *sample_shape)
    X_global_mean = X_global_mean.reshape(sample_shape)
    Z_comp = Z_comp.reshape(-1, *input_shape)
    Z_global_mean = Z_global_mean.reshape(input_shape)

    # Compute stdev in latent space if non-Gaussian
    lat_stdev = np.ones_like(X_stdev)
    if use_w:
        seed_global = np.random.randint(np.iinfo(np.int32).max) # use (reproducible) global rand state
        rng = np.random.RandomState(seed_global)
        z = rng.standard_normal(512 * 5000).reshape(5000, 512)
        samples = Gs.components.mapping.run(z, None, dlatent_broadcast=None).reshape(5000, input_dims)
        coords = np.dot(Z_comp.reshape(-1, input_dims), samples.T)
        lat_stdev = coords.std(axis=1)

    np.savez_compressed(dump_path, **{
        'act_comp': X_comp.astype(np.float32),
        'act_mean': X_global_mean.astype(np.float32),
        'act_stdev': X_stdev.astype(np.float32),
        'lat_comp': Z_comp.astype(np.float32),
        'lat_mean': Z_global_mean.astype(np.float32),
        'lat_stdev': lat_stdev.astype(np.float32),
        'var_ratio': X_var_ratio.astype(np.float32),
        'random_stdevs': X_stdev_random.astype(np.float32),
    })

    if canceled:
        sys.exit(1)

    del X
    del X_comp
    del random_dirs
    del batch
    del samples
    del latents
