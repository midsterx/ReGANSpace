# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import hashlib
import string
import sys
sys.path.append('./models/stylegan2')

import numpy as np


# StyleGAN1/2 classes
out_classes = {
    1: {
        "celebahq": "https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf",
        "ffhq": "https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ",
        "wikiart": "https://drive.google.com/uc?id=1cJQtMeTy_QldOP7n64F8stCDXY6Esup9",
        "anime": "https://mega.nz/#!vawjXISI!F7s13yRicxDA3QYqYDL2kjnc2K7Zk3DwCIYETREmBP4"
    },
    2: {
        "cars": "gdrive:networks/stylegan2-car-config-f.pkl",
        "cats": "gdrive:networks/stylegan2-cat-config-f.pkl",
        "ffhq": "gdrive:networks/stylegan2-ffhq-config-f.pkl",
        "horse": "gdrive:networks/stylegan2-horse-config-f.pkl",
        "ukiyoe": "gdrive:networks/ukiyoe-256-slim-diffAug-002789.pkl",
        "beetles": "gdrive:networks/beetles.pkl",
        "anime": "mega:file/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl.xz"
    }
}


def prettify_name(name):
    valid = "-_%s%s" % (string.ascii_letters, string.digits)
    return ''.join(map(lambda c: c if c in valid else '_', name))


# Add padding to sequence of images
# Used in conjunction with np.hstack/np.vstack
# By default: adds one 64th of the width of horizontal padding
def pad_frames(strip, pad_fract_horiz=64, pad_fract_vert=0, pad_value=None):
    dtype = strip[0].dtype
    if pad_value is None:
        if dtype in [np.float32, np.float64]:
            pad_value = 1.0
        else:
            pad_value = np.iinfo(dtype).max

    frames = [strip[0]]
    for frame in strip[1:]:
        if pad_fract_horiz > 0:
            frames.append(pad_value * np.ones((frame.shape[0], frame.shape[1] // pad_fract_horiz, 3), dtype=dtype))
        elif pad_fract_vert > 0:
            frames.append(pad_value * np.ones((frame.shape[0] // pad_fract_vert, frame.shape[1], 3), dtype=dtype))
        frames.append(frame)
    return frames


def centre_strip_stylegan(Gs, Gs_kwargs, z, lat_comp, lat_mean, lat_stdev, idx, sigma, num_frames, layer_start, layer_end):

        input_shape = Gs.input_shape[1]
        num_layers = Gs.components.mapping.output_shape[1]
    
        w_avg = Gs.get_var('dlatent_avg')

        w = Gs.components.mapping.run(z, None, dlatent_broadcast=None)
        
        w = w.reshape((1, input_shape))

        sigma_range = np.linspace(-sigma, sigma, num_frames)
        
        dotp = np.sum((w - lat_mean) * lat_comp[idx] / (np.linalg.norm(lat_comp[idx]) + 1e-8), axis=-1, keepdims=True)
        zeroing_offset_lat = dotp * lat_comp[idx] / (np.linalg.norm(lat_comp[idx] + 1e-8))

        batch_frames = []

        for j in range(len(sigma_range)):
            
            ws = Gs.components.mapping.run(z, None)
            ws = ws.reshape((num_layers, 1, input_shape))

            s = sigma_range[j]

            delta = lat_comp[idx] * s * lat_stdev[idx]

            for k in range(layer_start, layer_end):
                ws[k] = ws[k] - zeroing_offset_lat + delta

            ws = w_avg + (ws - w_avg) * Gs_kwargs.truncation_psi
            imgs = Gs.components.synthesis.run(ws.reshape((1, num_layers, input_shape)), **Gs_kwargs)

            batch_frames.append(imgs[0])
        
        return batch_frames


def get_hash(url):
    url_ffhq        = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    url_celebahq    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl
    url_bedrooms    = 'https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF' # karras2019stylegan-bedrooms-256x256.pkl
    url_cars        = 'https://drive.google.com/uc?id=1MJ6iCfNtMIRicihwRorsM3b7mmtmK9c3' # karras2019stylegan-cars-512x384.pkl
    url_cats        = 'https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ' # karras2019stylegan-cats-256x256.pkl
    url_wikiart     = 'https://drive.google.com/uc?id=1cJQtMeTy_QldOP7n64F8stCDXY6Esup9' # network-snapshot-011125.pkl

    url = url.encode("utf-8")
    return hashlib.md5(url).hexdigest()
