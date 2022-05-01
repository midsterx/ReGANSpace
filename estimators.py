# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import itertools
from types import SimpleNamespace

import numpy as np
from sklearn.decomposition import IncrementalPCA


# Incremental PCA
class IPCAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.whiten = False
        self.transformer = IncrementalPCA(n_components, whiten=self.whiten, batch_size=max(100, 2 * n_components))
        self.batch_support = True

    def get_param_str(self):
        return "ipca_c{}{}".format(self.n_components, '_w' if self.whiten else '')

    def fit(self, X):
        self.transformer.fit(X)

    def fit_partial(self, X):
        try:
            self.transformer.partial_fit(X)
            self.transformer.n_samples_seen_ = \
                self.transformer.n_samples_seen_.astype(np.int64)  # avoid overflow
            return True
        except ValueError as e:
            print(f'\nIPCA error:', e)
            return False

    def get_components(self):
        stdev = np.sqrt(self.transformer.explained_variance_)  # already sorted
        var_ratio = self.transformer.explained_variance_ratio_
        return self.transformer.components_, stdev, var_ratio  # PCA outputs are normalized


def get_estimator(name, n_components, alpha):
    if name == 'ipca':
        return IPCAEstimator(n_components)
    else:
        raise RuntimeError('Unknown estimator')
