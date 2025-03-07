# ===============================================================================
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

from onedal.primitives.kernel_functions import rbf_kernel as onedal_rbf_kernel

def rbf_kernel(X, Y=None, gamma=None, queue=None):
    """
    Compute the RBF (Gaussian) kernel between X and Y:
        K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features)
    gamma : float, default=None
        If None, defaults to 1.0 / n_features.
    queue : optional
        Execution queue.

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """
    return onedal_rbf_kernel(X, Y, gamma, queue)
