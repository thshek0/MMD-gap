# coding=utf-8
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
# Note it requires estimatly 2.5 mins to run on projgpu33 (on train)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import math
import os
import argparse
import random
import numpy as np
from typing import Any, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
# from torchmetrics.image.fid import NoTrainInceptionV3

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import utils.losses as losses

def maximum_mean_discrepancy(k_xx: Tensor, k_xy: Tensor, k_yy: Tensor) -> Tensor:
    """Adapted from `KID Score`_."""
    m = k_xx.shape[0]

    diag_x = torch.diag(k_xx)
    diag_y = torch.diag(k_yy)

    kt_xx_sums = k_xx.sum(dim=-1) - diag_x
    kt_yy_sums = k_yy.sum(dim=-1) - diag_y
    k_xy_sums = k_xy.sum(dim=0)

    kt_xx_sum = kt_xx_sums.sum()
    kt_yy_sum = kt_yy_sums.sum()
    k_xy_sum = k_xy_sums.sum()

    value = (kt_xx_sum + kt_yy_sum) / (m * (m - 1))
    value -= 2 * k_xy_sum / (m**2)
    return value


def poly_kernel(f1: Tensor, f2: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0) -> Tensor:
    """Adapted from `KID Score`_."""
    if gamma is None:
        gamma = 1.0 / f1.shape[1]
    return (f1 @ f2.T * gamma + coef) ** degree


def poly_mmd(
    f_real: Tensor, f_fake: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0
) -> Tensor:
    """Adapted from `KID Score`_."""
    k_11 = poly_kernel(f_real, f_real, degree, gamma, coef)
    k_22 = poly_kernel(f_fake, f_fake, degree, gamma, coef)
    k_12 = poly_kernel(f_real, f_fake, degree, gamma, coef)
    return maximum_mean_discrepancy(k_11, k_12, k_22)


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def gaussian_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenation along the zero dimension."""
    if isinstance(x, torch.Tensor):
        return x
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)


def compute_kid_from_embedding(
        real_features: Tensor,
        fake_features: Tensor,
        kernel_fn: str = "rbf",
        subsets: int = 1000,
        subset_size: int = 100,
        kernel_mul=2.0,
        kernel_num=5,
        fix_sigma=None,
        degree: int = 3,
        gamma: Optional[float] = None,
        coef: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """Calculate KID score based on accumulated extracted features from the two distributions.

    Implementation inspired by `Fid Score`_

    """
    real_features = dim_zero_cat(real_features)
    fake_features = dim_zero_cat(fake_features)

    n_samples_real = real_features.shape[0]
    if n_samples_real < subset_size:
        raise ValueError(
            "Argument `subset_size` should be smaller than the number of samples")
    n_samples_fake = fake_features.shape[0]
    if n_samples_fake < subset_size:
        raise ValueError(
            "Argument `subset_size` should be smaller than the number of samples")

    kid_scores_ = []
    for _ in range(subsets):
        perm = torch.randperm(n_samples_real)
        f_real = real_features[perm[: subset_size]]
        perm = torch.randperm(n_samples_fake)
        f_fake = fake_features[perm[: subset_size]]

        if kernel_fn == "rbf":
            o = gaussian_mmd(f_real, f_fake, kernel_mul, kernel_num, fix_sigma)
        else:
            o = poly_mmd(f_real, f_fake, degree, gamma, coef)
        kid_scores_.append(o)
    kid_scores = torch.stack(kid_scores_)
    return kid_scores.mean(), kid_scores.std(unbiased=False)


def compute_mmdgap(train_features, test_features, fake_features):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    train_embedding_tensor = torch.from_numpy(train_features).to("cuda:0")
    test_embedding_tensor = torch.from_numpy(test_features).to("cuda:0")
    custom_gen_embedding_tensor = torch.from_numpy(fake_features).to("cuda:0")

    kid_train = compute_kid_from_embedding(
        train_embedding_tensor, custom_gen_embedding_tensor)[0].item()
    kid_test = compute_kid_from_embedding(
        test_embedding_tensor, custom_gen_embedding_tensor)[0].item()
    kid_train_baseline = compute_kid_from_embedding(
        train_embedding_tensor, train_embedding_tensor)[0].item()
    kid_test_baseline = compute_kid_from_embedding(
        test_embedding_tensor, train_embedding_tensor)[0].item()

    return dict(value=kid_test - kid_train,
                baseline=kid_test_baseline - kid_train_baseline)

def compute_real_embeddings(data_loader, batch_size, eval_model, quantize, world_size, DDP, disable_tqdm):
    data_iter = iter(data_loader)
    num_batches = int(math.ceil(float(len(data_loader.dataset)) / float(batch_size)))
    if DDP: num_batches = num_batches = int(math.ceil(float(len(data_loader.dataset)) / float(batch_size*world_size)))

    real_embeds = []
    for i in tqdm(range(num_batches), disable=disable_tqdm):
        try:
            real_images, real_labels = next(data_iter)
        except StopIteration:
            break

        real_images, real_labels = real_images.to("cuda"), real_labels.to("cuda")

        with torch.no_grad():
            real_embeddings, _ = eval_model.get_outputs(real_images, quantize=quantize)
            real_embeds.append(real_embeddings)

    real_embeds = torch.cat(real_embeds, dim=0)
    if DDP: real_embeds = torch.cat(losses.GatherLayer.apply(real_embeds), dim=0)
    real_embeds = np.array(real_embeds.detach().cpu().numpy(), dtype=np.float64)
    return real_embeds[:len(data_loader.dataset)]


def calculate_mmd(train_feats, eval_feats, fake_feats, train_data_loader, eval_data_loader, eval_model, num_generate, cfgs, quantize,
                    world_size, DDP, disable_tqdm):
    eval_model.eval()

    rng = np.random.default_rng(42)

    if train_feats is None:
        train_embeds = compute_real_embeddings(data_loader=train_data_loader,
                                              batch_size=cfgs.OPTIMIZATION.batch_size,
                                              eval_model=eval_model,
                                              quantize=quantize,
                                              world_size=world_size,
                                              DDP=DDP,
                                              disable_tqdm=disable_tqdm)
    else:
        train_embeds = train_feats
        
    if eval_feats is None:
        eval_embeds = compute_real_embeddings(data_loader=eval_data_loader,
                                              batch_size=cfgs.OPTIMIZATION.batch_size,
                                              eval_model=eval_model,
                                              quantize=quantize,
                                              world_size=world_size,
                                              DDP=DDP,
                                              disable_tqdm=disable_tqdm)
    else:
        eval_embeds = eval_feats

    fake_embeds = np.array(fake_feats.detach().cpu().numpy(), dtype=np.float64)[:num_generate]

    num_output_samples = min(train_embeds.shape[0], fake_embeds.shape[0])

    for method in ['none', 'pca_100', 'rp_100', 'pca_500', 'rp_500']:
        if method == 'none':
            reduced_train_embeds = train_embeds
            reduced_eval_embeds = eval_embeds
            reduced_fake_embeds = fake_embeds
        else:
            if method == "pca_100":
                red_model = PCA(n_components=100)
            elif method == "rp_100":
                red_model = GaussianRandomProjection(n_components=100)
            elif method == "pca_500":
                red_model = PCA(n_components=500)
            elif method == "rp_500":
                red_model = GaussianRandomProjection(n_components=500)
            reduced_train_embeds = red_model.fit_transform(train_embeds)
            reduced_eval_embeds = red_model.fit_transform(eval_embeds)
            reduced_fake_embeds = red_model.fit_transform(fake_embeds)
            
        for alpha in np.linspace(0.0, 1.0, 11):
            indices1 = rng.choice(train_embeds.shape[0], int(alpha * num_output_samples), replace=False)
            indices2 = rng.choice(fake_embeds.shape[0], int((1-alpha) * num_output_samples), replace=False)
            combined_embeds = np.vstack((reduced_train_embeds[indices1], reduced_fake_embeds[indices2]))
            metrics = compute_mmdgap(train_features=reduced_train_embeds, test_features=reduced_eval_embeds, fake_features=combined_embeds)
            print(method, alpha, metrics)

    val, base = metrics["value"], metrics["baseline"]
    return val, base