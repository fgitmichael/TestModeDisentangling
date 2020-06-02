"""
Code from MIGE
https://arxiv.org/abs/2005.01123
https://github.com/zhouyiji/MIGE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


# ## SSGE
#
# MIGE is rely on the implicit distributions' score estimation method Spectral Stein Gradient Estimation (SSGE)[2]. We implement a PyTorch version.
#
#
# The official SSGE implementation (tensorflow):
# https://github.com/thjashin/spectral-stein-grad
#
# [2]. Jiaxin Shi, Shengyang Sun, and Jun Zhu. A spectral approach to gradient estimation for implicit distributions. arXiv preprint arXiv:1806.02925, 2018.

# In[11]:


class ScoreEstimator:
    def __init__(self):
        pass

    def rbf_kernel(self, x1, x2, kernel_width):
        return torch.exp(
            -torch.sum(torch.mul((x1 - x2), (x1 - x2)), dim=-1) / (2 * torch.mul(kernel_width, kernel_width))
        )

    def gram(self, x1, x2, kernel_width):
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        kernel_width = kernel_width[..., None, None]
        return self.rbf_kernel(x_row, x_col, kernel_width)

    def grad_gram(self, x1, x2, kernel_width):
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        kernel_width = kernel_width[..., None, None]
        G = self.rbf_kernel(x_row, x_col, kernel_width)
        diff = (x_row - x_col) / (kernel_width[..., None] ** 2)
        G_expand = torch.unsqueeze(G, -1)
        grad_x2 = G_expand * diff
        grad_x1 = G_expand * (-diff)
        return G, grad_x1, grad_x2

    def heuristic_kernel_width(self, x_samples, x_basis):
        n_samples = x_samples.size()[-2]
        n_basis = x_basis.size()[-2]
        x_samples_expand = torch.unsqueeze(x_samples, -2)
        x_basis_expand = torch.unsqueeze(x_basis, -3)
        pairwise_dist = torch.sqrt(
            torch.sum(torch.mul(x_samples_expand - x_basis_expand, x_samples_expand - x_basis_expand), dim=-1)
        )
        k = n_samples * n_basis // 2
        top_k_values = torch.topk(torch.reshape(pairwise_dist, [-1, n_samples * n_basis]), k=k)[0]
        kernel_width = torch.reshape(top_k_values[:, -1], x_samples.size()[:-2])
        return kernel_width.detach()

    def compute_gradients(self, samples, x=None):
        raise NotImplementedError()


class SpectralScoreEstimator(ScoreEstimator):
    def __init__(self, n_eigen=None, eta=None, n_eigen_threshold=None):
        self._n_eigen = n_eigen
        self._eta = eta
        self._n_eigen_threshold = n_eigen_threshold
        super().__init__()

    def nystrom_ext(self, samples, x, eigen_vectors, eigen_values, kernel_width):
        M = torch.tensor(samples.size()[-2]).to(samples.device)
        Kxq = self.gram(x, samples, kernel_width)
        ret = torch.sqrt(M.float()) * torch.matmul(Kxq, eigen_vectors)
        ret *= 1. / torch.unsqueeze(eigen_values, dim=-2)
        return ret

    def compute_gradients(self, samples, x=None):
        if x is None:
            kernel_width = self.heuristic_kernel_width(samples, samples)
            x = samples
        else:
            _samples = torch.cat([samples, x], dim=-2)
            kernel_width = self.heuristic_kernel_width(_samples, _samples)

        M = samples.size()[-2]
        Kq, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        if self._eta is not None:
            Kq += self._eta * torch.eye(M)

        eigen_values, eigen_vectors = torch.symeig(Kq, eigenvectors=True, upper=True)

        if (self._n_eigen is None) and (self._n_eigen_threshold is not None):
            eigen_arr = torch.mean(
                torch.reshape(eigen_values, [-1, M]), dim=0)

            eigen_arr = torch.flip(eigen_arr, [-1])
            eigen_arr /= torch.sum(eigen_arr)
            eigen_cum = torch.cumsum(eigen_arr, dim=-1)
            eigen_lt = torch.lt(eigen_cum, self._n_eigen_threshold)
            self._n_eigen = torch.sum(eigen_lt)
        if self._n_eigen is not None:
            eigen_values = eigen_values[..., -self._n_eigen:]
            eigen_vectors = eigen_vectors[..., -self._n_eigen:]
        eigen_ext = self.nystrom_ext(samples, x, eigen_vectors, eigen_values, kernel_width)
        grad_K1_avg = torch.mean(grad_K1, dim=-3)
        M = torch.tensor(M).to(samples.device)
        beta = -torch.sqrt(M.float()) * torch.matmul(torch.transpose(eigen_vectors, -1, -2),
                                                     grad_K1_avg) / torch.unsqueeze(eigen_values, -1)
        grads = torch.matmul(eigen_ext, beta)
        self._n_eigen = None
        return grads


def entropy_surrogate(estimator, samples):
    # detach to not backprob through sample creation
    dlog_q = estimator.compute_gradients(samples.detach(), None)
    # detach to only push gradients through the sample process
    # (rho requires grad, samples depend on rho -> equal grad_psi in formula 11)
    surrogate_cost = torch.mean(torch.sum(dlog_q.detach() * samples, -1))
    return surrogate_cost


def MIGE(d, range_rho, num_sample, GenerateData, threshold=None, n_eigen=None):
    spectral_j = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=threshold)
    spectral_m = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=threshold)
    approximations = []
    for rho in range_rho:
        rho = torch.FloatTensor([rho]).cuda()
        rho.requires_grad = True
        xs_ys, xs, ys = GenerateData(d, rho, num_sample)

        ans = entropy_surrogate(spectral_j, xs_ys) - entropy_surrogate(spectral_m, ys)

        # Backward to get gradients with respect to rho
        ans.backward()
        approximations.append(rho.grad.data)

    approximations = torch.stack(approximations).view(-1).detach().cpu().numpy()
    return approximations

