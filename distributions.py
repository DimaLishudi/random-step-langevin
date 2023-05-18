import numpy as np
import torch
import torch.distributions as D
from torch.distributions.distribution import Distribution
from torch.distributions import Normal, Cauchy, Categorical, Beta, MixtureSameFamily, Independent
from torch.distributions import MultivariateNormal as MNormal
from scipy.stats import gaussian_kde

from typing import Optional
import matplotlib.pyplot as plt


PROJ_DIM1 = -2
PROJ_DIM2 = -1


class BernoulliStep(Distribution):
    def __init__(self, val1, val2, p=0.5):
        super().__init__()
        self.vals = torch.tensor([val1, val2])
        self.bern = D.Categorical(probs=torch.tensor([1-p, p]))
        self.p = p
        
    def sample(self, size):
        idxs = self.bern.sample(size).long()
        return self.vals[idxs]

    @property
    def probs(self):
        return self.p

    @property
    def mean(self):
        mean = self.vals[1] * self.bern.probs.item() +\
               self.vals[0] * (1-self.bern.probs.item())
        return mean

    @property
    def arg_constraints(self):
        return {}
    
@torch.inference_mode()
def plot_distribution(distr: D.distribution, name=None, left=-5, right=5, N=1000):
    plt.figure(figsize=(6, 4))
    if name is None:
        name = f"D: {distr}, mean={round(distr.mean.item(), 1)}"
    plt.title(name)

    sup = distr.support
    if hasattr(sup, "lower_bound"):
        left = max(left, sup.lower_bound)
    if hasattr(sup, "upper_bound"):
        right = min(right, sup.upper_bound)
    
    if sup.is_discrete:
        x = torch.arange(left, right+1, dtype=int)
        probs = torch.exp(distr.log_prob(x))
        plt.bar(x, probs)
        return
    x = torch.linspace(left, right, N)
    probs = torch.exp(distr.log_prob(x.unsqueeze(1)))
    plt.plot(x, probs)


def plot_kdes(kde_list, locs, covs, name="", N=1000):
    plt.figure(figsize=(6, 4))
    alpha = 1/np.sqrt(len(kde_list))

    rad   = covs.max() **.5 * 5
    left  = locs.min()-rad
    right = locs.max()+rad

    for kde in kde_list:
        x = np.linspace(left, right, N)
        probs = kde.pdf(x)
        plt.plot(x, probs, alpha=alpha)
    plt.title(name)


class MoG(object):
    """
    Mixture of Gaussians distribution.

    Args:
        locs - locations of mean parameters for each Gaussian
        covs - covariances for each Gaussian
    """
    def __init__(self, 
            locs: torch.FloatTensor, 
            covs: torch.FloatTensor, 
            weights: Optional[torch.FloatTensor]=None):
        self.n_comp = len(locs)
        self.locs = locs
        self.covs = covs
        self.weights = weights if weights is not None else \
            torch.ones(self.n_comp, device=locs.device)
        self.weights /= self.weights.sum()
        self.gaussians = [MNormal(loc, cov) for loc, cov in zip(locs, covs)]
        mix = Categorical(self.weights)
        comp = Independent(MNormal(locs, covs), 0)
        self.mog = MixtureSameFamily(mix, comp)

    def sample(self, size: torch.Size):
        return self.mog.sample(size)

    @property
    def dim(self) -> int:
        return self.locs.shape[-1]

    def log_prob(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.mog.log_prob(x)
        
    def display_1d(self, points=None, label=None, grid_size=1000, ax=None):
        if self.dim != 1:
            return
        
        if ax is None:
            ax = plt
            plt.title("Mixture Of Gaussians")

        rad = self.covs.max() **.5 * 5
        left = self.locs.min()-rad
        right =  self.locs.max()+rad
        x = torch.linspace(left, right, grid_size)
        probs = torch.exp(self.log_prob(x.unsqueeze(1)))
        ax.plot(x, probs)
        if points is not None:
            points = points.clamp(left, right)
            ax.scatter(points, torch.exp(self.log_prob(points.unsqueeze(1))), label=label, marker="x", c="red", alpha=0.2, s=5**2)
            ax.legend()
        
    def plot_2d_countour(self, ax):
        rad = self.covs.max() **.5 * 5
        left = self.locs.min()-rad
        right =  self.locs.max()+rad
        x = np.linspace(left, right, 100)
        y = np.linspace(left, right, 100)
        X, Y = np.meshgrid(x, y)
        inp = torch.FloatTensor(np.stack([X, Y], -1))

        proj_slice = [PROJ_DIM1, PROJ_DIM2]
        gaussians = [MNormal(loc[proj_slice], cov[proj_slice, :][:, proj_slice]) for loc, cov in zip(self.locs, self.covs)]
        log_ps = torch.stack([
            torch.log(weight) + gauss.log_prob(inp.reshape(-1, 2)) for weight, gauss in zip(self.weights, gaussians)
            ], dim=0)
        Z = torch.logsumexp(log_ps, dim=0).reshape(inp.shape[:-1])
        #levels = np.quantile(Z, np.linspace(0.9, 0.99, 5))

        ax.contour(X, Y, Z.exp(), 
                   #levels = levels,
                   levels=10, 
                   alpha=1., cmap='inferno')
        

def plot_result(chains, dist, chain_id=0):
    proj_slice = [PROJ_DIM1, PROJ_DIM2]
    proj_dim1 = dist.dim + PROJ_DIM1 + 1 if PROJ_DIM1 < 0 else PROJ_DIM1 + 1
    proj_dim2 = dist.dim + PROJ_DIM2 + 1 if PROJ_DIM2 < 0 else PROJ_DIM2 + 1

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    n_chains = chains.shape[1]
    result = chains.reshape(-1, chains.shape[-1])
    dist.plot_2d_countour(axs[0])
    xmin, xmax = axs[0].get_xlim()
    ymin, ymax = axs[0].get_ylim()

    axs[0].scatter(*result[:, proj_slice].T, alpha=min(0.6, 1000./result.shape[0]), s=10)
    axs[0].set_title(f'{n_chains} chains')

    kernel = gaussian_kde(result[:, proj_slice].T)
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde = np.reshape(kernel(positions).T, X.shape)
    axs[1].contour(X, Y, kde, cmap='inferno')
    axs[1].set_title(f'KDE')

    # chain_id = 0
    # result = chains[:, chain_id, :]
    # dist.plot_2d_countour(axs[1,0])
    # axs[1,0].scatter(*result[:, proj_slice].T, s = 10)
    # axs[1,0].set_title(f'Trajectory of chain {chain_id}')

    # if len(np.unique(result[:, proj_slice], axis=0)) > 0:
    #   try:
    #     kernel = gaussian_kde(np.unique(result[:, proj_slice], axis=0).T)
    #     x = np.linspace(xmin, xmax, 100)
    #     y = np.linspace(ymin, ymax, 100)
    #     X, Y = np.meshgrid(x, y)
    #     positions = np.vstack([X.ravel(), Y.ravel()])
    #     kde = np.reshape(kernel(positions).T, X.shape)
    #     axs[1,1].contour(X, Y, kde, cmap='inferno')
    #     axs[1,1].set_title(f'KDE')
    #   except np.linalg.LinAlgError:
    #     pass


    for i in range(2):
    #   for j in range(2):
        axs[i].set_xlim(xmin, xmax)
        axs[i].set_ylim(ymin, ymax)
        axs[i].set_xlabel(fr'$X{proj_dim1}$')
        axs[i].set_ylabel(fr'$X{proj_dim2}$')
        # ax.axis('square')

    fig.tight_layout()
    plt.show()
    

def plot_kdes(kde_list, locs, covs, name="", N=1000, ax=None):
    if ax is None:
        ax = plt
    alpha = 1/np.sqrt(len(kde_list))

    rad   = covs.max() **.5 * 5
    left  = locs.min()-rad
    right = locs.max()+rad

    for kde in kde_list:
        x = np.linspace(left, right, N)
        probs = kde.pdf(x)
        ax.plot(x, probs, alpha=alpha)
    if name != "" or None:
        if ax is not plt:
            ax.set_title(name)
        else:
            plt.title(name)