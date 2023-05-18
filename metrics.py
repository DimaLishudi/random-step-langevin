import numpy as np
import torch
from scipy.stats import gaussian_kde
import ot

def get_modes_coverage(chains, locs, sigma, k_threshold=5):
    '''
        args:
            chains: torch.Tensor[n_samples, n_chains, dim] - sampled chains
            locs: torch.Tensor[N_Mixtures, dim] - centers of gaussians
            sigma: int - sigma of all gaussians (assume they have diagonal cov)
            k_threshold: int - minimum number of points from chain near center of gaussian to consider it covered
    
        returns:
            n_covered: torch.tensor[n_chains] -- number of covered modes for each chain
    '''
    n_samples, n_chains, dim = chains.shape
    chains = chains.clone().transpose(0, 1).reshape(n_chains, n_samples, 1, dim)
    dists = torch.linalg.norm(chains - locs.view(1, 1, -1, dim), dim=-1, ord=2)
    # dists: torch.Tensor[n_chains, n_samples, N_mixtures] -- distances to modes
    mode_idxs = dists.argmin(dim=-1)
    close_ind = torch.amin(dists, dim=-1) <= 2*sigma*np.sqrt(dim)
    results = np.zeros(n_chains)
    for i in range(n_chains):
        _, counts = torch.unique(mode_idxs[i][close_ind[i]], return_counts=True)
        results[i] = torch.sum(counts >= k_threshold)
    return results


def TV_1d(chain, reference, ref_kde=None, min=-20, max=20, grid_size=1000):
    '''
        args:
            chain: torch.Tensor[n_samples] - sampled chains
            reference: torch.Tensor[n_samples] - reference sample from true distribution
            ref_kde: kernel density estimation fitted on reference points (optional)
            min, max: min and max values for grid
            grid_size: number of grid segments
    
        returns:
            TV: 1d TV between KDEs of chain and reference points
    '''
    grid = np.linspace(min, max, grid_size)
    chain_kde = gaussian_kde(chain)
    if ref_kde is None:
        ref_kde = gaussian_kde(reference)
    res = np.abs(chain_kde(grid) - ref_kde(grid)).mean()
    return res * (max-min) / 2


def sliced_TV(chain, reference, n_proj=20, **kwargs):
    '''
        args:
            chain: torch.Tensor[n_samples, dim] - sampled chains
            reference: torch.Tensor[n_samples, dim] - reference sample from true distribution
            n_proj: number of random projections
    
        returns:
            sliced_TV: average 1d TV over n_proj projections
    '''
    chain = np.asarray(chain)
    reference = np.asarray(reference)
    n_samples, dim = chain.shape
    vecs = np.random.randn(dim, n_proj)
    vecs /= np.linalg.norm(vecs, ord=2, axis=1, keepdims=True)

    chain_proj = chain @ vecs
    ref_proj = reference @ vecs

    res = sum(TV_1d(chain_proj[:,i], ref_proj[:,i], **kwargs) for i in range(n_proj))
    return res / n_proj


@torch.inference_mode()
def calc_ess(chains):
    '''
        Computes Effective Sample Size for generated markov chain
        see http://www.stat.columbia.edu/~gelman/book/BDA3.pdf
        args:
            chains: torch.Tensor[n_samples, n_chains, dim] - sampled chains
    
        returns:
            ess: -- effective sample size coefficient averaged over dimensions
    '''
    n_samples, n_chains, dim = chains.shape
    chains = chains.transpose(0, 2) # dim, n_chains, n_samples

    means = chains.mean(dim=(1,2), keepdim=True)
    # within-sequence
    W = torch.mean((chains - chains.mean(dim=2, keepdim=True))**2, dim=(1,2))
    # between-sequence
    B = torch.sum((chains.mean(dim=2, keepdim=True) - means)**2, dim=(1,2)) / (n_chains-1)
    var = B + W

    corr_sum = torch.zeros(dim)

    # remaining_dims = torch.ones(dim, dtype=bool)
    for k in range(1, n_samples):
        cur_corr = torch.mean((chains[:,:,k:] - chains[:,:,:-k])**2, dim=(1,2))
        cur_corr = 1 - (0.5*cur_corr)/var
        if cur_corr.mean() <= 0.005:
            break
        corr_sum += cur_corr
    res = torch.mean(1 / (1 + 2*corr_sum))
    return res.item()


def calc_emd(chain, reference):
    '''
        args:
            chain: torch.Tensor[n_samples, dim] - sampled chain
            reference: torch.Tensor[n_samples, dim] - reference sample from true distribution
    
        returns:
            emd: Earth Mover's Distance
    '''
    d = chain.shape[1]
    return ot.emd2([], [], ot.dist(reference.numpy(), chain.numpy())) / d