import numpy as np
import torch
from torch.distributions import MultivariateNormal as MNormal
import scipy.special as sp
import levy

from typing import Tuple, Optional
from tqdm.auto import tqdm, trange

def langevin(
        start: torch.FloatTensor, 
        target, 
        n_samples: int,
        step_size: float, 
        burn_in: int=0,
        step_distr = None,
        normalize: bool=False,
        mh: bool=False,
        verbose: bool=False) -> Tuple[torch.FloatTensor, np.ndarray]:
    """
    Langevin Algorithm with Normal proposal

    Args:
        start - strating points of shape [n_chains, dim]
        target - target distribution instance with method "log_prob"
        n_samples - number of last samples from each chain to return
        burn_in - number of first samples from each chain to throw away
        step_distr - step size distribution (None for constant)
        normalize - whether to normalize step_size by step_distr mean
        mh - False to run ULA, True to run MALA
        verbose - whether to show iterations' bar

    Returns:
        tensor of chains with shape [n_samples, n_chains, dim], acceptance rates for each iteration
    """
    if normalize and step_distr is not None:
        step_size = step_size / step_distr.mean.item()

    n_chains, dim = start.shape
    std_normal = MNormal(torch.zeros(start.shape[-1]), torch.eye(start.shape[-1]))
    chains = []
    acceptance_rate = []
    
    x = start.clone()
    x.requires_grad_(True)
    x.grad = None
    logp_x = target.log_prob(x)
    grad_x = torch.autograd.grad(logp_x.sum(), x)[0]

    range_ = trange if verbose else range
    for step_id in range_(n_samples + burn_in):
        noise =  torch.randn_like(x)
        if step_distr is not None:
            cur_step = step_size * step_distr.sample((n_chains, 1))
        else:
            cur_step = step_size
        noise_coef = np.sqrt(2 * cur_step, dtype=np.float32)
        y = x + cur_step * grad_x + noise * noise_coef

        if mh:
            logp_y = target.log_prob(y)
            grad_y = torch.autograd.grad(logp_y.sum(), y)[0]

            log_qyx = std_normal.log_prob(noise)
            log_qxy = std_normal.log_prob((x - y - cur_step * grad_y) /  noise_coef)
            
            accept_prob = torch.clamp((logp_y + log_qxy - logp_x - log_qyx).exp(), max=1)
            mask = torch.rand_like(accept_prob) < accept_prob

            with torch.no_grad():
                x[mask, :] = y[mask, :]
                logp_x[mask] = logp_y[mask]
                grad_x[mask] = grad_y[mask]
            acceptance_rate.append(mask.float().mean().item())
        else:
            x = y
            logp_x = target.log_prob(x)
            grad_x = torch.autograd.grad(logp_x.sum(), x)[0]
        if step_id >= burn_in:
            chains.append(x.detach().data.clone())
    chains = torch.stack(chains, 0)
    return chains, np.array(acceptance_rate)

def stable_langevin(
        start: torch.FloatTensor, 
        target, 
        n_samples: int,
        step_size: Optional[float]=None,
        step_a: float=1e-6, 
        step_b: float=0.6,
        burn_in: int=0,
        mh: bool=False,
        alpha=1.8,
        verbose: bool=False) -> Tuple[torch.FloatTensor, np.ndarray]:
    """
    Langevin Algorithm with Normal proposal

    Args:
        start - strating points of shape [n_chains, dim]
        target - target distribution instance with method "log_prob"
        n_samples - number of last samples from each chain to return
        burn_in - number of first samples from each chain to throw away
        step_distr - step size distribution (None for constant)
        normalize - whether to normalize step_size by step_distr mean
        mh - False to run ULA, True to run MALA
        alpha - parameter for alpha-stable distribution
        verbose - whether to show iterations' bar

    Returns:
        tensor of chains with shape [n_samples, n_chains, dim], acceptance rates for each iteration
    """

    beta = 0
    C = sp.gamma(alpha-1) / sp.gamma(alpha/2)**2
    n_chains, dim = start.shape
    chains = []
    acceptance_rate = []
    
    x = start.clone()
    x.requires_grad_(True)
    x.grad = None
    logp_x = target.log_prob(x)
    grad_x = torch.autograd.grad(logp_x.sum(), x)[0]

    range_ = trange if verbose else range
    for step_id in range_(n_samples + burn_in):
        if step_size is not None:
            cur_step = step_size
        else:
            cur_step = (step_a / (step_id+1))**step_b
        noise = torch.from_numpy(levy.random(alpha, beta, shape=x.shape)).to(x.dtype)
        noise_coef = cur_step ** (1/alpha)
        y = x + C * cur_step * grad_x + noise * noise_coef

        if mh:
            logp_y = target.log_prob(y)
            grad_y = torch.autograd.grad(logp_y.sum(), y)[0]

            log_qyx = levy.levy(noise.numpy(force=True), alpha, beta)
            log_qxy = levy.levy(((x - y - C*cur_step*grad_y) / noise_coef).numpy(force=True), alpha, beta)
            log_qyx = torch.from_numpy(log_qyx).sum(dim=-1)
            log_qxy = torch.from_numpy(log_qxy).sum(dim=-1)
            
            accept_prob = torch.clamp((logp_y + log_qxy - logp_x - log_qyx).exp(), max=1)
            mask = torch.rand_like(accept_prob) < accept_prob

            with torch.no_grad():
                x[mask, :] = y[mask, :]
                logp_x[mask] = logp_y[mask]
                grad_x[mask] = grad_y[mask]
            acceptance_rate.append(mask.float().mean().item())
        else:
            x = y
            logp_x = target.log_prob(x)
            grad_x = torch.autograd.grad(logp_x.sum(), x)[0]
        if step_id >= burn_in:
            chains.append(x.detach().data.clone())
    chains = torch.stack(chains, 0)
    return chains, np.array(acceptance_rate)