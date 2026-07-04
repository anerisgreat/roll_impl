import torch
import logging
from functools import partial
from torch.autograd import Function
# from .beta_dist import Beta
import numpy as np
from KDEpy.bw_selection import improved_sheather_jones


# ── Gaussian ROLL helpers ─────────────────────────────────────────────────────

def _torch_mean_std(yh):
    yhmean = torch.mean(yh)
    yhstd = torch.std(yh)
    return yhmean, yhstd

def _torch_normal_fit(yh):
    yhmean, yhstd = _torch_mean_std(yh)
    return torch.distributions.normal.Normal(yhmean, yhstd)


# ── Batch splitting ───────────────────────────────────────────────────────────

def split_true_false_indeces(yh, y):
    true_indeces = torch.argwhere(y)[:, 0]
    false_indeces = torch.argwhere(torch.logical_not(y))[:, 0]
    return true_indeces, false_indeces

def split_true_false(yh, y):
    true_indeces, false_indeces = split_true_false_indeces(yh, y)
    return yh[true_indeces], yh[false_indeces]


# ── Gaussian ROLL ─────────────────────────────────────────────────────────────

def _roll_loss_from_fpr_internal(yh, y, fpr):
    true_yh, false_yh = split_true_false(yh, y)
    true_normal = _torch_normal_fit(true_yh)
    false_normal = _torch_normal_fit(false_yh)
    return true_normal.cdf(false_normal.icdf(torch.tensor(1 - fpr)))

def roll_loss_from_fpr(fpr):
    return partial(_roll_loss_from_fpr_internal, fpr=fpr)


# ── Beta ROLL ─────────────────────────────────────────────────────────────────

def _roll_beta_loss_from_fpr_internal(yh, y, fpr):
    yh = torch.nn.functional.sigmoid(yh)
    true_yh, false_yh = split_true_false(yh, y)
    true_beta = Beta.from_sample(true_yh)
    false_beta = Beta.from_sample(false_yh)
    return true_beta.cdf(false_beta.icdf(torch.tensor(1 - fpr)))

def roll_beta_loss_from_fpr(fpr):
    return partial(_roll_beta_loss_from_fpr_internal, fpr=fpr)

def roll_beta_aoc_loss(yh, y):
    yh = torch.nn.functional.sigmoid(yh)
    true_yh, false_yh = split_true_false(yh, y)
    true_beta = Beta.from_sample(true_yh)
    false_beta = Beta.from_sample(false_yh)
    r = 0
    for fpr in np.arange(0.001, 1.0, 0.001):
        return true_beta.cdf(false_beta.icdf(torch.tensor(1 - fpr)))
    return r / 1000


# ── Unused: moment-based NIG parameter estimation ────────────────────────────
# https://discuss.pytorch.org/t/statistics-for-whole-dataset/74511/2
def _calc_moments(arr):
    mean = torch.mean(arr)
    diffs = arr - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    return mean, var, skews, kurtoses

# Equations 2.3 in: Comparison of parameter estimation methods for normal inverse
# gaussian distribution (Jeongyoen Yoon, Jiyeon Kim, Seongjoo Song)
def _calc_gaussian_norm_inv_params_from_moments(mean, var, skews, kurtoses):
    gamma = 3 / (torch.sqrt(var) * torch.sqrt(3 * kurtoses - 5 * (skews ** 2)))
    beta  = (S * torch.sqrt(var) * (gamma ** 2)) / 3
    alpha = torch.sqrt(gamma ** 2 + beta ** 2)
    delta = (var * (gamma ** 3)) / ((gamma ** 2) + (beta ** 2))
    mu    = mean - (beta * delta) / gamma
    return (alpha, beta, gamma, delta, mu)


# ── KDE ROLL ──────────────────────────────────────────────────────────────────

class KernelizedROLLoss(Function):
    """KDE-ROLL loss with custom backward pass.

    Notation matches the thesis throughout:
      v0, v1      — KDE bandwidths for the negative (B_0) and positive (B_1) class
      tau (τ)     — operating threshold, found by inverting F̂_0
      alpha (α)   — target CDF value passed to _icdf (= 1 − FPR for the FPR objective)
      scores_pos  — s^(1), positive-class model scores in the current batch
      scores_neg  — s^(0), negative-class model scores in the current batch
    """

    # ── Sigmoid kernel primitives ─────────────────────────────────────────────

    @staticmethod
    def _sigma(v, x):
        """σ(x; v) = 1/(1+exp(−vx))  — kernel CDF  [eq:sigmoid-kernel]."""
        return 1 / (1 + torch.exp(-v * x))

    @staticmethod
    def _sigma_prime(v, x):
        """σ'(x; v) = v·exp(−v|x|)/(1+exp(−v|x|))²  — kernel PDF  [eq:sigmoid-kernel].

        |x| avoids overflow: exp(−v|x|)→0 for large |x|, whereas exp(−vx)→∞
        for large negative x.  σ' is even so the two forms are mathematically
        identical  [sec:roll-kde-sigmoid].
        """
        x = torch.abs(x)
        return (v * torch.exp(-v * x)) / ((1 + torch.exp(-v * x)) ** 2)

    # ── KDE CDF and ICDF ──────────────────────────────────────────────────────

    @staticmethod
    def _cdf(v, tau, scores):
        """F̂(τ; scores) = (1/n) Σ_j σ(τ−s_j; v)  [eq:kde-sigmoid-cdf]."""
        return torch.mean(KernelizedROLLoss._sigma(v, tau - scores))

    @staticmethod
    def _icdf(v, scores, alpha):
        """Find τ s.t. F̂(τ; scores) = alpha via Newton-Raphson.

        Update rule: τ ← τ + (alpha − F̂(τ)) / F̂'(τ)  [eq:kde-sigmoid-pdf]
        Initial guess: empirical alpha-quantile of scores.
        Terminates when |F̂(τ) − alpha| < 1e-4  [sec:roll-kde-forward].
        """
        if v > 10000.:
            logging.warning(f'Bandwidth v={v:.1f} exceeds maximum 10000, capping')
            v = 10000.

        tau = torch.quantile(scores, alpha)
        MAX_ITERS = 10000
        diff = torch.tensor(float('inf'))
        for _ in range(MAX_ITERS):
            F_tau = KernelizedROLLoss._cdf(v, tau, scores)
            diff  = alpha - F_tau
            if torch.isnan(diff):
                logging.debug(f'NaN in Newton-Raphson (F_tau={F_tau})')
                break
            if torch.abs(diff) < 1e-4:
                break
            # F̂'(τ) = (1/n) Σ_j σ'(τ−s_j; v)  [eq:kde-sigmoid-pdf]
            F_prime = torch.mean(KernelizedROLLoss._sigma_prime(v, tau - scores))
            if torch.isnan(F_prime):
                logging.debug('NaN in Newton-Raphson F_prime')
                break
            tau = tau + diff / F_prime
        else:
            logging.warning(f'Newton-Raphson did not converge in {MAX_ITERS} iters, '
                            f'residual={diff:.2e}, v={v:.4f}')
        return tau

    # ── Gradient helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _grad_scores_pos(v1, tau, scores_pos):
        """∂F̂_1/∂f_θ(x_i) = −(1/|B_1|) σ'_1(τ−x_i)  [eq:kde-grad-y1].

        Negative: raising a positive-class score pushes it above τ and
        decreases F̂_1(τ) = P(pos-score ≤ τ), which is what we optimise.
        """
        n1 = scores_pos.shape[0]
        return -(1 / n1) * KernelizedROLLoss._sigma_prime(v1, tau - scores_pos)

    @staticmethod
    def _grad_tau_scores_neg(v0, tau, scores_neg):
        """∂τ/∂f_θ(x_i) = φ_0(τ−x_i) / Σ_k φ_0(τ−x_k)  [eq:kde-grad-tau-efficient].

        Uses the sigmoid identity σ(u)(1−σ(u)) = σ'(u;v)/v  [eq:sigmoid-identity]
        so that v_0 cancels in the ratio, avoiding a redundant exponential.
        """
        s = KernelizedROLLoss._sigma(v0, tau - scores_neg)
        kernel = s * (1 - s)          # ∝ σ'(τ−x_i; v0), v0 cancels in ratio
        return kernel / torch.sum(kernel)

    # ── Forward pass ──────────────────────────────────────────────────────────

    @staticmethod
    def forward(ctx, alpha, yh, y, gamma=1.0):
        # Normalise scores by their std for numerical stability.
        # The chain rule correction is applied in backward().
        score_scale = 1 / torch.sqrt(torch.var(yh, correction=1))
        yh_norm = yh * score_scale

        pos_idx, neg_idx = split_true_false_indeces(yh, y)
        scores_pos = yh_norm[pos_idx]     # s^(1)
        scores_neg = yh_norm[neg_idx]     # s^(0)

        # Bandwidths via Improved Sheather-Jones.
        # gamma > 1 widens kernels early in training  [sec:kde-bandwidth-scaling].
        v0 = torch.minimum(
            torch.tensor(1 / improved_sheather_jones(
                scores_neg.numpy().astype(np.float64)[:, np.newaxis])),
            torch.tensor(10000.)) / gamma
        v1 = torch.minimum(
            torch.tensor(1 / improved_sheather_jones(
                scores_pos.numpy().astype(np.float64)[:, np.newaxis])),
            torch.tensor(10000.)) / gamma

        # τ = F̂_0^{−1}(alpha): threshold where the negative-class CDF equals alpha.
        # Callers pass alpha = 1 − FPR so that P(neg-score > τ) = FPR  [eq:roll-tpr-at-fpr].
        tau  = KernelizedROLLoss._icdf(v0, scores_neg, alpha)
        loss = KernelizedROLLoss._cdf(v1, tau, scores_pos)

        logging.debug(f'KDE-ROLL  gamma={gamma:.4f}  v0={v0:.4f}  v1={v1:.4f}  '
                      f'τ={tau:.4f}  loss={loss:.4f}')
        ctx.save_for_backward(y, yh, scores_pos, scores_neg,
                              pos_idx, neg_idx, tau, loss, v1, v0, score_scale)
        return loss

    # ── Backward pass ─────────────────────────────────────────────────────────

    @staticmethod
    def backward(ctx, grad_output):
        y, yh, scores_pos, scores_neg, pos_idx, neg_idx, \
            tau, loss, v1, v0, score_scale = ctx.saved_tensors

        # ∂L/∂f_θ(x_i) for positive class  [eq:kde-grad-y1]
        grad_pos = KernelizedROLLoss._grad_scores_pos(v1, tau, scores_pos)

        # ∂F̂_1/∂τ = (1/|B_1|) Σ_j σ'_1(τ−x_j)  [eq:kde-dF1-dtau]
        dF1_dtau = -torch.sum(grad_pos)

        # ∂L/∂f_θ(x_i) for negative class  [eq:kde-grad-combined, y_i=0]
        grad_neg = dF1_dtau * KernelizedROLLoss._grad_tau_scores_neg(v0, tau, scores_neg)

        grad_input = torch.zeros(yh.shape[0])
        grad_input[pos_idx] = grad_pos * grad_output
        grad_input[neg_idx] = grad_neg * grad_output

        if torch.any(torch.isnan(grad_input)):
            logging.error('NaN gradient detected in KDE-ROLL backward')

        # Chain-rule correction for score normalisation:
        # yh_norm = yh * score_scale  ⟹  ∂L/∂yh = ∂L/∂yh_norm · score_scale
        return None, grad_input * score_scale, None, None


# ── Public constructors ───────────────────────────────────────────────────────

def _kernelized_roll_fpr_internal(yh, y, fpr, gamma=1.0):
    return KernelizedROLLoss.apply(fpr, yh, y, gamma)

def kernelized_roll_fpr(fpr):
    # Pass 1−fpr as the CDF target: F̂_0(τ) = 1−fpr ⟺ P(neg-score > τ) = fpr = FPR
    return partial(_kernelized_roll_fpr_internal, fpr=1 - fpr)

def kernelized_roll_tpr(tpr):
    # Reduce TPR@FPR to FPR@TPR by negating scores and swapping labels  [sec:roll-formulation]
    def _wrapper(yh, y, gamma=1.0):
        return KernelizedROLLoss.apply(tpr, 1 - yh, 1 - y, gamma)
    return _wrapper
