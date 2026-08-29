import torch
import logging
from functools import partial
from torch.autograd import Function
# from .beta_dist import Beta
import numpy as np
from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones


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


# ── KDE bandwidth selection ───────────────────────────────────────────────────

def _bandwidth(scores_1d: np.ndarray, max_v: float = 10000.0) -> float:
    """Return 1/h (kernel parameter v), clipped to max_v.

    Tries ISJ; falls back to Silverman when ISJ fails to converge (common on
    small or near-constant batches early in training).
    """
    data = scores_1d[:, np.newaxis]
    try:
        h = float(improved_sheather_jones(data))
    except Exception:
        h = float(silvermans_rule(data))
    return min(1.0 / h, max_v)


# ── KDE ROLL ──────────────────────────────────────────────────────────────────

# FPR grid for AOC: 0.01, 0.02, ..., 0.99  (99 values).  Sum divided by 100
# approximates ∫₀¹ (1−TPR(α)) dα = AOC = 1−AUC  [sec:roll-aoc].
_AOC_FPRS    = np.arange(0.01, 1.0, 0.01).tolist()
_AOC_ALPHAS  = [1.0 - fpr for fpr in _AOC_FPRS]
_AOC_DIVISOR = 100


class KernelizedROLLoss(Function):
    """KDE-ROLL loss with custom backward pass.

    Accepts a vector of alpha values (CDF targets, α_k = 1 − FPR_k) and a
    divisor, so one class serves both objectives:
      point-FPR: alphas=[1−fpr], divisor=1
      AOC:       alphas=_AOC_ALPHAS, divisor=_AOC_DIVISOR

    Notation matches the thesis throughout:
      v0, v1      — KDE bandwidths for the negative (B_0) and positive (B_1) class
      taus (τ_k)  — per-alpha thresholds, found by inverting F̂_0
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
    def _icdf(v, scores, alpha, tol=1e-4, max_iters=60):
        """Find τ s.t. F̂(τ; scores) = alpha via bisection.

        Bracket: [min(scores) − 10/v, max(scores) + 10/v] always satisfies
        F̂(lo) < alpha < F̂(hi) since the sigmoid tails decay within ~10/v of
        the data range.  Bisection converges in ≤ log2(range/tol) ≈ 16 steps.
        """
        if v > 10000.:
            logging.warning(f'Bandwidth v={v:.1f} exceeds maximum 10000, capping')
            v = 10000.

        margin = 10.0 / v
        lo = scores.min() - margin
        hi = scores.max() + margin

        for _ in range(max_iters):
            mid = (lo + hi) / 2
            F_mid = KernelizedROLLoss._cdf(v, mid, scores)
            if torch.abs(F_mid - alpha) < tol:
                break
            if F_mid < alpha:
                lo = mid
            else:
                hi = mid

        return mid

    @staticmethod
    def _icdf_batch(v, scores, alphas, tol=1e-4, max_iters=60):
        """Vectorized bisection: find τ_k s.t. F̂(τ_k) = α_k for all K alphas at once.

        Each iteration evaluates F̂ at all K midpoints via a single (K, n) matmul,
        replacing K sequential _icdf calls.  Converges in ≤ 16 steps for normalised
        scores; 60 is a generous cap.
        """
        if float(v) > 10000.:
            logging.warning(f'Bandwidth v={float(v):.1f} exceeds maximum 10000, capping')
            v = v.clamp(max=10000.) if isinstance(v, torch.Tensor) else 10000.

        K = len(alphas)
        alphas_t = torch.tensor(alphas, dtype=scores.dtype, device=scores.device)   # (K,)
        margin = 10.0 / float(v)
        lo = torch.full((K,), float(scores.min()) - margin, device=scores.device)
        hi = torch.full((K,), float(scores.max()) + margin, device=scores.device)
        mid = (lo + hi) / 2

        for _ in range(max_iters):
            diff = mid.unsqueeze(1) - scores.unsqueeze(0)                        # (K, n)
            F_mid = torch.mean(KernelizedROLLoss._sigma(v, diff), dim=1)         # (K,)
            if torch.all(torch.abs(F_mid - alphas_t) < tol):
                break
            too_low = F_mid < alphas_t
            lo = torch.where(too_low, mid, lo)
            hi = torch.where(too_low, hi, mid)
            mid = (lo + hi) / 2

        return mid  # (K,)

    # ── Forward pass ──────────────────────────────────────────────────────────

    @staticmethod
    def forward(ctx, alphas, divisor, yh, y, gamma=1.0, normalize=True):
        # Normalise scores by their std for numerical stability.
        # The chain rule correction is applied in backward().
        # clamp avoids inf when all batch scores are identical (zero variance)
        if normalize:
            score_scale = 1 / torch.sqrt(torch.var(yh, correction=1).clamp(min=1e-8))
        else:
            score_scale = torch.tensor(1.0, device=yh.device)
        yh_norm = yh * score_scale

        pos_idx, neg_idx = split_true_false_indeces(yh, y)
        scores_pos = yh_norm[pos_idx]     # s^(1)
        scores_neg = yh_norm[neg_idx]     # s^(0)

        # Bandwidths via ISJ; fall back to Silverman on convergence failure.
        # gamma > 1 widens kernels early in training  [sec:kde-bandwidth-scaling].
        v0 = (torch.tensor(_bandwidth(scores_neg.cpu().numpy().astype(np.float64))) / gamma).to(yh.device)
        v1 = (torch.tensor(_bandwidth(scores_pos.cpu().numpy().astype(np.float64))) / gamma).to(yh.device)

        # τ_k = F̂_0^{−1}(alpha_k): threshold where the negative-class CDF equals alpha_k.
        # Callers pass alpha_k = 1 − FPR_k so that P(neg-score > τ_k) = FPR_k  [eq:roll-tpr-at-fpr].
        if len(alphas) == 1:
            taus = KernelizedROLLoss._icdf(v0, scores_neg, alphas[0]).unsqueeze(0)
        else:
            taus = KernelizedROLLoss._icdf_batch(v0, scores_neg, alphas)  # (K,)

        # loss_k = F̂_1(tau_k) = 1 − TPR@FPR_k  — vectorised over K
        diff_pos = taus.unsqueeze(1) - scores_pos.unsqueeze(0)              # (K, n1)
        losses = torch.mean(KernelizedROLLoss._sigma(v1, diff_pos), dim=1)  # (K,)

        total_loss = losses.sum() / divisor

        logging.debug(f'KDE-ROLL  K={len(alphas)}  gamma={gamma:.4f}  v0={v0:.4f}  v1={v1:.4f}  '
                      f'loss={total_loss:.4f}')
        ctx.save_for_backward(yh, scores_pos, scores_neg,
                              pos_idx, neg_idx, taus, v1, v0, score_scale)
        ctx.divisor = divisor
        return total_loss

    # ── Backward pass ─────────────────────────────────────────────────────────

    @staticmethod
    def backward(ctx, grad_output):
        yh, scores_pos, scores_neg, pos_idx, neg_idx, \
            taus, v1, v0, score_scale = ctx.saved_tensors
        divisor = ctx.divisor

        n1 = scores_pos.shape[0]

        # diff_pos[k, i] = tau_k − s_i^(1)   shape (K, n1)
        diff_pos = taus.unsqueeze(1) - scores_pos.unsqueeze(0)
        sp_pos = KernelizedROLLoss._sigma_prime(v1, diff_pos)  # (K, n1)

        # ∂(Σ_k loss_k)/∂s_i^(1) = −(1/n1) Σ_k σ'(tau_k − s_i)  [eq:kde-grad-y1]
        grad_pos = -(1.0 / n1) * sp_pos.sum(dim=0) / divisor  # (n1,)

        # ∂F̂_1(tau_k)/∂tau_k = (1/n1) Σ_j σ'(tau_k − s_j^(1))  [eq:kde-dF1-dtau]
        dF1_dtau = sp_pos.mean(dim=1)  # (K,)

        # ∂tau_k/∂s_i^(0)  [eq:kde-grad-tau-efficient]
        diff_neg = taus.unsqueeze(1) - scores_neg.unsqueeze(0)  # (K, n0)
        s_neg = KernelizedROLLoss._sigma(v0, diff_neg)           # (K, n0)
        kernel = s_neg * (1 - s_neg)                             # (K, n0), v0 cancels in ratio
        grad_tau_neg = kernel / kernel.sum(dim=1, keepdim=True).clamp(min=1e-10)  # (K, n0)

        # ∂(Σ_k loss_k)/∂s_i^(0) = Σ_k dF1_dtau_k · ∂tau_k/∂s_i^(0)  [eq:kde-grad-combined]
        grad_neg = (dF1_dtau.unsqueeze(1) * grad_tau_neg).sum(dim=0) / divisor  # (n0,)

        grad_input = torch.zeros(yh.shape[0], device=yh.device)
        grad_input[pos_idx] = grad_pos * grad_output
        grad_input[neg_idx] = grad_neg * grad_output

        if torch.any(torch.isnan(grad_input)):
            logging.error('NaN gradient detected in KDE-ROLL backward')

        # Chain-rule correction for score normalisation:
        # yh_norm = yh * score_scale  ⟹  ∂L/∂yh = ∂L/∂yh_norm · score_scale
        return None, None, grad_input * score_scale, None, None, None


# ── Public constructors ───────────────────────────────────────────────────────

def _kernelized_roll_fpr_internal(yh, y, fpr, gamma=1.0, normalize=True,
                                   bce_weight=0.0, bce_pos_weight=1.0):
    roll = KernelizedROLLoss.apply([fpr], 1, yh, y, gamma, normalize)
    if bce_weight == 0.0:
        return roll
    pos_weight = torch.tensor([bce_pos_weight], dtype=yh.dtype, device=yh.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        yh, y.float(), pos_weight=pos_weight)
    return roll + bce_weight * bce

def kernelized_roll_fpr(fpr, normalize=True, bce_weight=0.0, bce_pos_weight=1.0):
    # Pass 1−fpr as the CDF target: F̂_0(τ) = 1−fpr ⟺ P(neg-score > τ) = fpr = FPR
    return partial(_kernelized_roll_fpr_internal, fpr=1 - fpr, normalize=normalize,
                   bce_weight=bce_weight, bce_pos_weight=bce_pos_weight)

def _kernelized_roll_tpr_internal(yh, y, tpr, gamma=1.0, bce_weight=0.0, bce_pos_weight=1.0):
    # Reduce TPR@FPR to FPR@TPR by negating scores and swapping labels  [sec:roll-formulation]
    roll = KernelizedROLLoss.apply([tpr], 1, 1 - yh, 1 - y, gamma)
    if bce_weight == 0.0:
        return roll
    pos_weight = torch.tensor([bce_pos_weight], dtype=yh.dtype, device=yh.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        yh, y.float(), pos_weight=pos_weight)
    return roll + bce_weight * bce

def kernelized_roll_tpr(tpr, bce_weight=0.0, bce_pos_weight=1.0):
    return partial(_kernelized_roll_tpr_internal,
                   tpr=tpr, bce_weight=bce_weight, bce_pos_weight=bce_pos_weight)

def _kernelized_roll_aoc_internal(yh, y, gamma=1.0, normalize=True,
                                   bce_weight=0.0, bce_pos_weight=1.0):
    roll = KernelizedROLLoss.apply(_AOC_ALPHAS, _AOC_DIVISOR, yh, y, gamma, normalize)
    if bce_weight == 0.0:
        return roll
    pos_weight = torch.tensor([bce_pos_weight], dtype=yh.dtype, device=yh.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        yh, y.float(), pos_weight=pos_weight)
    return roll + bce_weight * bce

def kernelized_roll_aoc(normalize=True, bce_weight=0.0, bce_pos_weight=1.0):
    return partial(_kernelized_roll_aoc_internal, normalize=normalize,
                   bce_weight=bce_weight, bce_pos_weight=bce_pos_weight)


# ── LibAUC AUROC loss ─────────────────────────────────────────────────────────

class _PESGFactory:
    """Picklable opt_factory that creates a PESG optimizer bound to an AUCMLoss."""
    def __init__(self, loss_module, lr, weight_decay, epoch_decay, momentum, mode):
        self.loss_module = loss_module
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch_decay = epoch_decay
        self.momentum = momentum
        self.mode = mode

    def __call__(self, model):
        from libauc.optimizers import PESG
        device = next(model.parameters()).device
        return PESG(model.parameters(), self.loss_module, lr=self.lr,
                    weight_decay=self.weight_decay, epoch_decay=self.epoch_decay,
                    momentum=self.momentum, mode=self.mode, verbose=False,
                    device=device)


class libauc_auc_loss:
    """AUCMLoss (Yuan et al., ICCV 2021) adapted to the (yh, y) → scalar convention.

    Applies sigmoid to logits before passing to AUCMLoss, matching LibAUC's
    example usage.  Must be paired with PESG via pesg_opt_factory().
    """
    def __init__(self, margin=1.0, imratio=None, version='v2', device=None):
        from libauc.losses import AUCMLoss
        self._loss = AUCMLoss(margin=margin, imratio=imratio, version=version, device=device)

    def __call__(self, yh, y):
        return self._loss(torch.sigmoid(yh).unsqueeze(1), y.float().unsqueeze(1))

    def pesg_opt_factory(self, lr=0.1, weight_decay=1e-5, epoch_decay=2e-3,
                         momentum=0.9, mode='adam'):
        """Return an opt_factory for ExperimentConfiguration.opt_factory.

        Binds the same AUCMLoss instance used in __call__ so PESG can track
        and update a, b, alpha from that module.
        """
        return _PESGFactory(self._loss, lr, weight_decay, epoch_decay, momentum, mode)


# ── Robust loss functions (noise-tolerant baselines) ──────────────────────────

class mae_loss:
    """Mean Absolute Error loss on sigmoid outputs.

    Satisfies the symmetry condition (Ghosh 2017): provably noise-tolerant
    under class-conditional label noise.
    pos_weight: weight applied to positive-class samples (same convention as
    BCEWithLogitsLoss pos_weight). Set to n_neg/n_pos for imbalanced datasets.
    """
    def __init__(self, pos_weight: float = 1.0):
        self.pos_weight = pos_weight

    def __call__(self, yh, y):
        p = torch.sigmoid(yh)
        loss = torch.abs(y.float() - p)
        if self.pos_weight != 1.0:
            w = torch.where(y.bool(),
                            torch.full_like(loss, self.pos_weight),
                            torch.ones_like(loss))
            return (w * loss).mean()
        return loss.mean()


class gce_loss:
    """Generalized Cross Entropy loss (Zhang & Sabuncu, NeurIPS 2018).

    L_q interpolates between MAE (q→0, noise-robust) and CE (q→1, fast).
    q=0.7 is the value recommended in the original paper.
    pos_weight: weight applied to positive-class samples for class imbalance.
    """
    def __init__(self, q: float = 0.7, pos_weight: float = 1.0):
        self.q = q
        self.pos_weight = pos_weight

    def __call__(self, yh, y):
        p = torch.sigmoid(yh)
        p_true = p * y.float() + (1.0 - p) * (1.0 - y.float())
        p_true = p_true.clamp(min=1e-7)
        loss = (1.0 - p_true.pow(self.q)) / self.q
        if self.pos_weight != 1.0:
            w = torch.where(y.bool(),
                            torch.full_like(loss, self.pos_weight),
                            torch.ones_like(loss))
            return (w * loss).mean()
        return loss.mean()


class focal_loss:
    """Binary focal loss (Lin et al., ICCV 2017).

    Applies a modulating factor (1-p_t)^gamma to down-weight easy examples,
    focusing training on hard / misclassified samples. gamma=2 is the standard.
    pos_weight: alpha class-balancing weight for positive samples (alpha in the
    original paper). Set to n_neg/n_pos for imbalanced datasets.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        self.gamma = gamma
        self.pos_weight = pos_weight

    def __call__(self, yh, y):
        import torch.nn.functional as F
        p = torch.sigmoid(yh)
        bce = F.binary_cross_entropy_with_logits(yh, y.float(), reduction='none')
        p_t = p * y.float() + (1.0 - p) * (1.0 - y.float())
        loss = (1.0 - p_t) ** self.gamma * bce
        if self.pos_weight != 1.0:
            w = torch.where(y.bool(),
                            torch.full_like(loss, self.pos_weight),
                            torch.ones_like(loss))
            return (w * loss).mean()
        return loss.mean()


class asymmetric_loss:
    """Binary asymmetric loss (Ridnik et al., ICCV 2021).

    Applies different focal strengths to positive (gamma_pos) and negative
    (gamma_neg) samples, plus a probability clip that zeroes the gradient for
    negative samples whose predicted positive probability is below `clip`.
    This suppresses high-confidence positive predictions with noisy negative
    labels — suited to one-sided class-conditional label noise.
    Default params: gamma_pos=0, gamma_neg=4, clip=0.05 (from the paper).
    pos_weight: class-balancing weight for positive samples.
    """
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                 clip: float = 0.05, pos_weight: float = 1.0):
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.pos_weight = pos_weight

    def __call__(self, yh, y):
        p = torch.sigmoid(yh)
        y = y.float()

        # shift negative probability down; samples with p < clip contribute zero loss
        p_neg = torch.clamp(p - self.clip, min=0.0)

        loss_pos = -y * torch.log(p.clamp(min=1e-8)) * (1.0 - p) ** self.gamma_pos
        loss_neg = -(1.0 - y) * torch.log((1.0 - p_neg).clamp(min=1e-8)) * p_neg ** self.gamma_neg
        loss = loss_pos + loss_neg
        if self.pos_weight != 1.0:
            w = torch.where(y.bool(),
                            torch.full_like(loss, self.pos_weight),
                            torch.ones_like(loss))
            return (w * loss).mean()
        return loss.mean()
