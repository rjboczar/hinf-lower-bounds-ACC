import numpy as np
from numpy.random import standard_normal as std_normal
from numpy.random import randn
from numpy.linalg import norm
from numpy.fft import rfft, irfft

from hinf_utils import hinfnorm_fir


# Plug-in estimator
def plugin_est(fn=None, N=1000, r=-1, power=1.0, history=False):
    # Estimates from N trajectories, with a per-trajectory power constraint(?)
    # fn(u,n) should return trajectories of y = Gu + noise
    u = np.zeros(r)
    u[0] = power
    ys = fn(u, N, r)
    if not history:
        g_ls = np.mean(ys, axis=0) / power
        return hinfnorm_fir(g_ls)[0]
    else:
        # Broadcasting to get running average
        g_ls = np.cumsum(ys, axis=0) / (power * np.arange(1, N+1)[:, None])
        return np.array([hinfnorm_fir(g_ls[k, :])[0] for k in range(N)])


# Power iteration estimator c.f. Rojas
def power_est(fn=None, N=1000, r=-1, power=1.0, history=False, smooth=1):
    # fn(u,n) should return trajectories of y = Gu + noise
    u = randn(1, r)
    old_u_mu = u
    estimate = np.zeros(N)
    for k in range(N):
        u = u / norm(u) * power
        y = fn(u, 1, r)
        ytilde = np.fliplr(y)
        mu = norm(ytilde) / power  # some gain
        dp = old_u_mu.dot(ytilde.T)[0, 0]
        estimate[k] = np.sqrt(np.clip(dp, 0, None))/power
        old_u_mu = mu * u
        u = ytilde / mu
    return estimate if history else np.mean(estimate[-smooth:])


# Power iteration c.f. Wahlberg
def wahlberg_est(fn=None, N=1000, r=-1, power=1.0, history=False, smooth=1):
    # fn(u,n) should return trajectories of y = Gu + noise
    u = randn(1, r)
    u = u/norm(u)*power
    estimate = np.zeros(N)
    # Requires 2 exps
    for k in range(N//2):
        y = fn(u, 1, r)
        ytilde = np.fliplr(y)
        z = fn(ytilde, 1, r)
        ztilde = np.fliplr(z)
        dp = np.abs(u.dot(ztilde.T))[0, 0]
        estimate[2*k:2*k+2] = np.sqrt(dp) / power
        u = power*ztilde/np.linalg.norm(ztilde)
    return estimate if history else np.mean(estimate[-smooth:])


def wts_est(fn=None, N=1000, r=-1, power=np.sqrt(2),
            history=False, M=2000, sigma=0.1, lam2=1, DEBUG=False):
    # Weighted thompson sampling estimator
    K = r  # because of DFT
    p = np.ones(K)/K  # Power profile, initialized at uniform
    p_sum = np.zeros(K)  # Running sum of power
    Xp_sum = np.zeros(K)  # Running sum of Xp product
    post_mean = np.zeros(K)  # Complex in general
    post_var = np.zeros(K)
    Xs = np.zeros(K)
    beta = np.zeros(N)

    eps = 1e-15
    sigma2 = sigma**2

    for t in range(N):
        # Perform experiment (assume no DC or Nyquist power, per footnote)
        uF = np.concatenate(
            (np.array([0.0]), np.sqrt(p), (np.array([0.0]))))
        # sqrt(2) bc of negative freqs
        uF = uF * power / np.sqrt(2)
        u = irfft(uF, norm="ortho")
        y = fn(u, 1, 2*K+2)
        # Assumes uF has power everywhere for no Nans
        # Make sure we leave out 0 and pi
        Xs = (rfft(y, norm="ortho").ravel()[1:-1]) / uF[1:-1]

        # Update posterior of mu
        p_sum += p
        Xp_sum = Xp_sum + Xs * p  # pointwise product
        post_mean = lam2 * Xp_sum / (sigma2 + lam2 * p_sum)
        post_var = lam2/(1+lam2/sigma2 * p_sum)

        # Update posterior of rho:
        # Get samples
        post_cov = np.sqrt(post_var)
        s_real = (std_normal(size=(M, K)) * post_cov) + post_mean.real
        s_imag = (std_normal(size=(M, K)) * post_cov) + post_mean.imag
        s = s_real**2 + s_imag**2

        # Hinfinity estimate
        beta[t] = np.max(np.abs(Xp_sum)/p_sum)

        # Update power profile
        # eps ensures dividing by uF will give no NaNs
        p = np.bincount(np.argmax(s, axis=1), minlength=K)/M + eps
        p = p/np.sum(p)

    return beta if history else beta[-1]

