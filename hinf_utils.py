import numpy as np
from scipy.signal import convolve
from numpy.random import randn, rand
from slycot.analysis import ab13dd
import pickle
import pywren


def FIR2ss(g):
    # FIR column vector [g0; ... ; g_{r-1}] to ss form
    r = len(g)
    return (np.diag(np.ones(r-2), -1),
            np.eye(r-1, 1),
            g[1:, np.newaxis].T,
            g[0])


def hinfnorm_fir(g, tol=1e-8):
    # returns the hinfnorm of an FIR filter g
    A, B, C, D = FIR2ss(g)
    gpeak, fpeak = ab13dd('D', 'I', 'S', 'D',
                          len(g)-1, 1, 1,
                          A, np.eye(len(g)-1), B, C, D, tol)
    return gpeak, fpeak


def random_filter(r, normalize=False):
    # Returns a random FIR filter of length r
    g = rand(r)*2-1
    if normalize:
        return g/hinfnorm_fir(g)
    else:
        return g


def make_simulator(g, sigma):
    def fn(u, n, r):
        ys = np.zeros((n, r))
        # slow?
        if len(u.shape) == 1:
            # just broadcast
            ys = convolve(g, u)[:r] + randn(n, r)*sigma
        else:
            # slow? is instantiating toep(g) better?
            for k in range(n):
                ys[k, :] = convolve(g, u[k, :])[:r] + randn(1, r)*sigma
        return ys
    return fn


def pytry(fn, arg, DEBUG=False):
    # Script to attempt a pywren job
    tries = 10
    results = []
    while tries > 0:
        try:
            pwex = pywren.default_executor()
            futures = pwex.map(fn, arg)
            dones, not_dones = pywren.wait(futures, pywren.ALL_COMPLETED)
            results = [f.result() for f in dones]
        except Exception as e:
            raise
            if DEBUG:
                pickle.dump({"e": e, "futures": futures},
                            open("debug.pickle", 'wb'))
                print('Pickle')
                return None
            print('oops')
            tries -= 1
        else:
            print('OK')
            return results
    print('NOT OK')
    return results
