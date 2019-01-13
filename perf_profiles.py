import numpy as np
import matplotlib.pyplot as plt
# Taken from Ben's blog


def pdist(data, true_hs):
    # Data is num_prob by num_method
    # Distance is difference in relative error from the best method
    rel_err = np.abs(data.T - true_hs)/true_hs
    best_err = np.min(rel_err, axis=0)
    return (rel_err - best_err).T


def perf_prof(data, true_hs, data_names, tau_min=0.1,tau_max=1.0,npts=100):
    # Data is num_prob by num_method
    num_prob, num_method = data.shape
    rho = np.zeros((npts, num_method))

    # This is the d[p,m] function discussed in the blog.
    dist_like_fun = pdist(data, true_hs)

    # Compute the cumulative rates of the distance being less than a fixed threshold
    tau = np.linspace(tau_min, tau_max, npts)
    for method in range(num_method):
        for k in range(npts):
            rho[k, method] = np.sum(dist_like_fun[:, method] < tau[k])/num_prob

    # make plot
    colors = ['#2D328F', '#F15C19', "#81b13c", "#39CCCC", "#B10DC9", "#000000"]
    label_fontsize = 18
    tick_fontsize = 14
    linewidth = 3

    plt.figure(figsize=(7,5))
    for method in range(num_method):
        plt.plot(tau, rho[:, method], color=colors[method],
                 linewidth=linewidth, label=data_names[method])

    plt.xlabel(r'$\tau$', fontsize=label_fontsize)
    plt.ylabel(r'fraction with $\Delta_{rel} \geq \tau$', fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.grid(True)
    return tau, rho
