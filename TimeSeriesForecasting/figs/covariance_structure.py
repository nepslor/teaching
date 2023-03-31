import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
import seaborn as sb
from scipy.interpolate import interp1d
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, ShrunkCovariance
from TimeSeriesForecasting.utils.plot_utils import SeabornFig2Grid

estimation = 'shrunk'
N = 2000
n_scen = 10000


def get_pdf(x, q_vect=np.linspace(0, 1, 100)):
    return np.quantile(x, q_vect, axis=0)


def joinplot(x, q_vect=np.linspace(0, 1, 100)):
    x1, x2 = x.T
    pdfs = get_pdf(x, q_vect)
    g1 = sb.jointplot(x=x1.ravel(), y=x2.ravel(), marker=".", s=10,
                 marginal_kws=dict(bins=25, fill=False))

    g1.ax_marg_x.plot(pdfs[:, 0], q_vect*N/20)
    g1.ax_marg_y.plot(q_vect*N/20, pdfs[:, 1])
    return g1


def multivariate_normal_copula(x):
    # estimate normalized covariance of the original signals
    est_cov = np.corrcoef(x.T)

    # sample from the multivariate normal distribution with the same covariance
    mvr = multivariate_normal([0, 0], est_cov)
    mvr_samples = mvr.rvs((n_scen))

    # retrieve the copula samples (multivariate normal CDF)
    return norm.cdf(mvr_samples)


def sample_scenarios(n_scen, pdfs, c, q_vect=np.linspace(0, 1, 100)):
    scenarios = np.zeros((n_scen, 2))
    for i, pdf in enumerate(pdfs.T):
        q_interpfun_at_step = interp1d(q_vect, pdf, fill_value='extrapolate')
        scenarios[:, i] = q_interpfun_at_step(c[:, i])
    return scenarios


def sample_from_empirical_copula(x, n_scens, q_vect=np.linspace(0, 1, 100)):
    x_struct = x[np.random.choice(len(x), n_scens), :]
    pdfs = get_pdf(x, q_vect)
    u = np.random.rand(n_scens, 2)
    independent_samples = sample_scenarios(n_scens, pdfs, u, q_vect)
    sorted_indep_samples = np.vstack([np.sort(s) for s in independent_samples.T]).T
    ordered_stats = np.vstack([x_i.argsort().argsort() for x_i in x_struct.T]).T
    samples = np.zeros((n_scens, 2))
    for i in range(x.shape[1]):
        samples[:, i] = sorted_indep_samples[ordered_stats[:, i], i]
    return samples


# generate correlated signals
z = np.random.randn(int(N), 1) + 1.6
x1 = z
x2 = np.log(z+10) + 0.1*np.random.randn(N, 1)
x = np.hstack([x1, x2])

c = multivariate_normal_copula(x)
pdfs = get_pdf(x)
scens = sample_scenarios(n_scen, pdfs, c)

g1 = joinplot(x)
g2 = joinplot(c)
g3 = joinplot(scens)
SeabornFig2Grid([g1, g2, g3], ['observations', 'copula', 'sampled, Gaussian-copula'], figsize=(15, 5))
plt.savefig('TimeSeriesForecasting/figs/covariance_structure_unimodal.png', dpi=300)

z = np.vstack([np.random.randn(int(N/2), 1) + 1.6,  np.random.randn(int(N/2), 1)*0.8 + 6])
x1 = z
x2 = np.log(z+10)+ 0.1*np.random.randn(N, 1)
x = np.hstack([x1, x2])

c = multivariate_normal_copula(x)
pdfs = get_pdf(x)
scens = sample_scenarios(n_scen, pdfs, c)

g1 = joinplot(x)
g2 = joinplot(c)
g3 = joinplot(scens)
SeabornFig2Grid([g1, g2, g3], ['observations', 'copula', 'sampled, Gaussian-copula'],  figsize=(15, 5))
plt.savefig('TimeSeriesForecasting/figs/covariance_structure_bimodal.png', dpi=300)


g1 = joinplot(x)
g2 = joinplot(sample_scenarios(n_scen, pdfs, c))
g3 = joinplot(sample_from_empirical_copula(x, n_scen))
SeabornFig2Grid([g1, g2, g3], ['observations', 'sampled, Gaussian-copula', 'sampled, empirical-copula'], figsize=(15, 5))
plt.savefig('TimeSeriesForecasting/figs/covariance_structure_bimodal_empirical_vs_gaussian.png', dpi=300)
