import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

figsize = (5, 4)
dpi = 150
n = 200
n_max = 10
n_frames = 1000


def animate(c, process_fun, name, ymin, ymax):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x_pool, alpha_pool = [], []
    for i in range(n_frames):
        draw_realization = True if i % 30 == 0 else False
        if draw_realization:
            x = process_fun(n)
            x_pool.append(x)
            alpha_pool.append(1)
            if len(x_pool) > n_max:
                x_pool = x_pool[1:]
                alpha_pool = alpha_pool[1:]
        ax.cla()
        for j, x in enumerate(x_pool):
            ax.plot(x, alpha=alpha_pool[j], color=c)
            alpha_pool[j] *= 0.95
        plt.ylim(ymin, ymax)
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('t')
        plt.savefig('TimeSeriesForecasting/figs/random_processes/{}_{:04d}'.format(name, i), dpi=dpi)


process_fun = lambda n: np.random.randn(n)
c = plt.get_cmap('plasma', 10)(3)
animate(c, process_fun, 'randn_iid', -30, 30)


process_fun = lambda n: np.cumsum(np.random.randn(n))
c = plt.get_cmap('plasma', 10)(5)
animate(c, process_fun, 'rand_walk', -50, 50)



# Fat tail expectation
from scipy.stats import pareto
N = 7000
randn = np.random.randn(N)
a = 1.12
rv = pareto(a)
r = pareto.rvs(a, size=N)
plt.subplot(211)
plt.hist(r, 800, density=True, alpha=0.5)
plt.hist(randn, 25, density=True, alpha=0.5)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.ylabel('pdf, log scale')
plt.yscale('log')
plt.xlim(-4, 200)
plt.subplot(212)
plt.plot((np.cumsum(r)/np.arange(N))[:1000])
plt.plot((np.cumsum(randn)/np.arange(N))[:1000])
plt.ylabel(r'sample mean $\frac{1}{N}\sum x_i$')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('TimeSeriesForecasting/figs/pareto.png', dpi=200)
