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

plt.figure(figsize=(10, 6))
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





# Stability of quantiles estimates

class RunningPercentile:
    """From https://stats.stackexchange.com/questions/103258/online-estimation-of-quartiles-without-storing-observations
    """
    def __init__(self, percentile=0.5, step=0.1):
        self.step = step
        self.step_up = 1.0 - percentile
        self.step_down = percentile
        self.x = np.nan

    def push(self, observation):
        if self.x is np.nan:
            self.x = observation
            return

        if self.x > observation:
            self.x -= self.step * self.step_up
        elif self.x < observation:
            self.x += self.step * self.step_down
        if abs(observation - self.x) < self.step:
            self.step /= 2.0
        return self.x


class ReservoirSampling:
    """
    Implemented from the three line description in: https: // www.cs.utexas.edu / ~ecprice / courses / sublinear / scribe / lec6.pdf
    We maintain a random sample S with |S| = m.
    Whenever a new element arrives, we include it with probability min(1,mi). Adding an element to S
    would make it larger than m, we eject a random element from S initially. This results in a uniform
    sample of size m over our stream
    """
    def __init__(self, m):
        self.m = m
        self.S = []
        self.i = 0
    def observe(self, x):
        if len(self.S) < self.m:
            self.S.append(x)
        else:
            if np.random.rand() < self.m / self.i:
                self.S[np.random.randint(self.m)] = x
        self.i += 1


from scipy.stats import pareto
rp = RunningPercentile(percentile=0.9, step=0.3)
q_running_est = np.array([rp.push(x) for x in randn])
q_95_normal = np.array([np.quantile(randn[:i+1], 0.95) for i in range(len(randn))])
q_95_pareto = np.array([np.quantile(r[:i+1], 0.95) for i in range(len(r))])

from tdigest import TDigest
from lmoments3 import distr
digest_r = TDigest()
digest_p = TDigest()
q_95_pareto_td, q_95_normal_td = np.zeros(len(randn)), np.zeros(len(r))
q_95_pareto_lm = np.zeros(int(len(randn)/50))
for i, r_i in enumerate(randn[:1000]):
    digest_r.update(r_i)
    q_95_normal_td[i] = digest_r.percentile(95)
    digest_p.update(r[i])
    q_95_pareto_td[i] = digest_p.percentile(95)
    if i%50 == 0:
        q_95_pareto_lm[i//50] = distr.gpa.ppf(0.95, *distr.gpa.fit(r[:i+1]))

q_50_normal = np.array([np.quantile(randn[:i+1], 0.5) for i in range(len(randn))])
q_50_pareto = np.array([np.quantile(r[:i+1], 0.5) for i in range(len(r))])

print('q95 pareto: {}'.format(np.quantile(r, 0.95)))


for i in np.linspace(100, len(r), 4).astype(int):
    plt.plot(np.sort(r[:i]), np.arange(i)/i)
    plt.semilogx()


plt.figure(figsize=(10, 6))
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
l1 = plt.plot(q_95_pareto[:1000])
plt.plot(q_95_pareto_td[:1000])
plt.plot(np.arange(0, 7000, 50)[:20], q_95_pareto_lm[:20])

plt.plot(q_50_pareto[:1000], color=l1[0]._color, linestyle='--')
plt.plot(q_50_normal[:1000], color=l2[0]._color, linestyle='--')


plt.ylabel(r'sample mean $\frac{1}{N}\sum x_i$')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('TimeSeriesForecasting/figs/pareto_90q.png', dpi=200)