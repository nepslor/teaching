import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


lagged_mav = lambda x, k: x.rolling(k).mean()

def ts_generator(n_seasons=10, season_0=96, ns_ratio=0.1):
    n_steps = n_seasons * season_0
    t = np.arange(n_steps)
    y = np.sin(np.pi*t/season_0)**2
    rw = ns_ratio * pd.DataFrame(np.cumsum(np.random.randn(n_steps)))
    rw -= lagged_mav(rw, int(season_0/4)).bfill()
    return y+rw.values.ravel()


season_0 = 96
y_0 = ts_generator(16, ns_ratio=0.05, season_0=season_0)
fig, ax = plt.subplots(1, 1, figsize=(15, 4))
plt.plot(np.arange(len(y_0)), y_0)

q_vect = np.linspace(0, 1, 10)

# empirical quantiles over whole dataset
qs = np.tile(np.quantile(y_0, q_vect).reshape(-1, 1),  season_0).T

for i in range(int(len(q_vect)/2)):
    plt.fill_between(len(y_0)+np.arange(season_0), qs[:, i], qs[:, -i-1], color='blue', alpha=0.1)

# empirical quantiles conditioned to time of day
qs_conditional = np.quantile(y_0.reshape(-1,season_0).T, q_vect, axis=1).T

for i in range(int(len(q_vect)/2)):
    plt.fill_between(len(y_0)+np.arange(season_0), qs_conditional[:, i], qs_conditional[:, -i-1], color='red', alpha=0.1)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('TimeSeriesForecasting/figs/reliability/quantiles.png', dpi=200)

def reliability(y, q, q_vect=q_vect, name=''):
    """
    Compute reliability of a time series y for each quantile vector contained in q.
    """
    res = {}
    for alpha, q_i in zip(q_vect, q):
        res[alpha] = np.mean(y < q_i)
    return pd.DataFrame(res, index=['reliability{}'.format('_'+name)]).T


def quantile_score(y, q, q_vect=q_vect, name=''):
    """
    Compute quantile score of a time series y for each quantile vector contained in q.
    """
    res = {}
    for alpha, q_i in zip(q_vect, q):
        err_alpha = q_i - y
        I = (err_alpha > 0).astype(int)
        res[alpha] = np.mean((I - alpha) * err_alpha)
    return pd.DataFrame(res, index=['qscore{}'.format('_'+name)]).T




res, y_sampled = [], []
for i in range(200):
    # generate a possible realization of the time series
    y = ts_generator(17, ns_ratio=0.05, season_0=season_0)[-season_0:]
    y_sampled.append(y)
    rel = reliability(y, qs.T, q_vect=q_vect, name='all')
    rel_cond = reliability(y, qs_conditional.T, q_vect=q_vect, name='cond')
    qscore = quantile_score(y, qs.T, q_vect=q_vect, name='all')
    qscore_cond = quantile_score(y, qs_conditional.T, q_vect=q_vect, name='cond')

    res.append(pd.concat([rel, rel_cond, qscore, qscore_cond], axis=1))
y_sampled = np.vstack(y_sampled)

from functools import reduce
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
plot_steps = np.arange(1, 200)
ax[0].plot(y_0[-season_0*2:], color='blue', alpha=0.5)
for i in plot_steps:
    ax[0].plot(np.arange(season_0*2, season_0*3), y_sampled[i], color='blue', alpha=0.02)
    ax[1].cla()
    ax[2].cla()
    res_i = reduce(lambda x,y: x+y, res[:i]) / i
    ax[1].plot(q_vect, res_i[['reliability_all', 'reliability_cond']].values)
    ax[1].plot(q_vect, q_vect, 'k--')
    ax[2].plot(q_vect, res_i[['qscore_all', 'qscore_cond']].values)
    ax[1].set_title('Reliability')
    ax[2].set_title('Quantile score')
    plt.suptitle('n_samples = {}'.format(i))
    ax[2].legend(['all', 'conditional'])
    # remove upper and left spines
    [a.spines['right'].set_visible(False) for a in ax]
    [a.spines['top'].set_visible(False) for a in ax]
    [a.set_xlabel('alpha quantile [-]') for a in ax[1:]]
    plt.savefig('TimeSeriesForecasting/figs/reliability/snap_{:03d}.png'.format(i), dpi=200)




plt.plot(rel)
plt.plot(rel_cond)
plt.plot(q_vect, q_vect, 'k--')




