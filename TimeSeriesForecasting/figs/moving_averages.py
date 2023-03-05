import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pd.read_pickle('TimeSeriesForecasting/data/power_ts.zip')
y = data.iloc[:96*21].values

centered_mav = lambda x, k: np.hstack([np.mean(x[t-int(k/2):t+int(k/2)]) for t in range(len(x)-int(k/2))])
lagged_mav = lambda x, k: np.hstack([np.mean(x[np.maximum(0, t-k):t]) for t in range(len(x))])


k=96*2
plt.plot(y)
plt.plot(centered_mav(y, k))
plt.plot(lagged_mav(y, k))

figsize = (10, 4)
dpi = 150
fig, ax = plt.subplots(1, 1, figsize=figsize)

c_mav = centered_mav(y, k)
l_mav = lagged_mav(y, k)


kb = 0.02
for i in np.arange(244, len(y)):
    ax.cla()
    plt.plot(y, linewidth=1.45)
    l1 = ax.plot(c_mav[:i])
    l2 = ax.plot(l_mav[:i])
    ylims = ax.get_ylim()
    c_relpos = (c_mav[i]-ylims[0])/(ylims[1]-ylims[0])
    l_relpos = (l_mav[i]-ylims[0])/(ylims[1]-ylims[0])
    ax.axvspan(i-int(k/2), i+int(k/2), ymin=c_relpos-kb, ymax=c_relpos+kb, alpha=0.3, color=l1[0]._color)
    ax.axvspan(i-k, i, ymin=l_relpos-kb, ymax=l_relpos+kb, alpha=0.3, color=l2[0]._color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.pause(0.001)
    plt.savefig('TimeSeriesForecasting/figs/moving_avg/snap_{:04d}.png'.format(i), dpi=dpi)
