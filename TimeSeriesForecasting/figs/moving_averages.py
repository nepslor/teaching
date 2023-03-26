import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pd.read_pickle('TimeSeriesForecasting/data/power_ts.zip')
y = data.iloc[:96*100].values

centered_mav = lambda x, k: np.hstack([np.mean(x[t-int(k/2):t+int(k/2)]) for t in range(len(x)-int(k/2))])
lagged_mav = lambda x, k: np.hstack([np.mean(x[np.maximum(0, t-k):t]) for t in range(len(x))])


k=96
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



figsize = (15, 3.5)
fig, ax = plt.subplots(1, 2, figsize=figsize)
ax[0].plot(y)
l1 = ax[0].plot(c_mav)
l2 = ax[0].plot(l_mav)

ax[0].plot(y[:len(c_mav)]-c_mav, color=l1[0]._color, label='centered MA')
ax[0].plot(y[:len(l_mav)]-l_mav, color=l2[0]._color, label='lagged MA')

ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].legend()


acf = lambda x, k: np.hstack([np.corrcoef(x[:-l-1], x[l+1:])[0,1] for l in range(k)])

max_l = 96*5
ax[1].plot(acf(y, max_l))
ax[1].plot(acf(y[np.argwhere(~np.isnan(c_mav)).ravel()]-c_mav[~np.isnan(c_mav)], max_l))
ax[1].plot(acf(y[np.argwhere(~np.isnan(l_mav)).ravel()]-l_mav[~np.isnan(l_mav)], max_l))
ax[1].hlines(1.96/np.sqrt(len(y)), 0, max_l, linestyles='--')
ax[1].hlines(-1.96/np.sqrt(len(y)), 0, max_l, linestyles='--')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

plt.savefig('TimeSeriesForecasting/figs/moving_avg/ma_comparison.png', dpi=dpi)







def rk4(odes, state, parameters, dt=0.01):
    k1 = dt * odes(state, parameters)
    k2 = dt * odes(state + 0.5 * k1, parameters)
    k3 = dt * odes(state + 0.5 * k2, parameters)
    k4 = dt * odes(state + k3, parameters)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) /6

def generate(data_length, odes, state, parameters):
    data = np.zeros([state.shape[0], data_length])

    for i in range(5000):
        state = rk4(odes, state, parameters)

    for i in range(data_length):
        state = rk4(odes, state, parameters)
        data[:, i] = state

    return data

def lorenz_odes(state, parameters):
    x, y, z = state
    sigma, beta, rho = parameters
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def lorenz_generate(data_length):
    return generate(data_length, lorenz_odes, \
        np.array([-8.0, 8.0, 27.0]), np.array([10.0, 8/3.0, 28.0]))

data = lorenz_generate(2**15)


from mpl_toolkits.mplot3d.axes3d import Axes3D

figure = plt.figure()
axes = Axes3D(figure)
axes.plot3D(data[0], data[1], data[2], linewidth=0.1)
figure.add_axes(axes)
plt.show()

plt.figure()
plt.plot(acf(data[0], 1000))
plt.plot(acf(data[1], 1000))
plt.plot(acf(data[2], 1000))

figure = plt.figure()
axes = Axes3D(figure)
axes.plot3D(data[0][:8000], data[0][75:][:8000], data[0][155:][:8000], linewidth=0.7)
figure.add_axes(axes)
plt.show()


n=2
data_tr = [d[:8000] for d in data]
data_te = [d[8000:] for d in data]

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for embedding in np.arange(1, 2):
    for n in np.arange(2, 10):
        plt.cla()
        embeddings_1 = np.vstack([data_tr[0][i+np.arange(0, embedding*n, embedding)] for i in range(len(data_tr[0])-embedding*n)])
        embeddings_2 = np.vstack([data_tr[1][i+np.arange(embedding, embedding*n, embedding)] for i in range(len(data_tr[0])-embedding*n)])
        embeddings_3 = np.vstack([data_tr[2][i+np.arange(embedding, embedding*n, embedding)] for i in range(len(data_tr[0])-embedding*n)])
        embeddings = np.hstack([embeddings_1, embeddings_2, embeddings_3])
        embeddings = embeddings_1
        embeddings = embeddings[:, :]
        y = embeddings[:, 0]
        x = embeddings[:, 1:]
        coeffs = np.linalg.pinv(x.T@x)@(x.T@y)

        embeddings_1 = np.vstack([data_te[0][i+np.arange(0, embedding*n, embedding)] for i in range(len(data_te[0])-embedding*n)])
        embeddings_2 = np.vstack([data_te[1][i+np.arange(embedding, embedding*n, embedding)] for i in range(len(data_te[0])-embedding*n)])
        embeddings_3 = np.vstack([data_te[2][i+np.arange(embedding, embedding*n, embedding)] for i in range(len(data_te[0])-embedding*n)])
        embeddings = np.hstack([embeddings_1, embeddings_2, embeddings_3])
        embeddings = embeddings_1
        embeddings = embeddings[:3000, :]
        y = embeddings[:, 0]
        x = embeddings[:, 1:]
        y_hat = x @ coeffs
        ax[0].cla()
        ax[0].scatter(y, y_hat, s=1)
        ax[0].set_title('embedding:{}, m:{}'.format(embedding, n))
        ax[1].cla()
        ax[1].plot(y)
        ax[1].plot(y_hat, linestyle='--')
        plt.pause(1)


x = data_tr[0][:8]
y_hats=[]
for i in range(1000):
    y_hat = x @ coeffs
    x = np.hstack([y_hat, x[:-1]])
    y_hats.append(y_hat)

plt.plot(np.hstack(y_hats))


from statsmodels.tsa.seasonal import STL

stmf = pd.read_csv("https://www.mortality.org/File/GetDocument/Public/STMF/Outputs/stmf.csv",skiprows=1)


def get_stl_nation(cc='ITA'):
    data = stmf.loc[stmf['CountryCode'] == cc, ['Week', 'Year', 'RTotal']]
    data.index = pd.DatetimeIndex(
        pd.DatetimeIndex(data['Year'].apply(lambda x: '{}-01-01'.format(x), 0)) + pd.to_timedelta(
            data['Week'].apply(lambda x: '{}d'.format(x * 7), 0)))
    y = data.groupby(data.index).sum()['RTotal']
    stl = STL(y, seasonal=53, period=52)
    res = stl.fit()
    return res


cc = 'NLD'
res = get_stl_nation(cc)


stmf['CountryCode'].value_counts()

data = stmf.loc[stmf['CountryCode'] == cc, ['Week', 'Year', 'RTotal']]
data.index = pd.DatetimeIndex(
    pd.DatetimeIndex(data['Year'].apply(lambda x: '{}-01-01'.format(x), 0)) + pd.to_timedelta(
        data['Week'].apply(lambda x: '{}d'.format(x * 7), 0)))
y = data.groupby(data.index).sum()

y[['Year', 'Week']] /=3
y[['Year', 'Week']] = y[['Year', 'Week']].astype(int)
y = y.loc[(y['Week']<53) & (y['Year']<2023)]
y_mat = y.pivot(index='Year', columns='Week', values='RTotal').T


y_season = y_mat.quantile(0.5, axis=1)
y_mat.plot(alpha=0.3)
y_season.plot(linestyle='--', color='r')
y_res = pd.DataFrame((y_mat-y_season.values.reshape(-1, 1)).T.values.ravel()[:len(y)], index=y.index)

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
y['RTotal'].plot(ax=ax[0])
y_seasonal = pd.DataFrame(np.tile(y_season.values, len(y['Year'].value_counts()))[:len(y)], index=y.index)
y_seasonal.plot(ax=ax[0])
(y['RTotal'] -y_seasonal[0]).plot(ax=ax[1])



import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

stmf = pd.read_csv("https://www.mortality.org/File/GetDocument/Public/STMF/Outputs/stmf.csv",skiprows=1)
stmf.index = pd.DatetimeIndex(pd.DatetimeIndex(stmf['Year'].apply(lambda x: '{}-01-01'.format(x), 0)) + pd.to_timedelta( stmf['Week'].apply(lambda x: '{}d'.format(x * 7), 0)))
stmf.head()

def get_national_obs(data, cc='ITA'):
  """
  Get death rate for a given country
  """
  data = data.loc[stmf['CountryCode'] == cc, ['Week', 'Year', 'RTotal']]
  return data.groupby(data.index).sum()['RTotal']




y1 = get_national_obs(stmf, 'NLD')
y1 = pd.concat([y1, pd.Series(y1.index.year, name='Year', index=y1.index), pd.Series(y1.index.dayofyear, name='Day', index=y1.index)], axis=1)
y1 = y1.loc[(y1.Day>7) & (y1.Year<2023)]
y_mat = y1.pivot(index='Year', columns='Day', values='RTotal').T


y_season = y_mat.quantile(0.5, axis=1)
y_mat.plot(alpha=0.3)
y_season.plot(linestyle='--', color='r')
y_res = pd.DataFrame((y_mat-y_season.values.reshape(-1, 1)).T.values.ravel()[:len(y)], index=y.index)

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
y['RTotal'].plot(ax=ax[0])
y_seasonal = pd.DataFrame(np.tile(y_season.values, len(y['Year'].value_counts()))[:len(y)], index=y.index)
y_seasonal.plot(ax=ax[0])
(y['RTotal'] -y_seasonal[0]).plot(ax=ax[1])


from statsmodels.datasets import macrodata
ds = macrodata.load_pandas()
data = np.log(ds.data.m1)
base_date = f"{int(ds.data.year[0])}-{3*int(ds.data.quarter[0])+1}-1"
data.index = pd.date_range(base_date, periods=data.shape[0], freq="QS")
data.plot()
plt.plot(data)



