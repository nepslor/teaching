import pandas as pd
import numpy as np
from os.path import join
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# import box-cox transformation
from scipy import stats


data_path = 'TimeSeriesForecasting/data/final_challenge/'

# --------------------------------------------------
# ------------------ PM10 dataset ------------------
# --------------------------------------------------

data_pm10 = pd.read_csv(join(data_path, 'TI_MOR_20160101_20221231_dataset_final.csv'), index_col=0, parse_dates=True)


(data_pm10.iloc[:, -48::6]).plot(alpha=0.6)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
data_pm10[[c for c in data_pm10.columns if 'GLOB' in c and 'step33' in c]].plot(alpha=0.6, ax=ax[0])
data_pm10[[c for c in data_pm10.columns if 'T_2M' in c and 'step120' in c]].plot(alpha=0.6, ax=ax[1])
data_pm10[[c for c in data_pm10.columns if 'RELHUM_2M_c2' in c and 'step120' in c]].plot(alpha=0.6, ax=ax[2])
[a.legend(fontsize=5, ncols=2) for a in ax]
plt.suptitle('NWP forecasts')
plt.savefig('TimeSeriesForecasting/figs/pm10_dataset_1.png', dpi=200)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
data_pm10[[c for c in data_pm10.columns if 'NO2' in c and 'm0' in c]].plot(alpha=0.6, ax=ax[0])
data_pm10[[c for c in data_pm10.columns if 'O3' in c and 'm0' in c]].plot(alpha=0.6, ax=ax[1])
data_pm10[[c for c in data_pm10.columns if 'NOx' in c and 'm20' in c]].plot(alpha=0.6, ax=ax[2])
[a.legend(fontsize=5, ncols=2) for a in ax]
plt.suptitle('measures')
plt.savefig('TimeSeriesForecasting/figs/pm10_dataset_2.png', dpi=200)


# --------------------------------------------------
# ------------------ LIC dataset -------------------
# --------------------------------------------------

data_lic = pd.read_pickle(join(data_path, 'energy_lic_2021-2022.pk'))

# anonymize data
meters = data_lic['timeseries']['community_meters']
signals = ["e_pos",   "e_neg"]
meters = pd.concat({i: v[signals] for i, v in enumerate(meters.values())}, axis=1)

meters['battery', 'e_pos'] = data_lic['timeseries']['battery_meters']['asilo']['e_pos']
meters['battery', 'e_neg'] = data_lic['timeseries']['battery_meters']['asilo']['e_neg']

meters['PCC', 'e_pos'] = data_lic['timeseries']['pcc']['e_pos']
meters['PCC', 'e_neg'] = data_lic['timeseries']['pcc']['e_neg']

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
meters.loc[:, meters.columns.get_level_values(1)=='e_pos'].droplevel(1, axis=1).plot(alpha=0.6, ax=ax)
plt.gca().set_title('positive energy flows')
plt.gca().legend(fontsize=5, ncol=2)
plt.savefig('TimeSeriesForecasting/figs/lic_1.png', dpi=200)

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
meters.loc[:, meters.columns.get_level_values(1)=='e_neg'].droplevel(1, axis=1).plot(alpha=0.6, ax=ax)
plt.gca().set_title('negative energy flows')
plt.gca().legend(fontsize=5, ncol=2)
plt.savefig('TimeSeriesForecasting/figs/lic_2.png', dpi=200)

# keep only e_pos e_neg
meters.to_pickle(join(data_path, 'lic_meters.zip'))


# ------------ download meteo forecasts ------------

from influxdb import DataFrameClient
import pickle as pk
from tqdm import tqdm

db_pars = {"host": "",
           "port": 443,
           "user": "forecast_meteoblue_ro",
           "password": "",
           "db": "forecast_meteoblue",
           "ssl": True,
           }
client = DataFrameClient(db_pars['host'],
                         port=db_pars['port'],
                         username=db_pars['user'],
                         password=db_pars['password'],
                         database=db_pars['db'],
                         ssl=db_pars['ssl'])

query = "SELECT value FROM forecasts_history WHERE (location = 'Lugano') AND time>= '2020-01-01T00:00:00Z' AND time < '2023-02-23T05:45:00Z' GROUP BY signal, step"
df_dict = client.query(query)


nwp_signals = ['ghi_backwards', 'temperature', 'relativehumidity']
cols = []
multiindex = []
for k, v in tqdm(df_dict.items()):
    k1 = k[1][0][1]
    k2 = int(k[1][1][1])
    if k1 not in nwp_signals or k2 <0:
        continue
    else:
        v.columns = [(k1, k2)]
        cols.append(v)
        multiindex.append((k1, k2))

mi = pd.DataFrame(columns=pd.MultiIndex.from_tuples(multiindex))
for k, v in zip(mi.columns, cols):
    mi[k] = v[k]

mi.to_pickle(join(data_path, 'lic_nwp.zip'))

# ---------------- download meteo ----------------
db_pars = {"host": "",
           "port": 443,
           "user": "lic_ro",
           "password": "",
           "db": "lic",
           "ssl": True,
           }
client = DataFrameClient(db_pars['host'],
                         port=db_pars['port'],
                         username=db_pars['user'],
                         password=db_pars['password'],
                         database=db_pars['db'],
                         ssl=db_pars['ssl'])

df = pd.DataFrame()
signals = ['AirPressure', 'AirTemp_Avg', 'PyrIrradiance_Avg', 'Ramount_Tot', 'RelHumidity', 'WindSpeed']
for signal in signals:
    query = "SELECT mean({}) as {} FROM lic_meteo WHERE time>= '2020-01-01T00:00:00Z' AND time < '2023-02-22T00:00:00Z' GROUP BY time(900s)".format(signal, signal)
    df = client.query(query)['lic_meteo'].combine_first(df)

df.to_pickle(join(data_path, 'lic_meteo.zip'))

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
df.plot(ax=ax)
plt.gca().legend(fontsize=5, ncol=2)
plt.savefig('TimeSeriesForecasting/figs/lic_4.png', dpi=200)


