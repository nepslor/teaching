import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import wget
from glob import glob
import seaborn as sb

figsize = (10, 5)

# ------------------------------- Electricity Transformer Temperature 1 -----------------------------------------------
# FROM https://github.com/zhouhaoyi/ETDataset

save_folder = 'TimeSeriesForecasting/data/minichallenge'
df_oil = pd.read_csv('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv', index_col=0,
                     parse_dates=True)

fig, ax = plt.subplots(1, 1, figsize=figsize)
df_oil.iloc[:2000, :].plot(ax=ax, alpha=0.58)
plt.savefig(join(save_folder, 'oil_1.png'))
df_oil.to_pickle(join(save_folder, 'oil_1.pk'))

# ------------------------------- Electricity Transformer Temperature 2 -----------------------------------------------
save_folder = 'TimeSeriesForecasting/data/minichallenge'
df_oil = pd.read_csv('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv', index_col=0,
                     parse_dates=True)
df_oil.plot()
plt.pause(0.001)
df_oil.to_pickle(join(save_folder, 'oil_2.pk'))

# ------------------------------- AIR data quality -----------------------------------------------
# FROM https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data

dat = wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip", out=save_folder)
import zipfile
with zipfile.ZipFile(dat, 'r') as zip_ref:
    zip_ref.extractall(save_folder)

dfs = []
for f in glob("{}/PRSA_Data_20130301-20170228/*.csv".format(save_folder)):
    df_i = pd.read_csv(f)
    df_i["datetime"] = pd.to_datetime(df_i[["year", "month", "day", "hour"]])
    df_i.set_index("datetime", inplace=True)
    df_i.drop(["year", "month", "day", "hour", "No"], axis=1, inplace=True)
    dfs.append(df_i)


df_air = pd.concat(dfs, axis=0)
fig, ax = plt.subplots(1, 1, figsize=figsize)
for l in df_air.station.unique():
    df_air[df_air.station == l]['PM10'].iloc[:1000].plot(ax=ax, label=l, alpha=0.8)
    plt.pause(0.001)
ax.legend(ncol=2)
plt.savefig(join(save_folder, 'air.png'))
df_air.to_pickle(join(save_folder, 'air.pk'))

# do a markdown table from the df_air columns
df_air.iloc[:10, :].to_markdown()


# ------------------------------- HOTELS  -----------------------------------------------
# FROM https://github.com/ashfarhangi/COVID-19

df_c1 = pd.read_excel("https://github.com/ashfarhangi/COVID-19/raw/main/data/COVID19_time_series.xlsx", parse_dates=True)
df_c1.drop(["Year", "Month", "Day"], axis=1, inplace=True)
df_c1.set_index('Date', inplace=True, drop=True)
len(df_c1.Location.unique())

lagged_mav = lambda x, k: x.copy().rolling('{}d'.format(k)).mean()

df_c1.loc[df_c1['Location']=='NewYork', 'Revenue'].plot()
lagged_mav(df_c1.loc[df_c1['Location']=='NewYork', 'Revenue'], 30).plot()
lagged_mav(df_c1.loc[df_c1['Location']=='NewYork', 'Revenue'], 60).plot()

fig, ax = plt.subplots(1, 1, figsize=figsize)
sb.lineplot(data=df_c1, x=df_c1.index, y="Revenue", hue="Location", ax=ax, alpha=0.8)
plt.savefig(join(save_folder, 'hotels.png'))

df_c1.to_pickle(join(save_folder, 'hotels.pk'))

df_c1.iloc[:10, :].to_markdown()


# ------------------------------- METRO TRAFFIC -----------------------------------------------
# FROM https://archive-beta.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
dat = wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz", out=save_folder)
# extract gz file from dat
import gzip
with gzip.open(dat, 'rb') as f_in:
    with open(dat[:-3], 'wb') as f_out:
        f_out.write(f_in.read())

df_traffic = pd.read_csv(dat[:-3], parse_dates=True, index_col='date_time')
df_traffic.loc[:, df_traffic.dtypes.isin(['float64', 'int64', 'int'])].plot()


# ------------------------------- WEATHER DATA -----------------------------------------------
# FROM https://www.tensorflow.org/tutorials/structured_data/time_series

dat = wget.download('https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip', out=save_folder)
with zipfile.ZipFile(dat, 'r') as zip_ref:
    zip_ref.extractall(save_folder)
df_weather = pd.read_csv(dat[:-4], parse_dates=True, index_col='Date Time')
df_weather = df_weather.resample('1h').mean()
df_weather.to_pickle(join(save_folder, 'weather.pk'))

fig, ax = plt.subplots(1, 1, figsize=figsize)
((df_weather-df_weather.mean())/df_weather.std()).iloc[:1000, :].plot(ax=ax, alpha=0.2)
((df_weather['T (degC)']-df_weather['T (degC)'].mean())/df_weather['T (degC)'].std()).iloc[:1000].plot(color='orange')
fig.savefig(join(save_folder, 'weather.png'))

df_weather.head().to_markdown()