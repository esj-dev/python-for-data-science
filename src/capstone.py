# 1. Importing modules and classes

import cufflinks as cf
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import plotly as py
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity as kd

cf.go_offline()

# 2. Fetching time series data

list_countries = ['AT', 'CH', 'CN', 'DK', 'GB', 'JP']
k = len(list_countries)

list_dfs = [np.nan] * k

for i in np.arange(0, k, 1):

    ticker = 'NASDAQOMX/NQ' + list_countries[i]

    list_dfs[i] = pdr.data.DataReader(
        data_source='quandl',
        name=ticker,
        start='2005-01-01',
        end='2017-12-31',
        access_key='7X-yDZjXQ8DePVQCEGef'
        )

# 3. EDA - Univariate

j = 4

# time series

fig_ts = plt.figure()
axes = fig_ts.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(list_dfs[j].index.values, list_dfs[j]['IndexValue'], 'b')
axes.set_xlabel('Date')
axes.set_ylabel('Closing Value')
axes.set_title('NASDAQ OMX Global Index ' + '(' + list_countries[j] + ')')
plt.grid()
plt.show()

# log-returns

X = np.log(list_dfs[j]['IndexValue'])

dX = np.diff(X).tolist()
dX = pd.DataFrame(dX)
dX = dX.rename(columns={0: 'Log-return'})  # constructing data frame

fig_lr = dX['Log-return'].iplot(kind='hist', bins=30, asFigure=True)

py.offline.plot(fig_lr, filename='figures/log-returns.html')

# 4. EDA - Multivariate

# create data frame on long form

list_dfs_long = [np.nan] * k

for j in np.arange(0, k, 1):

    df_tidy = list_dfs[j]['IndexValue'].to_frame()
    df_tidy.index.names = ['date']
    df_tidy['country'] = list_countries[j]
    df_tidy['variable_name'] = 'nasdaq_omx_global_index'
    df_tidy['variable_value'] = df_tidy['IndexValue']
    df_tidy = df_tidy[['country', 'variable_name', 'variable_value']]

    list_dfs_long[j] = df_tidy

df_long = pd.concat(list_dfs_long)
df_long.shape

# collinearity scatter plot (bivariate)

i = 1
j = 3

list_i = df_long[df_long['country'] == list_countries[i]]['variable_value'].values.tolist()
list_j = df_long[df_long['country'] == list_countries[j]]['variable_value'].values.tolist()

fig_col = plt.figure()
axes = fig_col.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(list_i, list_j, 'bo', markersize='2')
axes.set_xlabel(list_countries[i])
axes.set_ylabel(list_countries[j])
axes.set_title('NASDAQ OMX Global Index (Scatterplot)')
plt.show()

# convert df to wide format

df_wide = df_long.pivot(columns="country", values='variable_value')
df_wide.head()

# index correlation plot

df_corr = df_wide.corr()

fig_corr = plt.figure()
axes = fig_corr.add_axes([0.1, 0.1, 0.8, 0.8])
sns.heatmap(df_corr, cmap='magma', linewidths=1, linecolor='white')
axes.set_xlabel('Country')
axes.set_ylabel('Country')
axes.set_title('NASDAQ OMX Global Index Correlation')
plt.show()

# 5. Time Series Analysis

j = 3

df_wide_asc = df_wide.sort_index(ascending=True)
df_wide_ema = df_wide_asc.rolling(window=25, win_type='gaussian').mean(std=10)

fig_ema = plt.figure()
axes = fig_ema.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(df_wide_asc.index.values, df_wide_asc[list_countries[j]], 'b')
axes.plot(df_wide_asc.index.values, df_wide_ema[list_countries[j]], 'r')
axes.set_xlabel('Date')
axes.set_ylabel('Value')
axes.set_title('NASDAQ OMX Global Index ' + '(' + list_countries[j] + ')')
axes.legend([list_countries[j] + '_raw', list_countries[j] + '_ema'], fontsize=9)
plt.grid()
plt.show()

density = kd(kernel='gaussian', bandwidth=0.75).fit(dX)
dX_kde = density.score_samples(dX)

fig_kde = plt.figure()
axes = fig_kde.add_axes([0.1, 0.1, 0.8, 0.8])
axes.hist(dX, bins=30)
axes.plot(dX_kde, 'r')
axes.set_xlabel('Date')
axes.set_ylabel('Value')
axes.set_title('NASDAQ OMX Global Index ' + '(' + list_countries[j] + ')')
plt.grid()
plt.show()
