import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

warnings.filterwarnings("ignore")

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.color'] = 'k'

raw_data = pd.read_excel("C:/Users/Jurie/PycharmProjects/Thesis/Data/Superstore.xls")

data = raw_data.loc[raw_data['Category'] == 'Furniture']
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
data.drop(cols, axis=1, inplace=True)
data = data.sort_values('Order Date')
data = data.groupby('Order Date').sum()

y_true = data['Sales'].resample('MS').mean()
y_train = y_true['2014':'2015']
y_validate = y_true['2016':'2016']
y_test = y_true['2017':]

y_true.plot(figsize=(15, 6))
plt.show()

#

