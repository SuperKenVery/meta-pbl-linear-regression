#!/usr/bin/env python

# Analyze data we collected
# Notes:
    # 1. We did not have R&D intensity of DeepSeek, so we set it as the average
    #    R&D intensity of ther companies.
    # 2. Anthropic has 0 patents after 2020 while we analyze new citations vs
    #    total citations after 2020, so we set it as the average of other companies.

# %%
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from matplotlib import pyplot as plt
from typing import List, Union

data = pd.read_csv('data.csv')

X_key = "TMT past work/education experience in R&D"
Y_key = [
    "New citations / Total citations",
    "R&D intensity",
]
Z_key = "Model capability"

# %%
def analyze(x_key: str, y_key: str):
    x = data[x_key]
    x = sm.add_constant(x)
    y = data[y_key]
    result = sm.OLS(y, x).fit()
    return result

def plot(x_key: str, y_key: str, ax=None, label=None):
    sns.regplot(data=data, x=x_key, y=y_key, ax=ax, label=label)

def show_results(x_key: str, y_key: str, ax=None, label=None):
    model = analyze(x_key, y_key)
    plot(x_key, y_key, ax, label)
    # print(model.summary())

fig, ax = plt.subplots(1, 1)
show_results(Y_key[0], Z_key, ax, label=Y_key[0])
show_results(Y_key[1], Z_key, ax, label=Y_key[1])
ax.set_xlabel(" and ".join(Y_key))
ax.legend()
fig.savefig("y_to_z.png")
# %%
fig, ax = plt.subplots(1, 1)
show_results(X_key, Y_key[0], ax, Y_key[0])
show_results(X_key, Y_key[1], ax, Y_key[1])
ax.legend()
ax.set_ylabel(" and ".join(Y_key))
fig.savefig("x_to_y.png")
