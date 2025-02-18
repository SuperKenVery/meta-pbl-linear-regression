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

def plot(x_key: str, y_key: str):
    sns.regplot(data=data, x=x_key, y=y_key)

def show_results(x_key: str, y_key: str):
    model = analyze(x_key, y_key)
    plot(x_key, y_key)
    print(model.summary())

show_results(Y_key[1], Z_key)
show_results(Y_key[0], Z_key)
# %%
show_results(X_key, Y_key[0])
show_results(X_key, Y_key[1])
