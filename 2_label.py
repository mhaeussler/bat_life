# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:09:25 2020

@author: mh
"""

import pandas as pd
import numpy as np


def to_date(df, cols):    
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df    


###################################
datapath = "data/"
filename = "B0018.mat" + "_transformed.csv"


df = pd.read_csv(datapath + filename)
df = to_date(df, ["abs_time", "glob_time"])
df["Time"] = pd.to_timedelta(df["Time"])

print(df.dtypes)


"""
According to documentation:
    The experiments were stopped when the batteries\
    reached end-of-life (EOL) criteria, which was a\
    30% fade in rated capacity (from 2Ahr to 1.4Ahr).
"""

EOL_criteria = 1.4

# Label End Of Life indicator
df["EOL"] = None
df.loc[(df["Capacity"] < EOL_criteria), "EOL"] = True
df["EOL"].fillna(method="ffill", inplace=True)
df["EOL"].fillna(False, inplace=True)

# Label Time To Event
if df["EOL"].max() == False:              # assume last observation is EOL
    failure = df.iloc[-1]["abs_time"]
else:
    failure = df[df["EOL"]==True].iloc[0]["abs_time"]   # first occurence

tte = failure - df["abs_time"]
df["tte"] = tte

#############################
# Persist
df.to_csv(datapath + filename + "_labeled.csv")
