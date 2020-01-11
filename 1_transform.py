# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 07:48:16 2020

@author: mh

Try to predict the remaining lifetime of batteries.
Use nasa research data.

"""

import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pprint import pprint as pp



def load_raw_mat(datapath, filename):
    """
    Load the .mat data and parse it to retrieve cycle data.
    """
    
    mat = loadmat(datapath + filename, squeeze_me=True) # struct_as_record=False
    data = mat[os.path.splitext(filename)[0]]   # skip .mat
    
    cycle = data["cycle"]
    df = pd.DataFrame(cycle.item().T)
    
    return df
    

def parse_charge_or_discharge(df, selected_state):
    
    charge = df[df["type"]==selected_state]
    item = charge["data"].iloc[0]
    select_cols = item.dtype.names
    print(f"{selected_state} Columns:", select_cols)
    
    def extract_charge(series, colname):
        return series[colname].item()
    
    
    CHARGE = pd.DataFrame()
    for item in charge["data"].iteritems():
        
        number = item[0]
        ambient_temp = charge["ambient_temperature"].loc[number]
        
        # parse global time for each charging cycle start
        global_time = charge["time"].loc[number]
        first = global_time[:-1].astype(int).tolist()
        last  = float(global_time[-1])
        tmp = "-".join([str(i) for i in first])
        tmp += "-" + str(round(last, 5))
        
        dt = datetime.datetime.strptime(tmp, '%Y-%m-%d-%H-%M-%S.%f')
        # print(dt)
    
        charge_ = pd.DataFrame()
        for col in select_cols:
            charge_[col] = extract_charge(item[1], col)
        
        charge_["state"] = selected_state
        charge_["counter"] = filename.split(".")[0] + "_" + str(number)
        charge_["ambient_temperature"] = ambient_temp
        charge_["glob_time"] = dt
        # print(charge_.head())
        
        CHARGE = pd.concat((CHARGE, charge_))
    
    CHARGE["Time"] = pd.to_timedelta(CHARGE["Time"], unit="seconds")
    CHARGE["abs_time"] = CHARGE["glob_time"] + CHARGE["Time"]
    CHARGE.reset_index(inplace=True)
    print(CHARGE.shape)
    # print(CHARGE.head())
    # print(CHARGE.tail())
    
    return CHARGE


##################
datapath = "data/"
filename = "B0018.mat"

df = load_raw_mat(datapath, filename)
available_states = df["type"].unique().tolist()
print("Types of states:", available_states)


df1 = parse_charge_or_discharge(df, "charge")
df2 = parse_charge_or_discharge(df, "discharge")
dff = pd.concat((df1, df2), sort=False)
assert len(dff) == len(df1) + len(df2)

dff.sort_values(by="abs_time", inplace=True)
dff.set_index("abs_time", inplace=True)
dff["Voltage_measured"].plot()


################
# Persist transformed data
dff.to_csv(datapath + filename + "_transformed.csv")


    
# ### IMPEDANCE needs to be treaded extra
# selected_state = "impedance"
# impedance    = df[df["type"]==selected_state]
# item = impedance["data"].iloc[0]
# select_cols = item.dtype.names
# print(f"{selected_state} Columns:", select_cols)

# def extract_impedance(series, colname):
    
#     tmp = series[colname]
    
#     if "complex" in tmp.dtype.name:
#         return tmp.item()
        
#     # if colname not in ["Re", "Rct"]:
#         # return series[colname].item()

# impedance = pd.DataFrame()
# for col in select_cols:
#     impedance[col] = extract_impedance(item, col)

# charge["state"] = selected_state
# print(charge.head())

# discharge = df[df["type"]=="discharge"]    
    
    
    
    
    
    
    