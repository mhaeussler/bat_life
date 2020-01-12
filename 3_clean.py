# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:17:26 2020

@author: mh
"""

import pandas as pd
import numpy as np
from wtte import WTTE
import matplotlib.pyplot as plt
from scipy.special import gamma


def to_date(df, cols):    
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df    
    

def to_dt_delta(df, cols):
    for col in cols:
        df[col] = pd.to_timedelta(df[col])
    return df



#######################################
# Loading

def load(frompath):
    
    print("Loading...", frompath)
    
    return pd.read_csv(frompath, index_col=[0])
    

####################################
# Transform

def transform(df):
    
    print("Transforming...")
    
    df = to_date(df, ["abs_time", "glob_time"])
    df = to_dt_delta(df, ["Time", "tte"])
    
    # print(df.dtypes)

    return df


################################
# Cleaning

def clean(df):
    
    print("Cleaning...")
    
    df["Capacity"].fillna(method="ffill", inplace=True)
    df["Capacity"].fillna(method="bfill", inplace=True)
    
    # No load on charge
    df["Voltage_load"].fillna(0, inplace=True)
    df["Current_load"].fillna(0, inplace=True)
    
    # No charge on discharge
    df["Voltage_charge"].fillna(0.0, inplace=True)
    df["Current_charge"].fillna(0.0, inplace=True)
    
    
    # One hot encode state
    df.loc[df[df["state"]=="charge"].index, "charge"] = 1
    df["charge"].fillna(0, inplace=True)
    
    return df


#########################################
# Prepare for Training

def sample(df, seq_len, sample_id=None):
    # TODO currently does oversampling
    
    if sample_id:
        rand_idx = sample_id
    else:
        rand_idx = np.random.choice(df.index)
    samp = df.iloc[rand_idx : rand_idx + seq_len]
    # print("WARNING SHAPE:", sample.shape) if len(sample) != seq_len else None
    
    return samp


def tte_to_float(df):
    
    # df["y_tte"] = None
    tte = df.tte.dt.total_seconds() / 86400
    df.loc[df.index, "y_tte"] = tte
    
    return df


def sample_df_to_tensor(sample, features, labels):
    
    x = sample[features].values
    y = sample.iloc[-1][labels].values.astype(float)    # tte, u
    
    # consider padding later:
    # np.pad(a, (1,0), 'constant', constant_values=0)
    
    # print(x.shape, y.shape)
    return x, y
    

def stack_samples_to_tensor(samples, features, labels, seq_len):
    
    X, Y = [], []
    for sample in samples:

        if len(sample) == seq_len:        
            x, y = sample_df_to_tensor(sample, features, labels)
            X.append(x), Y.append(y)
        
    return np.array(X), np.array(Y)


#########################################
# Learning

def weib_pdf(x, k, lam):
    return (k / lam * (x / lam)**(k-1)) * np.exp(-(x/lam)**k)

def weib_mean(k, lam):
    return lam * gamma(1+(1/k))

def weib_mode(k, lam):
    # k   = shape / b / beta
    # lam = scale / a / alpha
    return lam * ((k-1)/k)**(1/k)


#####################################
# Reporting
    
def plot(row):
    tte = row.loc["y_tte"]
    u = row.loc["y_u"]
    a = row.loc["a"]
    b = row.loc["b"]
    c = "r" if u == 1.0 else "g"
    x = np.linspace(start=0, stop=int(tte+2), num=100)
    
    y = weib_pdf(x, b, a)
    # wb_mean = weib_mean(b, a)
    wb_mode = weib_mode(b, a)
    
    fig, ax = plt.subplots(1)
    # ax.axvline(x=wb_mean, linestyle="--", c="b")
    ax.axvline(x=wb_mode, linestyle="--", c="r")
    ax.plot(x, y)
    ax.plot(tte, 0.00001, marker="o", color=c)

    return fig, ax


#########################################
# Exec
datapath = "data/"
filename = "B0005.mat" + "_transformed.csv" + "_labeled.csv"

df = load(datapath + filename)
df = transform(df)
df = clean(df)


# Visualize
charge = df[df["state"] == "charge"]
discharge = df[df["state"] == "discharge"]
df.groupby("state").count()["abs_time"].plot(kind="bar")


# Features
loc = ["abs_time", "index", "Time", "state", "counter", "glob_time", "EOL", "tte"]
blacklist = ["ambient_temperature"]
features = [i for i in df.columns if i not in loc and i not in blacklist]
labels = ["y_tte", "y_u"]


# Uncensoring Label
df.loc[:, "y_u"] = df["EOL"].max()
df = df[df["tte"].dt.total_seconds() >= 0]   # crop measures later than event
df = tte_to_float(df)


##########################################
# Configure
features_ = [
    "Voltage_measured",
    # "Current_measured",
    "Temperature_measured",
    # "Current_charge",
    "Voltage_charge",
    # "Current_load",
    # 'Voltage_load',
    'Capacity',
    'charge'
    ]


t_x = 50               # Length of series
x_n = len(features_)   # number of features
num_samples = 200      # Samples to pick from data


df = df[df["state"]=="charge"]      # Train on state

samples_tr = [sample(df, t_x) for i in range(num_samples)]   # sampling
samples_te = [sample(df, t_x) for i in range(num_samples)]   # sampling


X_tr, Y_tr = stack_samples_to_tensor(samples_tr, features_, labels, t_x)
print("X_tr:", X_tr.shape, " Y_tr:", Y_tr.shape)

X_te, Y_te = stack_samples_to_tensor(samples_te, features_, labels, t_x)
print("X_te:", X_te.shape, " Y_te:", Y_te.shape)


model = WTTE(t_x, n_features=x_n)
model.compile_model()
model.fit(X_tr, Y_tr, epochs=1000)
loss = model.model.history.history["loss"]
plt.plot(loss)

y_pred = model.model.predict(X_te)
report = pd.DataFrame(np.hstack((Y_te, y_pred)), columns=labels + ["a", "b"])
report["weib_mode"] = weib_mode(report["b"], report["a"])

# 
# for i in range(report.index.values.max()):
#     row = report.iloc[i]
#     fig, ax = plot(row)
#     plt.savefig(f"results/{i}.png")
#     plt.close()

#################################
# Scoring
# RUL using Heaviside Step Function

# Zwischenergebnis (for verification)
report["heavi"] = np.heaviside(report["y_tte"] - report["weib_mode"], 0)
print("Plain Heaviside-Score:", report["heavi"].sum() / len(report))

def score_rul(y, y_pred):
    
    term1 = np.heaviside(y - y_pred, 0) * (np.exp((y - y_pred) / 13) - 1)  # term activated on early predictions
    term2 = np.heaviside(y_pred - y, 0) * (np.exp((y - y_pred) / 10) - 1)  # term activated on too late predictions
    score_ = term1 + term2
    
    return score_


report["rul_score"] = score_rul(report["y_tte"], report["weib_mode"])
print("Remaining Useful Time (RUL) Score:", report["rul_score"].sum() / len(report))























