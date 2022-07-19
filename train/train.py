import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import joblib
import json

# Parameters
data_processed_path="data/data_processed.csv"
model_path="model/model.joblib.dat"
metrics_path="metrics/metrics_model.json"

# Load data

amount_by_month=pd.read_csv(data_processed_path)
amount_by_month["paidAt_Month"]=pd.to_datetime(amount_by_month["paidAt_Month"])
amount_by_month=amount_by_month.set_index("paidAt_Month")

# Transform data

amount_by_month["amountfinancedByXepelin"]=np.log(amount_by_month["amountfinancedByXepelin"])

# Test if the serie is stational

print("Test if the time serie is stational:")
print("-"*100)
df_stationarityTest = adfuller(amount_by_month['amountfinancedByXepelin'], autolag='AIC')

p_value=df_stationarityTest[1]
print("P-value: ", p_value)

if p_value<=0.5:
    print("There is no evidence to conclude the data is non-stationary")
else:
    print("The data is non-stationary")

# Plot Partial Autocorrelation Function
lags=round(len(amount_by_month)/2)-1

pacf = plot_pacf(amount_by_month['amountfinancedByXepelin'], lags=lags)
plt.savefig("img/plot_pacf.png")

# Train an AR model

metrics_models=pd.DataFrame({"LAG":[],"AIC":[],"BIC":[]})


for i in range(1,lags+1):
    ar_model = AutoReg(amount_by_month['amountfinancedByXepelin'], lags=i).fit()
    metrics=pd.DataFrame({"LAG":[i],"AIC":[ar_model.aic],"BIC":[ar_model.bic]})
    metrics_models=pd.concat([metrics_models,metrics])

# Select best model based on BIC
best_lag=metrics_models.sort_values(by="BIC").reset_index().iloc[0]["LAG"]

# Train model
ar_model=AutoReg(amount_by_month['amountfinancedByXepelin'], lags=best_lag).fit()

print(ar_model.summary())

# Save the model

joblib.dump(ar_model, model_path)
print("Model Saved!")

# Save the metrics of the model

metrics={"AIC":ar_model.aic,"BIC":ar_model.bic,
    "Intercept":ar_model.params[0],"Coef_1":ar_model.params[1],
    "Coef_2":ar_model.params[2],"Coef_3":ar_model.params[3]}

with open(metrics_path, 'w') as fd:
    json.dump(metrics, 
            fd, indent=4
    )






