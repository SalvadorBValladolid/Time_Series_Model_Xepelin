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
from datetime import datetime

# Parameters
data_processed_path="data/data_processed.csv"
model_path="model/model.joblib.dat"
prediction_img_path="img/predictions.png"
csv_prediction_path="data/predictions.csv"


# Load model

model=joblib.load(model_path)

# Load data

amount_by_month=pd.read_csv(data_processed_path)
amount_by_month.loc[-1] = ["2021-10-01",None]

amount_by_month["paidAt_Month"]=pd.to_datetime(amount_by_month["paidAt_Month"])
amount_by_month=amount_by_month.set_index("paidAt_Month")

# Predictions

amount_by_month["amountfinancedByXepelin_prediction"]=np.exp(model.predict(start=0, end=(len(amount_by_month)), dynamic=False))
amount_by_month["amountfinancedByXepelin_prediction"]=amount_by_month["amountfinancedByXepelin_prediction"].round(2)

real=amount_by_month["amountfinancedByXepelin"]
pred=amount_by_month["amountfinancedByXepelin_prediction"]

plt.figure(figsize=(6,6))
plt.plot(pred,label="Prediction",color="#fc766aff",marker='o',markersize=3)
plt.plot(real, color='#a5a5a5ff',label="Real amount",marker='o',markersize=3)
plt.legend()
plt.xticks(fontsize=12,alpha=1,rotation=30)
plt.box(False)
plt.savefig(prediction_img_path)
plt.show()

# Save Predictions
amount_by_month.to_csv(csv_prediction_path)
print(amount_by_month)
print("Predictions Done!")


