import numpy as np
import pandas as pd

# Preprocessing functions

def quit_missing_transactions(data):
    data=data[~data["paidAt"].isna()]
    data=data=data[data["status"]=="PAID"]
    return data

def convert_to_datetime(data):
    data["paidAt"]=pd.to_datetime(data["paidAt"])
    return data

def extract_month_transaction(data):
    data["paidAt_Month"]=pd.to_datetime(data["paidAt"].dt.strftime('%Y-%m'))
    return data

def aggregate_amount_by_month(data):
    amount_by_month=pd.DataFrame(data.groupby("paidAt_Month")["amountfinancedByXepelin"].sum())
    return amount_by_month