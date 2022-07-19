import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import quit_missing_transactions,convert_to_datetime,extract_month_transaction

# Parameters
dataset_path = "data/data.csv"

# Load raw data
data=pd.read_csv(dataset_path)

data=convert_to_datetime(data)

data["Porc_Amount_financiated"]=data["amountfinancedByXepelin"]/data["amount"]

# Analysis

print("Distribution type TXN:")
print(data["status"].value_counts())

# Filter only completed TXN
data=quit_missing_transactions(data)

# Extract month of the TXN
data=extract_month_transaction(data)

# Payer analysis
empresas=pd.DataFrame(data.groupby("PayerId")["amountfinancedByXepelin"].sum()).reset_index()
empresas=empresas.sort_values(by="amountfinancedByXepelin",ascending=False)
empresas["Cum_sum_amount"]=empresas["amountfinancedByXepelin"].cumsum()
empresas["Porc_cum_sum"]=empresas["Cum_sum_amount"]/empresas["amountfinancedByXepelin"].sum()
empresas=empresas.reset_index()

print(empresas[0:31])

# Generate images
print(data[(data["amountfinancedByXepelin"]>0)]["amountfinancedByXepelin"].describe().round())
print(data["amount"].describe().round())



# Amount financiated by Xepelin
plt.figure(figsize=(16,10))
plt.hist(data[(data["amountfinancedByXepelin"]>0)&(data["amountfinancedByXepelin"]<=57157.0)]["amountfinancedByXepelin"], density=False, bins=30,color="#fc766aff")  
plt.xlabel('Monto Financiado por Xepelin',fontsize=15,alpha=0.6)
plt.ylabel('Frecuencia',fontsize=15,alpha=0.6)
plt.xticks(fontsize=15,alpha=0.6)
plt.yticks(fontsize=15,alpha=0.6)
plt.title('',fontsize=15,alpha=0.6)
plt.box(False)
plt.savefig("img/amountfinancedByXepelin.png")

# Amount TXN

plt.figure(figsize=(16,10))
plt.hist(data[(data["amount"]>0)&(data["amount"]<=5541)]["amount"], density=False, bins=20,color="#fc766aff")  
plt.xlabel('Monto',fontsize=15,alpha=0.6)
plt.ylabel('Frecuencia',fontsize=15,alpha=0.6)
plt.xticks(fontsize=15,alpha=0.6)
plt.yticks(fontsize=15,alpha=0.6)
plt.title('',fontsize=15,alpha=0.6)
plt.box(False)
plt.savefig("img/amount.png")


amount_by_month=pd.DataFrame(data.groupby("paidAt_Month")["amount"].sum())
financiated_amount_by_month=pd.DataFrame(data.groupby("paidAt_Month")["amountfinancedByXepelin"].sum())

# Time serie amount
plt.figure(figsize=(6,6))
plt.plot(amount_by_month,label="Amount",color="#a5a5a5ff",marker='o',markersize=3)
plt.xticks(fontsize=12,alpha=1,rotation=30)
plt.box(False)
plt.savefig("img/serie_amount.png")

# Time serie amount financiated by Xepelin
plt.figure(figsize=(6,6))
plt.plot(financiated_amount_by_month,label="Amount financiated by Xepelin",color="#a5a5a5ff",marker='o',markersize=3)
plt.xticks(fontsize=12,alpha=1,rotation=30)
plt.box(False)
plt.savefig("img/serie_amount_financiated_by_xepelin.png")

print("Done!")