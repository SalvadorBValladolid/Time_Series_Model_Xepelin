import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import pandas as pd
from utils import *


# Parameters
dataset_path = "data/data.csv"
processed_dataset_path="data/data_processed.csv"

# Load raw data
raw_data=pd.read_csv(dataset_path)

# Preprocess raw data

raw_data=quit_missing_transactions(raw_data)

raw_data=convert_to_datetime(raw_data)

raw_data=extract_month_transaction(raw_data)

processed_data=aggregate_amount_by_month(raw_data)

# Save the processed data

processed_data.to_csv(processed_dataset_path)
print("Processed raw data: Done!")








