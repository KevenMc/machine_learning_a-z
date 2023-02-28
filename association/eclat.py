"""Eclat association learning model"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from apyori import apriori
from typing import List


# Name data source
DATA_SOURCE = 'Market_Basket_Optimisation.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE, header=None).astype(str)
transactions = dataset.values.tolist()

# Train apriori model from apyori
rules = apriori(transactions=transactions, min_support=0.003,
                min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# Visualise results
results = list(rules)


def inspect(results: List):
    """Convert results from apriori into a well formatted list"""
    lhs = [tuple(result[2][0][0]) for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


resultsDF = pd.DataFrame(inspect(results), columns=[
                         "Product 1", "Product 2", "Support"])
resultsDF = resultsDF.sort_values(
    by="Support", ascending=False, ignore_index=True)
print(resultsDF[:20])
