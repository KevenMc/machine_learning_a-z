"""Upper bound confidence model"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Name data source
DATA_SOURCE = 'Ads_CTR_Optimisation.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)

#Implement UBC
N = 750
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if (n_s:=numbers_of_selections[i]) > 0:
            average_reward = sum_of_rewards[i] / n_s
            delta_i = math.sqrt(3/2 * math.log(n+1)/ n_s)
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_reward += reward


#Visualise the results
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times aeach ad was selected")
plt.show()
