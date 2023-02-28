"""Upper bound confidence model"""

import math
import pandas as pd
import matplotlib.pyplot as plt

# Name data source
DATA_SOURCE = 'Ads_CTR_Optimisation.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)

# Implement UBC
N = 750
D = 10
ads_selected = []
numbers_of_selections = [0] * D
sum_of_rewards = [0] * D
TOTAL_REWARD = 0

for n in range(0, N):
    AD = 0
    MAX_UPPER_BOUND = 0
    
    for i in range(0, D):
        if (n_s := numbers_of_selections[i]) > 0:
            average_reward = sum_of_rewards[i] / n_s
            delta_i = math.sqrt(3/2 * math.log(n+1) / n_s)
            UPPER_BOUND = average_reward + delta_i
        else:
            UPPER_BOUND = 1e400

        if UPPER_BOUND > MAX_UPPER_BOUND:
            MAX_UPPER_BOUND = UPPER_BOUND
            AD = i
    ads_selected.append(AD)
    numbers_of_selections[AD] += 1
    reward = dataset.values[n, AD]
    sum_of_rewards[AD] += reward
    TOTAL_REWARD += reward


#Visualise the results
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()
