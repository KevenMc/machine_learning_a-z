"""Implement thompson sampling algorithm"""

import random
import pandas as pd
import matplotlib.pyplot as plt

# Name data source
DATA_SOURCE = 'Ads_CTR_Optimisation.csv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE)


# Implement Thompson Sampling
N = 500
D = 10
ads_selected = []
number_of_rewards_1 = [0] * 10
number_of_rewards_0 = [0] * 10
TOTAL_REWARD = 0

for n in range(0, N):
    AD = 0
    MAX_RANDOM = 0

    for i in range(0, D):
        RANDOM_BETA = random.betavariate(
            number_of_rewards_1[i]+1, number_of_rewards_0[i]+1)
        if RANDOM_BETA > MAX_RANDOM:
            MAX_RANDOM = RANDOM_BETA
            AD = i

    ads_selected.append(AD)
    reward = dataset.values[n, AD]
    if reward:
        number_of_rewards_1[AD] +=1
    else:
        number_of_rewards_0[AD] += 1
    TOTAL_REWARD += reward

# Visualise the results
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()
