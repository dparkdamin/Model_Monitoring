import pandas as pd
import numpy as np
import statistics

from imblearn.over_sampling import SMOTE
    
data = pd.read_csv('creditcard.csv')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=440)
# Total: 284,807 counts, 172,792 seconds (~ 48 hours)
# Train: 153,942 counts, 100,792 seconds(~ 28 hours)
# Test 1: 53,744 counts, 36,000 seconds (10 hours)
# Test 2: 77,121 counts, 36,000 seconds(10 hours)
# Sample: 65,432 counts (avg of Test 1 and Test 2)
train_end_index = int(data[data['Time']==100774].index.values) #153941
test1_end_index = int(data[data['Time']==136792].index.values[-1]) #207685

# y is fraud or not. X is  everything else 
X = data.drop(columns = ['Class', 'Time'])
y = data['Class']

X_train = X[0:train_end_index + 1]
X_test1 = X[train_end_index + 1:test1_end_index + 1]
X_test2 = X[test1_end_index + 1:]
y_train = y[0:train_end_index + 1]
y_test1 = y[train_end_index + 1:test1_end_index + 1]
y_test2 = y[test1_end_index + 1:]

# creating randomly sampled test batch with added noise (3%)
test = data[train_end_index + 1:]
test = test.drop(columns = 'Time')
avg_test_length = round(statistics.mean([len(X_test1), len(X_test2)]))
# #adding noise
noise = test.sample(n = avg_test_length, replace = True)
noise_mod = np.random.binomial(1, 0.03, noise.shape[0])
noise['Class'] = abs(np.subtract(noise['Class'].values, noise_mod))

X_noise = noise.drop(columns = ["Class"])
y_noise = noise['Class']

#oversampling training data
sm = SMOTE(random_state = 123, sampling_strategy=0.3) # oversamples to 3:10 ratio
X_train, y_train = sm.fit_resample(X_train, y_train)

amount = test.sample(random_state = 440, n = avg_test_length, replace = True)
# Transaction amount is exponentially increased by 1.2
X_amount = amount.drop(columns = ["Class"])
X_amount['Amount'] = X_amount['Amount'].apply(lambda x: x**1.2)

y_amount = amount['Class']

X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)

X_test1.to_csv('X1_test.csv', index=False)
y_test1.to_csv('y1_test.csv', index=False)

X_test2.to_csv('X2_test.csv', index=False)
y_test2.to_csv('y2_test.csv', index=False)

X_noise.to_csv('X_noise.csv', index=False)
y_noise.to_csv('y_noise.csv', index=False)

X_amount.to_csv('X_amount.csv', index=False)
y_amount.to_csv('y_amount.csv', index=False)