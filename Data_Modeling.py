import re
import pandas as pd
import numpy as np

from flaml import AutoML

import sklearn.metrics as metrics

from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import BinaryClassificationTestPreset

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

X1_test = pd.read_csv('X1_test.csv')
y1_test = pd.read_csv('y1_test.csv')

X2_test = pd.read_csv('X2_test.csv')
y2_test = pd.read_csv('y2_test.csv')

X_noise = pd.read_csv('X_noise.csv')
y_noise = pd.read_csv('y_noise.csv')

X_amount = pd.read_csv('X_amount.csv')
y_amount = pd.read_csv('y_amount.csv')

# change from DataFrame to Series for AutoML
y_train = y_train.squeeze()

#Using FLAML - AutoML

# Initialize an AutoML instance
automl = AutoML()
# Specify AutoML goal and constraint
automl_settings = {
    "time_budget": 30, # seconds
    "metric": 'f1',
    "task": 'classification',
    "log_file_name": "automl.log",
}
# Train with labeled input data
automl.fit(X_train, y_train, **automl_settings)

# Predict Test Batch 1
pred1 = automl.predict(X1_test)

# Predict Test Batch 2
pred2 = automl.predict(X2_test)

# Predict Noise Batch
pred_noise = automl.predict(X_noise)

# Predict Amount Batch
pred_amount = automl.predict(X_amount)

# preparing data for Evidently
pred1 = pd.DataFrame(pred1)
trueTest1 = pd.concat([X1_test, y1_test], axis=1)
predTest1 = pd.concat([X1_test, pred1], axis=1)
trueTest1.rename(columns={'Class': 'target'}, inplace=True)

pred2 = pd.DataFrame(pred2)
trueTest2 = pd.concat([X2_test, y2_test], axis=1)
predTest2 = pd.concat([X2_test, pred2], axis=1)
trueTest2.rename(columns={'Class': 'target'}, inplace=True)

pred_noise = pd.DataFrame(pred_noise)
trueTest_noise = pd.concat([X_noise, y_noise], axis=1)
predTest_noise = pd.concat([X_noise, pred_noise], axis=1)
trueTest_noise.rename(columns={'Class': 'target'}, inplace=True)

pred_amount = pd.DataFrame(pred_amount)
trueTest_amount = pd.concat([X_amount, y_amount], axis=1)
predTest_amount = pd.concat([X_amount, pred_amount], axis=1)
trueTest_amount.rename(columns={'Class': 'target'}, inplace=True)

test1 = trueTest1
test1['prediction'] = predTest1[0]

test2 = trueTest2
test2['prediction'] = predTest2[0]

test_noise = trueTest_noise
test_noise['prediction'] = predTest_noise[0]

test_amount = trueTest_amount
test_amount['prediction'] = predTest_amount[0]

classification_performance = Report(metrics=[
    ClassificationPreset()
])

classification_performance.run(reference_data=test1, current_data=test2)
classification_performance.save_html("Test1_Test2_Report.html")

classification_performance.run(reference_data=test2, current_data=test_noise)
classification_performance.save_html("Test2_Noise_Report.html")

classification_performance.run(reference_data=test2, current_data=test_amount)
classification_performance.save_html("Test2_Amount_Report.html")

label_binary_classification_performance = TestSuite(tests=[
    BinaryClassificationTestPreset(),
])

label_binary_classification_performance.run(reference_data=test1, current_data=test2)
label_binary_classification_performance.save_html("Test1_Test2_Test.html")

label_binary_classification_performance.run(reference_data=test2, current_data=test_noise)
label_binary_classification_performance.save_html("Test2_Noise_Test.html")

label_binary_classification_performance.run(reference_data=test2, current_data=test_amount)
label_binary_classification_performance.save_html("Test2_Amount_Test.html")