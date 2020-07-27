# Supervised-Machine-Learning

## Challenge 17 Summary 
We used the imbalanced-learn library to resample the data and we build and evaluate logistic regression classifiers using the resampled data.  We used Oversample the data using the RandomOverSampler and SMOTE algorithms, for undersample the data we used the cluster centroids algorithm and finally we used a combination approach with the SMOTEENN algorithm.

We did for each resample model:
- Train a logistic regression classifier (from Scikit-learn) using the resampled data.
- Calculate the balanced accuracy score using balanced_accuracy_score from sklearn.metrics.
- Generate a confusion_matrix.
- Print the classification report (classification_report_imbalanced from imblearn.metrics).

In our report we will determine which model fot better our data set analyzing the balanced accuracy score, the confusion matrix and classification report.

## Files
- We used Python and jupyter notebook.  
- File name: credit_risk_resampling.IPYNB

## Analysis
We were using a dataset about loan statistics.  We read the data and did some cleaning, we encoded the data in columns with strings and convert to binary code using the get_dummies() method from pandas.  After the cleaning was donde we created the X_train, X_test, y_train and y_test datasets to be used in the resample.  We started with oversampling, the Naive Random Oversampling.

### Naive Random Oversampling
We resample the training data with the RandomOversampler, then we trained a logistic regression classifier using the resample data.
Summary of the results:
- Balanced_accuracy_score: 65%
- Confusion_matrix:
  - [70, 31],
  - [6711, 10393]
- Classification_report_imbalanced:
  -                   pre       rec       spe        f1       geo       iba       sup

  -  high_risk       0.01      0.74      0.63      0.02      0.68      0.47       101
  -   low_risk       1.00      0.63      0.74      0.77      0.68      0.46     17104

-  avg / total       0.99      0.63      0.74      0.77      0.68      0.46     17205


