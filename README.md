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

## Results
We were using a dataset about loan statistics.  We read the data and did some cleaning, we encoded the data in columns with strings and convert to binary code using the get_dummies() method from pandas.  After the cleaning was donde we created the X_train, X_test, y_train and y_test datasets to be used in the resample.  We started with oversampling, the Naive Random and SMOTE, then undersampling using the ClusterCentroids resampler, and finally we did a combination (Over and Under) sampling using SMOTEENN.

## Oversampling
### Naive Random Oversampling
We resample the training data with the RandomOversampler, then we trained a logistic regression classifier using the resample data.
Summary of the results:
- Balanced_accuracy_score: 0.65 (LogisticRegression(max_iter=100)
- Balanced_accuracy_score: 0.69 (LogisticRegression(max_iter=200)
- Confusion_matrix:
  - [70, 31],
  - [6711, 10393]
- Classification_report_imbalanced:
  - ![Naive Random Oversampling](https://github.com/DahianaMC/Supervised-Machine-Learning/blob/master/Naive%20Random%20Oversampling.PNG)



### SMOTE Oversampling
We used another oversampling model to resample the training data with SMOTE, then we trained a logistic regression classifier using the resample data.
Summary of the results:
- Balanced_accuracy_score: 0.66 (LogisticRegression(max_iter=100)
- Balanced_accuracy_score: 0.68 (LogisticRegression(max_iter=200)
- Confusion_matrix:
  - [64, 37],
  - [5291, 11813]
- Classification_report_imbalanced:
  - ![SMOTE Oversampling](https://github.com/DahianaMC/Supervised-Machine-Learning/blob/master/SMOTE%20Oversampling.PNG)


## Undersampling
### ClusterCentroids Resampler
We resample the training data with the ClusterCentroids Resampler, then we trained a logistic regression classifier using the resample data.
Summary of the results:
- Balanced_accuracy_score: 0.55 (LogisticRegression(max_iter=100)
- Balanced_accuracy_score: 0.58 (LogisticRegression(max_iter=200)
- Confusion_matrix:
  - [69, 32],
  - [10075, 7029]
- Classification_report_imbalanced:
  - ![ClusterCentroids resampler](https://github.com/DahianaMC/Supervised-Machine-Learning/blob/master/ClusterCentroids%20resampler.PNG)


## Combination (Over and Under) Sampling
### SMOTEENN
We resample the training data with SMOTEEN, then we trained a logistic regression classifier using the resample data.
Summary of the results:
- Balanced_accuracy_score: 0.68 (LogisticRegression(max_iter=100)
- Balanced_accuracy_score: 0.68 (LogisticRegression(max_iter=200)
- Confusion_matrix:
  - [79, 22],
  - [7309, 9795]
- Classification_report_imbalanced:
  - ![SMOTEENN](https://github.com/DahianaMC/Supervised-Machine-Learning/blob/master/SMOTEENN.PNG)

## Analysis of Results
- Precision-Recall is a measure of success of prediction when classes are very imbalanced, precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned.  High precision relates to a low false positive rate, high recall relates to a low false negative rate.  If we have high scores for both show that the classifier is returning accurate results(high precision), as well as returning a majority of all positive results (high recall).

- A system with high recall but low precision returns many results, but most of its predicted labels are incorrect when compared to the training labels. A system with high precision but low recall is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels. An ideal system with high precision and high recall will return many results, with all results labeled correctly.

- The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. 

- Comparing the results from the oversampling, undersampling and combination, the balanced_accuracy_score for oversampling and combination was in average 0.67, while for undersampling was 0.55.  The balanced_accuracy_score is a average of the recall obtained on each class.

- The classification_report_imbalanced shows for all the resample models the precision of the high_risk class is 0.01 and the precision for the low_risk class is 1.0.  For the high risk_class since the precision is very low, relates to a high false positive rate.  Looking only the precision none of the models will predict the high_risk loans.

- Recall values for all the resample models for the high_risk class were: 
  - Naive Random Oversampling: 0.69
  - SMOTE: 0.63
  - Undersampling ClusterCentroids: 0.68
  - Combination: 0.78
 -Combination showed the higher recall.

- Recall values for all the resample models for the low_risk class were: 
  - Naive Random Oversampling: 0.61
  - SMOTE: 0.69
  - Undersampling ClusterCentroids: 0.41
  - Combination: 0.57
- Undersamplig showed the lowest recall.

- Since our high_risk class has a very low precision and high recall we expect many results, but most of its predicted labels are incorrect when compared to the training labels.  

- For the low_risk class the precision is very high and recall is low for undersampling and combination returning very few results, but most of its predicted labels are correct when compared to the training labels.

- Change the max_iter for the logistic regression model from 100 to 200, make the balanced_accuracy_score increases about 0.02 to 0.04 for the oversampling and undersampling algorithms, for the combination algorithm did not make any difference.  We also tried 300 but did not make any difference.

- An ideal system with high precision and high recall will return many results, with all results labeled correctly.  If we have high scores for precision and recall the classifier will return accurate results(high precision), as well as returning a majority of all positive results (high recall).  Since the precision is low for the high risk class and recall is less than 0.69 for all the models, I recommend to look other models (algorithms) that fit better the dataset that was used for training the model.  For low_risk the precision was 1, and higher recall was for Smote (0.69), but since the model does not fit the high risk class, I recommend to find another model (algorithm) that fits better both classes. 
