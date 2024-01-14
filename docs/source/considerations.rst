Considerations
==============
.. topic:: Metrics

    For fraud detection, the most important evaluation metric is the model's ability to accurately detect fraud cases. Therefore, you should choose the evaluation metric that focuses on the minority class (i.e., the fraud cases).

    The most commonly used evaluation metrics for fraud detection are precision, recall, and F1-score. 
    - Precision measures the proportion of true positive predictions among all positive predictions
    - Recall measures the proportion of true positive predictions among all actual positive cases. 
    - F1-score is the harmonic mean of precision and recall, and it combines both metrics to give a single score, balances the trade-off between precision and recall.

    In general, a model with high precision and recall, meaning that it can accurately identify fraud cases while minimizing false positives (legitimate transactions incorrectly identified as fraud) and false negatives (fraudulent transactions incorrectly identified as legitimate).

    Evaluation metrics such as AUC-ROC, which measures the model's ability to distinguish between the positive and negative classes, and to perform cross-validation to estimate the model's generalization performance.

    Evaluation metrics: For highly imbalanced data, used metrics that focus on the minority class. Use metrics like precision, recall, and F1-score for evaluating the model's performance. Additionally, use AUC-ROC to evaluate the model's ability to distinguish between the positive and negative classes.

    Transformations: Perform data balancing techniques such as undersampling, oversampling, or SMOTE (Synthetic Minority Over-sampling Technique) to address the class imbalance issue.

    EDA: Explore the distribution of the target variable and identify any patterns or correlations between the features and the target variable. Additionally, check for missing values, outliers, and other data quality issues.

    Feature engineering: Create new features based on domain knowledge or transform existing features to enhance the predictive power of the model. For example, you can create features based on time intervals, aggregate statistics, or interaction terms.

    Model selection: Several models for fraud detection, including logistic regression, random forests, gradient boosting, and neural networks. You may want to experiment with different models and compare their performance based on the evaluation metrics.

    Approach: Given the highly imbalanced data, a recommended approach would be to use a combination of data balancing techniques, feature engineering, and model selection to build a robust fraud detection model. Additionally, use cross-validation to assess the model's performance and fine-tune the model's hyperparameters.


.. topic:: EDA Conclusions:

    1. The target feature having 2 class (0,1): 
        - is higly imbalanced with IsFraud=1 (No fraud transactions)
            No Frauds: 99.87091795518198
            Frauds: 0.12908204481801522

    2. The dataset has features with 3 datatypes:
        - 'int64': ['step', 'isFraud', 'isFlaggedFraud']
        - 'float64': ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        - 'object': ['type', 'nameOrig', 'nameDest']
        
    3. The count of unique class in each feature are
        step                  743
        type                    5
        amount            5316900
        nameOrig          6353307
        oldbalanceOrg     1845844
        newbalanceOrig    2682586
        nameDest          2722362
        oldbalanceDest    3614697
        newbalanceDest    3555499
        isFraud                 2
        isFlaggedFraud          2
    4. The type has 5 class:
        - ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'] which can be converted to numeric feature
    5. The dataset is positively skewed indicating there are outliers in the features:
        - amount +30 This also indicates an extremely positively skewed distribution, even more so than the previous value. The distribution has an extremely long tail on the positive side of the mean, indicating that there may be a very large number of outliers or extreme values in the data.
        - oldbalanceOrg and newbalanceOrig +19: This indicates an extremely positively skewed distribution. The distribution has an extremely long tail on the positive side of the mean, indicating that there may be a large number of outliers or extreme values in the data.
    6.  Correlation
        - 'amount' has the highest absolute correlation with 'isFraud', with a coefficient of 0.076. This suggests that there may be some relationship between the transaction amount and the likelihood of fraud, but the correlation is relatively weak.
        - The other numerical variables ('oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', and 'newbalanceDest') have correlations with 'isFraud' ranging from -0.005 to 0.0005, indicating that there is little to no linear relationship between these variables and the likelihood of fraud.
    7. Based on the contingency table, we can observe that fraudulent transactions only occurred in the 'CASH_OUT' and 'TRANSFER' types of transactions. 
        - This can be interpreted by looking at the row with 'isFraud' equal to 1, where we see that there are no fraudulent transactions in the other three types.
        - The chi-squared test statistic of 22082.5357 indicates that there is a statistically significant association between the 'type' and 'isFraud' variables. The p-value of 0.0 indicates that the association is not likely due to chance.
        - Cramer's V coefficient of 861709.8685 and its corresponding p-value of 1.0434e-05 also suggest a strong association between the two variables. Cramer's V is a measure of association between two nominal variables, where a value of 0 indicates no association and a value of 1 indicates a perfect association. The small p-value suggests that the association is statistically significant.

    .. image:: _static/eda/pie-transactiontype.png
    :alt: Transaction proportion

    .. image:: _static/eda/class_imbalance.png
    :alt: Class imbalance (0: no fraud, 1: is fraud)
    