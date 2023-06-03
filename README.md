# Bank Customer Churn Prediction

This code is a Python implementation of a bank customer churn prediction project. The goal of the project is to predict whether a bank customer is likely to churn or not based on various features such as credit score, tenure, active membership, age, country, gender, credit card usage, estimated salary, balance, and number of products.

## Dataset

The dataset used for this project is the "Bank Customer Churn Prediction" dataset. It is a CSV file containing information about bank customers and their churn status.

## Insights from the Code

- Exploratory Data Analysis (EDA): The EDA performed on the dataset revealed the following insights:
  - The distribution of the credit score is approximately normal.
  - There is a negative correlation between age and churn, indicating that older customers are less likely to churn.
  - Customers from different countries have varying churn rates, with fewer churn customers from Spain compared to France and Germany.
  - Customers with lower credit scores are more likely to churn.
  - Females have a higher churn rate compared to males.
  - Active members are more likely to be churn customers.

- Model Evaluation: Several classification models were trained and evaluated for their predictive performance:
            
     - Naive Bayes 80.829642
     - KNN 76.761981
     - SVM 80.829642
     - Random_forest 79.661700
     - DecisionTree 79.420056
     - LogisticRegression 79.661700

Based on the evaluation, the Naive Bayes and Support Vector Machine models demonstrated the highest accuracy in predicting customer churn.

## Usage

1. Install the required dependencies: pandas, matplotlib, seaborn, numpy, scikit-learn.
2. Load the dataset using `pd.read_csv()` function.
3. Perform exploratory data analysis to gain insights into the dataset.
4. Preprocess the data (if required) by handling missing values, encoding categorical variables, and scaling numerical variables.
5. Split the dataset into training and testing sets using `train_test_split()` function from scikit-learn.
6. Train the chosen classification model(s) on the training set.
7. Evaluate the model(s) on the testing set using appropriate evaluation metrics such as accuracy and mean absolute error.
8. Choose the best-performing model based on the evaluation results.
9. Use the chosen model to make predictions on new, unseen data.

The insights gained from this project can help the bank in identifying potential churn customers and taking proactive measures to retain them.
