# Bank_customer_churn_prediction
Every bank wants to hold there customers for sustaining their business so the ABC Multinational bank.
Below is the customer data of account holders at ABC Multinational Bank and ""the aim of the data will be predicting the Customer Churn.""

Features used:

customer_id, unused variable.
credit_score, used as input.
country, used as input.
gender, used as input.
age, used as input.
tenure, used as input.
balance, used as input.
products_number, used as input.
credit_card, used as input.
active_member, used as input.
estimated_salary, used as input.
churn, used as the target. 1 if the client has left the bank during some period or 0 if he/she has not.

the dataset consists 12 columns and 9930 rows

Done Exploratory Data analysis to find out the important features and their relations

Used plots like KDE plot and pair plots to understand the relations between features

used the following libraries in the Data Pre-Processing Section
*Numpy
*Pandas
*seaborn
*Matplotlib

By understanding the problem statement properly we can understand it is a classification problem
## Classification_problem

So We have to change all the Object values to other form of numerical values

#### object_values -> Country
France  -> 1
Spain   -> 2
Germany -> 3

#### Similarly for Gender 
Male  -> 1
Female -> 0

Hence we changed all our features into some or other form of numericals which would be suitable 
for classification problem.

### Modeling
Used Sklearn libraries for machinelearning modeling
Applied different Classification algorithms which are given below along with their accuracy

	                  Accuracy
Naive Bayes 	      80.306081

KNN         	      76.318969

SVM	                80.306081

Random_forest	      79.540878

DecisionTree	      78.453484

LogisticRegression	79.540878

#### We are getting maximum accuracy from Naive Bayes And SVM










