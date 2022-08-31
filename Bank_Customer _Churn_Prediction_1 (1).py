#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df = pd.read_csv('Bank Customer Churn Prediction.csv')


# In[521]:


df.head()


# In[520]:


df.info()


# In[5]:


df.describe()


# In[8]:


sns.kdeplot(data = df.credit_score,shade=True)


# In[519]:


sns.jointplot(data=df,x=df.churn,y=df.tenure)


# In[11]:


df.columns


# In[31]:


sns.histplot(x=df.age,y=df.credit_score)


# In[34]:


sns.histplot(x=df.age,y=df.estimated_salary,bins=25)


# In[64]:


kd = df.select_dtypes(include=['int64'])
kd
kd.drop(columns=['customer_id'],inplace=True)


# In[47]:


np.corrcoef(df.age,df.churn)


# In[65]:


sns.pairplot(kd)
plt.figure(figsize = (20,10))


# ### Check outliers in important features
#     

# In[404]:


sns.boxplot(x=df['estimated_salary'])


# In[405]:


sns.boxplot(x=df['balance'])


# In[406]:


sns.boxplot(x=df['age'])


# In[408]:


df[df.age>70].age.value_counts()


# In[428]:


df.shape


# In[429]:


df.columns


# In[430]:


df.country.value_counts()


# In[431]:


df.products_number[df.churn == 1].value_counts()


# In[432]:


df.products_number.value_counts()


# In[433]:


df.gender[df.churn == 1].value_counts()


# ### More number of females are founded to be churn customers

# In[434]:


df[df.churn==1].groupby('country')['products_number','credit_score','age','tenure',].mean()


# In[435]:


df.country[df.churn==1].value_counts()


# ### Comparitively less number of churn customers from spain almost half from other 2 countries

# In[436]:


df.credit_score[df.churn==1].value_counts()


# #### Possibility that people with low credit score are less likely to be churn customers

# In[437]:


df[df.churn==1].groupby('gender')['products_number','credit_score','age','tenure',].mean()


# In[438]:


df[df.churn==0].groupby('gender')['products_number','credit_score','age','tenure',].mean()


# In[439]:


df.active_member[df.churn==1].value_counts()


# In[440]:


df['gender'].replace(['Male','Female'],[1,0],inplace=True)

df


# In[485]:


df['country'].replace(['France','Spain','Germany'],[1,2,3],inplace=True)
df


# #### In active members can be considered as churn customers

# Now we will extract our features for our model

# In[488]:


X = df[['credit_score','tenure','active_member','age','country','gender','credit_card','estimated_salary','balance','products_number']]
X
#


# ### Logistic regression

# In[489]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)


# In[490]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X,y)


# In[491]:


model.fit(train_x,train_y)


# In[492]:


y_predict = model.predict(test_x)
y_predict[0:10]


# In[493]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test_y,y_predict)
mae


# In[494]:


from sklearn.metrics import accuracy_score
acs = accuracy_score(test_y,y_predict)
acs


# In[ ]:





# ### Decision Tree Classification

# In[495]:


from sklearn.tree import DecisionTreeClassifier
model_dtc = DecisionTreeClassifier(random_state=0)
model_dtc


# In[496]:


model_dtc.fit(train_x,train_y)


# In[497]:


y_pred_dtc = model_dtc.predict(test_x)
y_pred_dtc


# In[498]:


mae_dtc = mean_absolute_error(test_y,y_pred_dtc)
mae_dtc


# In[499]:


acs_dtc = accuracy_score(test_y,y_pred_dtc)
acs_dtc


# In[ ]:





# ### Randomforest classifier

# In[500]:


from sklearn.ensemble import RandomForestClassifier
model_rfc = RandomForestClassifier(random_state=1)


# In[501]:


model_rfc.fit(train_x,train_y)


# In[502]:


y_pred_rfc = model.predict(test_x)


# In[503]:


mae_rfc = mean_absolute_error(test_y,y_pred_rfc)
mae_rfc


# In[504]:


acs_rfc = accuracy_score(test_y,y_pred_rfc)
acs_rfc


# ### Support Vector Machine

# In[505]:


from sklearn.svm import SVC
model_svm = SVC()


# In[506]:


model_svm.fit(train_x,train_y)


# In[507]:


y_pred_svm = model_svm.predict(test_x)


# In[508]:


mae_svm = mean_absolute_error(test_y,y_pred_svm)
mae_svm


# In[509]:


acs_svm = accuracy_score(test_y,y_pred_svm)
acs_svm


# ### KNN

# In[510]:


from sklearn.neighbors import KNeighborsClassifier
model_knc = KNeighborsClassifier()


# In[511]:


model_knc.fit(train_x,train_y)


# In[512]:


y_pred_knc = model_knc.predict(test_x)


# In[513]:


mae_knc = mean_absolute_error(test_y,y_pred_knc)
mae_knc


# ### Naive Bayes

# In[514]:


from sklearn.naive_bayes import BernoulliNB
model_nb = BernoulliNB()


# In[515]:


model_nb.fit(train_x,train_y)


# In[516]:


y_p_nb = model_nb.predict(test_x)


# In[517]:


mae_nb = mean_absolute_error(test_y,y_p_nb)
mae_nb


# In[ ]:





# In[518]:


ddd = pd.DataFrame({
    'Naive Bayes': (1-mae_nb)*100,
    'KNN':(1-mae_knc)*100,
    'SVM':acs_svm*100,
    'Random_forest':acs_rfc*100,
    'DecisionTree':acs_dtc*100,
    "LogisticRegression":acs*100
    
},index=['Accuracy'])
ddd.T


# ## We are getting maximum accuracy from Naive Bayes And SVM

# In[ ]:




