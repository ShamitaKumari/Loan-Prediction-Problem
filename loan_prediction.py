#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt


# In[2]:


#reas files
loan_file=pd.read_csv('/python programs/loan_train.csv')


# In[3]:


loan_file.head(5) #showing first 5 entries from dataset


# In[4]:


loan_file.tail(5)# last 5 entries


# In[5]:


loan_file.describe() #get summary


# In[6]:


#total number of missing values in dataframe
loan_file.isnull().sum(axis=0)


# In[7]:


loan_file['LoanAmount'].mean()


# In[ ]:





# In[8]:


#visualization of distribution
#lets plot the loan amount and income
loan_file['ApplicantIncome'].hist(bins=50)
plt.title('ApplicantIncome Analysis')
plt.xlabel('Amount')


# In[9]:


#boxplot to understand the destribution
loan_file.boxplot(['ApplicantIncome'])
plt.show()
'''it shows there are too many outliers present in this columns. it shows the income disparity'''


# In[10]:


#income check on the basis of education level
loan_file.boxplot(['ApplicantIncome'] , by=['Education'])
plt.show()


# In[11]:


loan_file['LoanAmount'].hist(bins=50)


# In[12]:


loan_file.boxplot(['LoanAmount'])


# In[13]:


#pivot table
temp1=loan_file['Credit_History'].value_counts()
temp2=loan_file.pivot_table(values='Loan_Status', index=['Credit_History'],aggfunc=lambda x:x.map({'Y':1,'N':0}).mean())
print('frequncy table of credit history')
print(temp1)
print('\nprobability of getting loan from each credit history')
print(temp2)


# In[14]:


#visual representation /bar chart of credit_history
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('count of Applicants')
ax1.set_title('Applicants by Credit_History')
temp1.plot(kind='bar')


# In[15]:



#prbability of getting loan
ax2=fig.add_subplot(1,2,2)
temp2.plot(kind='bar')
ax2.set_xlabel('credit_History')
ax2.set_ylabel("prbability of getting loan")
ax2.set_title('probability of getting loan by credit_history')

'''we can plot simmilar graphs by Gender, married and property_area'''


# In[16]:


temp3 = pd.crosstab(loan_file['Credit_History'], loan_file['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[17]:


#data muging
#check missing values in fields

loan_file.isnull().sum(axis=0)


# In[67]:


#fill missing values

loan_file['Self_Employed'].fillna('No', inplace=True)
 


# In[72]:


#pivot table
temp3=loan_file['LoanAmount'].value_counts()
loan_file.apply(lambda x: sum(x.isnull()),axis=0)

table = loan_file.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
print(table)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
loan_file['LoanAmount'].fillna(loan_file['LoanAmount'].median, inplace=True)


# In[ ]:





# In[63]:


loan_file['LoanAmount_log'] = np.log(loan_file['LoanAmount'])
loan_file['LoanAmount_log'].hist(bins=20)
"""extreme value distribution on loanAmount and Applicant Income, It may be possible because some people need high amount loans according to their need"""


# In[21]:


loan_file['TotalIncome'] = loan_file['ApplicantIncome'] + loan_file['CoapplicantIncome'] #some applicants may have a good support of co-applicant income.So, we can combine them to find out total income and transformation for the same
loan_file['TotalIncome_log'] = np.log(loan_file['TotalIncome'])
loan_file['LoanAmount_log'].hist(bins=20) 


# In[ ]:


#impute the missing values for Gender, Married, Dependents, Loan_Amount_Term, Credit_History.


# In[22]:


#convert all categorical values into numerice because sklearn needs imputs to be numeric
loan_file['Gender'].fillna(loan_file['Gender'].mode()[0], inplace=True)
loan_file['Married'].fillna(loan_file['Married'].mode()[0], inplace=True)
loan_file['Dependents'].fillna(loan_file['Dependents'].mode()[0], inplace=True)
loan_file['Loan_Amount_Term'].fillna(loan_file['Loan_Amount_Term'].mode()[0], inplace=True)
loan_file['Credit_History'].fillna(loan_file['Credit_History'].mode()[0], inplace=True)


# In[23]:


from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    loan_file[i] = le.fit_transform(loan_file[i])
loan_file.dtypes 


# In[45]:


#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold (n_splits=5)
  error = []

    
  for train, test in kf.split(data[predictors]):
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 


# In[46]:


#logistic Regression
#with credit history
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, loan_file,predictor_var,outcome_var)


# In[47]:


#We can try different combination of variables:
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model,loan_file,predictor_var,outcome_var)


# In[48]:


#decision tree
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, loan_file,predictor_var,outcome_var)


# In[49]:


#random_forest
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, loan_file,predictor_var,outcome_var)


# In[ ]:




