# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:42:20 2024

@author: Priyanka
"""


"""
It is vital for banks that customers put in long term fixed deposits as they 
use it to pay interest to customers and it is not viable to ask every customer 
if they will put in a long-term deposit or not. So, build a Logistic Regression
model to predict whether a customer will put in a long-term fixed deposit or 
not based on the different variables given in the data. The output variable in
the dataset is Y which is binary. 

Business Problem-
Q.What is the business objective?
TMarketing is a process by which companies create value for customers and 
build strong customer relationships in order to capture value from customers 
in return. Marketing campaigns are characterized by focusing on the customer 
needs and their overall satisfaction.Nevertheless, there are different 
variables that determine whether a marketing campaign will be successful 
or not. There are certain variables that we need to take into consideration 
when making amarketing campaign.
Predicting-whether-the-customer-will-subscribe-to-Term-Deposits

Q.Are there any constraints?
Although huge set of varibles have been given Varibles may be having poor 
collinearity.we need to check collinearity of each with target

""" 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
fd=pd.read_csv("C:/Data Set/bank_data.csv")
fd.dtypes
fd.columns="age","default","balance","housing","loan","duration","campaign","pdays","previous","poutfailure","poutother","poutsuccess","poutunknown","con_cellular","con_telephone","con_unknown","divorced","married","single","jobadmin","joblue_collar","jobentrepreneur","johousemaid","job_mgt","job_retired","joself_employed","jobservices","jostudent","jotechnician","job_unemployed","job_unknown","target"
fd.isnull().sum()
#There are zero null values
fd.describe()
#Avrage age is 40.93 ,min is 18 and max is 95
#The type of job the customers have. Let’s call the count plot function defined earlier to plot the count plot of the job feature.
import seaborn as sns
plt.figure(1, figsize=(16, 10))
sns.countplot(fd['jobadmin'])
sns.countplot(fd['joblue_collar'])
sns.countplot(fd['jobentrepreneur'])
sns.countplot(fd['johousemaid'])
sns.countplot(fd['job_mgt'])
sns.countplot(fd['job_retired'])
sns.countplot(fd['joself_employed'])
sns.countplot(fd['jobservices'])
sns.countplot(fd['jostudent'])
sns.countplot(fd['jobservices'])
sns.countplot(fd['jotechnician'])
sns.countplot(fd['job_unemployed'])
sns.countplot(fd['job_unknown'])
#There are more customers working as admin than any other profession.
#Now let us check the marital status
plt.figure(1, figsize=(16, 10))
sns.countplot(fd['divorced'])
sns.countplot(fd['married'])
sns.countplot(fd['single'])
#majority customers are divorced
#Default: Denotes if the customer has credit in default or not. The categories are yes, no and unknown
plt.figure(1, figsize=(16, 10))
sns.countplot(fd['default'])
#housing: Denotes if the customer has a housing loan. Three categories are ‘no’, ’yes’, ’unknown’.
sns.countplot(fd['housing'])
#poutcome: This feature denotes the outcome of the previous marketing campaign.
sns.countplot(fd['poutfailure'])
sns.countplot(fd['poutother'])
sns.countplot(fd['poutsuccess'])
#There are column names having spaces ,let us rename the columns

tc = fd.corr()
tc

fig,ax= plt.subplots()
fig.set_size_inches(40,20)
sns.heatmap(tc, annot=True, cmap='YlGnBu')
#From heat map it is clear that only duration,poutsuccess,con_cellular,pdays are highly correlated with target,remaining can be dropped
#We can infer that duration of the call,(outcome of the previous marketing campaign)poutsuccess is highly correlated with 
#the target variable. As the duration of the call is more, there are higher chances that the client is showing interest in the term deposit and hence there are higher chances that the client will subscribe to term deposit.
#con_cellular and pdays are upto some extent correlated
fd=fd.drop(["age","default","balance","housing","loan","campaign","previous","poutfailure","poutother","poutunknown","con_telephone","con_unknown","divorced","married","single","jobadmin","joblue_collar","jobentrepreneur","johousemaid","job_mgt","job_retired","joself_employed","jobservices","jostudent","jotechnician","job_unemployed","job_unknown"],axis=1)
#


#Let us re-arrange the columns
#DM=DM.iloc[:,[6,0,1,2,3,4,5]]
#Many columns have different scale values let us apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
fd1=norm_func(fd.iloc[:,:])
fd1.isna().sum()
#There are no null values
fd1.dtypes
#model bulding
logit_model=sm.logit('target ~ duration+ pdays+ poutsuccess+ con_cellular',data=fd1).fit()
logit_model.summary()
logit_model.summary2()
#let us go for prediction
pred=logit_model.predict(fd1.iloc[:,:4])

#To derive ROC curve
#ROC curve has tpr on y axis and fpr on x axis,ideally,tpr must be high
#fpr must be low
fpr,tpr,thresholds=roc_curve(fd1.target,pred)
#To identify optimum threshold
optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=thresholds[optimal_idx]
optimal_threshold
#0.0925219 ,by default you can take 0.5 value as a threshold
#Now we want to identify if new value is given to the model,it will
#fall in which region 0 or 1,for that we need to derive ROC curve
#To draw ROC curve
import pylab as pl
i=np.arange(len(tpr))
roc=pd.DataFrame({'fpr':pd.Series(fpr,index=i),'tpr':pd.Series(tpr,index=i),'1-fpr':pd.Series(1-fpr,index=i),'tf':pd.Series(tpr-(1-fpr),index=i),'thresholds':pd.Series(thresholds,index=i)})
#plot ROC curve
plt.plot(fpr,tpr)
plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc=auc(fpr,tpr)
print("Area under the curve %f"%roc_auc)

#Now let us add prediction column in dataframe
fd1["pred"]=np.zeros(45211)
fd1.loc[pred>optimal_threshold,"pred"]=1
#if predicted value is greater than optimal threshold then change pred column as 1
#Classification report
classification=classification_report(fd1["pred"],fd1["target"])
classification

#splitting the data into train and test data
train_data,test_data=train_test_split(fd1,test_size=0.3)
#model building using 
model=sm.logit('target ~ duration+ pdays+ poutsuccess+ con_cellular',data=train_data).fit()
model.summary()
model.summary2()
#AIC is 17110.0234
#prediction on test data
test_pred=model.predict(test_data)
test_data["test_pred"]=np.zeros(13564)
#taking threshold value as optimal threshold value
test_data.loc[test_pred>optimal_threshold,"test_pred"]=1
#Confusion_matrix
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.target)
confusion_matrix
accuracy_test=(9352+1225)/13564
accuracy_test
#Classification report
classification_test=classification_report(test_data["test_pred"],test_data["target"])
classification_test
#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(test_data["target"],test_pred)
#plot of ROC
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test
#prediction on train data
train_pred=model.predict(train_data.iloc[:,:4])
#creating new column
train_data["train_pred"]=np.zeros(31647)
train_data.loc[train_pred>optimal_threshold,"train_pred"]=1
#confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred,train_data.target)
confusion_matrix

#Accuracy test
accuracy_train=(21669+2898)/31647
accuracy_train
#classification report
classification_train=classification_report(train_data.train_pred,train_data.target)
classification_train

#ROC_AUC curve
roc_auc_train=metrics.auc(fpr,tpr)
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
