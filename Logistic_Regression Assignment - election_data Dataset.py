# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:43:37 2024

@author: Priyanka
"""
"""
Perform Logistic Regression on the dataset to predict whether a 
candidate will win or lose the election based on factors like amount 
of money spent and popularity rank. 

Business Problem-
Q.What is the business objective?
Suppose we are interested in the factors that influence whether a political 
candidate wins an election. The outcome (response) variable is binary (0/1); 
win or lose. The predictor variables of interest are the amount of money 
spent on the campaign, the amount of time spent campaigning 
negatively and whether or not the candidate is an incumbent.

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
election=pd.read_csv("C:/Data Set/election_data.csv")
election=election.drop(election.index[[0]])
election=election.drop(['Election-id'],axis=1)
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
election1=norm_func(election.iloc[:,:])
election1.columns='Result','Year','Amount_Spent','Popularity_Rank'


election1.isna().sum()
#There are no null values
tc = election1.corr()
tc
import seaborn as sns
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(tc, annot=True, cmap='YlGnBu')
#from heat map it is clear that popularity and results are poorly correlated
election1=election1.drop(['Popularity_Rank'],axis=1)

#model bulding
logit_model=sm.logit('Result~Year+Amount_Spent',data=election1).fit()
logit_model.summary()
logit_model.summary2()
#let us go for prediction
pred=logit_model.predict(election1.iloc[:,1:])

#To derive ROC curve
#ROC curve has tpr on y axis and fpr on x axis,ideally,tpr must be high
#fpr must be low
fpr,tpr,thresholds=roc_curve(election1.Result,pred)
#To identify optimum threshold
optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=thresholds[optimal_idx]
optimal_threshold
#0.0.25 ,by default you can take 0.5 value as a threshold
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
election1["pred"]=np.zeros(10)
election1.loc[pred>optimal_threshold,"pred"]=1
#if predicted value is greater than optimal threshold then change pred column as 1
#Classification report
classification=classification_report(election1["pred"],election1["Result"])
classification

#splitting the data into train and test data
train_data,test_data=train_test_split(election1,test_size=0.1)
#model building using 
model=sm.logit('Result~Year+Amount_Spent',data=train_data).fit()
model.summary()
model.summary2()
#AIC is 441.2892 
#prediction on test data
test_pred=model.predict(test_data)
test_data["test_pred"]=np.zeros(1)
#taking threshold value as optimal threshold value
test_data.loc[test_pred>optimal_threshold,"test_pred"]=1
#Confusion_matrix
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.Result)
confusion_matrix
#prediction on train data
train_pred=model.predict(train_data.iloc[:,1:])
#creating new column
train_data["train_pred"]=np.zeros(9)
train_data.loc[train_pred>optimal_threshold,"train_pred"]=1
#confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred,train_data.Result)
confusion_matrix

#Accuracy test
accuracy_train=(7)/9
accuracy_train
#classification report
classification_train=classification_report(train_data.train_pred,train_data.Result)
classification_train

#ROC_AUC curve
roc_auc_train=metrics.auc(fpr,tpr)
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
