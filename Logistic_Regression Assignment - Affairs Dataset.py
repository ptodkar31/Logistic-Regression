# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:43:13 2024

@author: Priyanka
"""

"""
A psychological study has been conducted by a team of students at a university
on married couples to determine the cause of having an extra marital affair.
They have surveyed and collected a sample of data on which they would like to 
do further analysis. Apply Logistic Regression on the data to correctly 
classify whether a given person will have an affair or not given the set of 
attributes. Convert the naffairs column to discrete binary type before
proceeding with the algorithm.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
affair=pd.read_csv("C:/Data Set/Affairs.csv")
affair.dtypes
ex_af=affair.drop('Unnamed: 0',axis=1)

ex_af.loc[ex_af.naffairs>0,"naffairs"]=1
ex_af.head()
ex_af.isna().sum()
#There are no null values

#model bulding
logit_model=sm.logit('naffairs~ kids+ vryunhap+ unhap+ avgmarr+ hapavg+ vryhap+ antirel+ notrel+ slghtrel+ smerel+ vryrel+ yrsmarr1+ yrsmarr2+ yrsmarr3+ yrsmarr4+ yrsmarr5+ yrsmarr6',data=ex_af).fit()
logit_model.summary()
logit_model.summary2()
#let us go for prediction
pred=logit_model.predict(ex_af.iloc[:,1:])

#To derive ROC curve
#ROC curve has tpr on y axis and fpr on x axis,ideally,tpr must be high
#fpr must be low
fpr,tpr,thresholds=roc_curve(ex_af.naffairs,pred)
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
ex_af["pred"]=np.zeros(601)
ex_af.loc[pred>optimal_threshold,"pred"]=1
#if predicted value is greater than optimal threshold then change pred column as 1
#Classification report
classification=classification_report(ex_af["pred"],ex_af["naffairs"])
classification

#splitting the data into train and test data
train_data,test_data=train_test_split(ex_af,test_size=0.3)
#model building using 
model=sm.logit('naffairs~ kids+ vryunhap+ unhap+ avgmarr+ hapavg+ vryhap+ antirel+ notrel+ slghtrel+ smerel+ vryrel+ yrsmarr1+ yrsmarr2+ yrsmarr3+ yrsmarr4+ yrsmarr5+ yrsmarr6',data=train_data).fit()
model.summary()
model.summary2()
#AIC is 441.2892 
#prediction on test data
test_pred=model.predict(test_data)
test_data["test_pred"]=np.zeros(181)
#taking threshold value as optimal threshold value
test_data.loc[test_pred>optimal_threshold,"test_pred"]=1
#Confusion_matrix
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.naffairs)
confusion_matrix
accuracy_test=(85+35)/181
accuracy_test
#Classification report
classification_test=classification_report(test_data["test_pred"],test_data["naffairs"])
classification_test
#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(test_data["naffairs"],test_pred)
#plot of ROC
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test
###prediction on train data
train_pred=model.predict(train_data.iloc[:,1:])
#creating new column
train_data["train_pred"]=np.zeros(420)
train_data.loc[train_pred>optimal_threshold,"train_pred"]=1
#confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred,train_data.naffairs)
confusion_matrix
####
#Accuracy test
accuracy_train=(227+65)/420
accuracy_train
#classification report
classification_train=classification_report(train_data.train_pred,train_data.naffairs)
classification_train

#ROC_AUC curve
roc_auc_train=metrics.auc(fpr,tpr)
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
#6.	The benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
#The objective of this case study was to predict whether a given person will have an affair 
# or not with given the set of attributes