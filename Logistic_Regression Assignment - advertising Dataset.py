# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:42:17 2024

@author: Priyanka
"""

"""
In this time and age of widespread internet usage,
effective and targeted marketing plays a vital role. 
A marketing company would like to develop a strategy by 
analyzing their customer data. For this, data like age, 
location, time of activity, etc. has been collected to 
determine whether a user will click on an ad or not. 
Perform Logistic Regression on the given data to 
predict whether a user will click on an ad or not.

Business Problem-
Q.What is the business objective?
The use of the internet and social media have changed consumer behavior 
and the ways in which companies conduct their business. Social and digital 
marketing offers significant opportunities to organizations through lower
costs, improved brand awareness and increased sales.
whether or not a particular internet user clicked on an Advertisement. 
We will try to create a model that will predict whether or not they will 
click on an ad based on the features of that user.

Q.Are there any constraints?
significant challenges exist from negative electronic word-of-mouth as well as
intrusive and irritating online brand presence.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
DM=pd.read_csv("C:/Data Set/advertising.csv")
DM.dtypes
from textblob import TextBlob
DM['Polarity']=DM['Ad_Topic_Line'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)

DM1=DM.drop(['Ad_Topic_Line','City','Country','Timestamp'],axis=1,inplace=True)
DM.loc[DM.Polarity>0,"Polarity"]=1
DM.dtypes
#There are column names having spaces ,let us rename the columns
DM.columns='Daily_time_spent','Age','Area_income','Internet_usage','Male','Clicked_on_ad','Polarity'
#Let us re-arrange the columns
DM=DM.iloc[:,[6,0,1,2,3,4,5]]
#Many columns have different scale values let us apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
DM_norm=norm_func(DM.iloc[:,:])
DM.isna().sum()
#There are no null values

#model bulding
logit_model=sm.logit('Clicked_on_ad ~ Polarity+ Daily_time_spent+ Age+ Area_income+ Internet_usage+ Male',data=DM_norm).fit()
logit_model.summary()
logit_model.summary2()
#let us go for prediction
pred=logit_model.predict(DM_norm.iloc[:,:6])

#To derive ROC curve
#ROC curve has tpr on y axis and fpr on x axis,ideally,tpr must be high
#fpr must be low
fpr,tpr,thresholds=roc_curve(DM_norm.Clicked_on_ad,pred)
#To identify optimum threshold
optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=thresholds[optimal_idx]
optimal_threshold
#0.63 ,by default you can take 0.5 value as a threshold
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
DM_norm["pred"]=np.zeros(1000)
DM_norm.loc[pred>optimal_threshold,"pred"]=1
#if predicted value is greater than optimal threshold then change pred column as 1
#Classification report
classification=classification_report(DM_norm["pred"],DM_norm["Clicked_on_ad"])
classification

#splitting the data into train and test data
train_data,test_data=train_test_split(DM_norm,test_size=0.3)
#model building using 
model=sm.logit('Clicked_on_ad ~ Polarity+ Daily_time_spent+ Age+ Area_income+ Internet_usage+ Male',data=train_data).fit()
model.summary()
model.summary2()
#AIC is 146.2055
#prediction on test data
test_pred=model.predict(test_data)
test_data["test_pred"]=np.zeros(300)
#taking threshold value as optimal threshold value
test_data.loc[test_pred>optimal_threshold,"test_pred"]=1
#Confusion_matrix
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.Clicked_on_ad)
confusion_matrix
accuracy_test=(157+135)/300
accuracy_test
#Classification report
classification_test=classification_report(test_data["test_pred"],test_data["Clicked_on_ad"])
classification_test
#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(test_data["Clicked_on_ad"],test_pred)
#plot of ROC
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test
###prediction on train data
train_pred=model.predict(train_data.iloc[:,:6])
#creating new column
train_data["train_pred"]=np.zeros(700)
train_data.loc[train_pred>optimal_threshold,"train_pred"]=1
#confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred,train_data.Clicked_on_ad)
confusion_matrix
#Accuracy test
accuracy_train=(338+341)/700
accuracy_train
#classification report
classification_train=classification_report(train_data.train_pred,train_data.Clicked_on_ad)
classification_train
#ROC_AUC curve
roc_auc_train=metrics.auc(fpr,tpr)
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
