# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 15:12:32 2023

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 11:37:34 2023

@author: lenovo
"""

1]PROBLEM:-

BUSINESS OBJECTIVE:-classify the Size_Categorie using SVM.





#Importing the Necessary Liabrary
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#Loading the Dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/SVM/forestfires.csv')

#EDA
df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isna().sum()#To check NA Values.

#Visualtions.to check outliers.
plt.boxplot(df.DMC)
plt.boxplot(df.day)
plt.boxplot(df.temp)
plt.boxplot(df.month)
plt.boxplot(df.FFMC)
plt.boxplot(df.DC)
plt.boxplot(df.ISI)
plt.boxplot(df.RH)
plt.boxplot(df.wind)
plt.boxplot(df.rain)
plt.boxplot(df.area)


#Removing the Outliers,using Winsorizer
#Data Cleaning
from feature_engine.outliers import Winsorizer
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['DMC'])
df['DMC']=w.fit_transform(df[['DMC']])

w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['temp'])
df['temp']=w.fit_transform(df[['temp']])

w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['FFMC'])
df['FFMC']=w.fit_transform(df[['FFMC']])

w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['DC'])
df['DC']=w.fit_transform(df[['DC']])

w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['ISI'])
df['ISI']=w.fit_transform(df[['ISI']])


w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['RH'])
df['RH']=w.fit_transform(df[['RH']])

w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['temp'])
df['temp']=w.fit_transform(df[['temp']])

w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['temp'])
df['temp']=w.fit_transform(df[['temp']])

w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['wind'])
df['wind']=w.fit_transform(df[['wind']])

w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['area'])
df['area']=w.fit_transform(df[['area']])

w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['rain'])
df['rain']=w.fit_transform(df[['rain']])


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['month']=l.fit_transform(df['month'])
df['day']=l.fit_transform(df['day'])


#Visulations
plt.plot(df.iloc[:,0:30])
sns.distplot(df.month)


inpu=df.iloc[:,0:30]#Predictors
inpu.head()
target=df.iloc[:,[30]]#Target
target.head()

#Split the Dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inpu,target,test_size=0.2)



#Kernel='linear'
from sklearn.svm import SVC
model=SVC(kernel='linear')
model.fit(x_train,y_train)



#Evaluations On Train data
trainpred=model.predict(x_train)
accuracy_score(y_train,trainpred)
train_report=classification_report(y_train,trainpred)
confusion_matrix(y_train,trainpred)

#Evaluations On Test data
testpred=model.predict(x_test)
accuracy_score(y_test,testpred)
test_report=classification_report(y_test,testpred)
confusion_matrix(y_test,testpred)

#Visualizations for predictions of train and test.
sns.displot(trainpred)
plt.hist(testpred)


#Kernel='rbf'
from sklearn.svm import SVC
model=SVC(kernel='rbf')
model.fit(x_train,y_train)



#Evaluations On Train data
trainpred=model.predict(x_train)
accuracy_score(y_train,trainpred)
train_report=classification_report(y_train,trainpred)
confusion_matrix(y_train,trainpred)

#Evaluations On Test data
testpred=model.predict(x_test)
accuracy_score(y_test,testpred)
test_report=classification_report(y_test,testpred)
confusion_matrix(y_test,testpred)

#Visualizations for predictions of train and test.
sns.displot(trainpred)
plt.hist(testpred)

#In between this two Kernel(linear,rbf) we found that..Instead of 'rbf kernel'.. "linear kernel" is best to choose,because of it got 100% accuracy in both train and test data which is Right Fit. 





2]PROBLEM::
    

BUSINESS OBJECTIVE::--1) Prepare a classification model using 'SVM' ML model. 
for salary data.:
    USE BOTH 'SalaryData_Test' and 'SalaryData_Train'
    


#Loading the Dataset
train=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/SVM/SalaryData_Train(1).csv')

test=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/SVM/SalaryData_Test(1).csv')

train
test
train.info()
test.info()
train.describe()
test.describe()
train.head()
test.head()
train.tail()
test.tail()
train.shape
test.shape

#Data Cleaning on train Datset
train[train.duplicated()]
Train = train.drop_duplicates()
Train.isna().sum()
Train.Salary.value_counts()

#Data Cleaning on test Datset
test['maritalstatus'].value_counts()
test[test.duplicated()]
Test=test.drop_duplicates()
Test.isna().sum()
Test.Salary.value_counts()


#Visulazations

sns.countplot(x='Salary',data=Train)#Train Dataset
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Train['Salary'].value_counts()


sns.countplot(x='Salary',data=Test)#Test Dataset
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Train['Salary'].value_counts()

sns.scatterplot(Train['occupation'],Train['workclass'],hue=Train['Salary'])
pd.crosstab(Train['Salary'],Train['occupation']).mean().plot(kind='bar')
pd.crosstab(Train['Salary'],Train['workclass']).mean().plot(kind='bar')
pd.crosstab(Train['Salary'],Train['relationship']).mean().plot(kind='bar')
pd.crosstab(Train['Salary'],Train['education']).mean().plot(kind='bar')


string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]


##Preprocessing the data.

from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
for i in string_columns:
        Train[i]= l.fit_transform(Train[i])
        Test[i]=l.fit_transform(Test[i])
        
##Capturing the column.as 'col'
col = Train.columns
col 
        
 
# storing the values in x_train,y_train,x_test & y_test for spliting the data in train and test for analysis
x_train = Train[col[0:13]].values
y_train = Train[col[13]].values
x_test = Test[col[0:13]].values
y_test = Test[col[13]].values        


##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

x_train=norm_func(x_train)
x_test=norm_func(x_test)


##Kernel='linear
from sklearn.svm import SVC
model=SVC(kernel='linear')

#Fitting as well as Predicting the model
train_pred=model.fit(x_train,y_train).predict(x_train)
test_pred=model.fit(x_train,y_train).predict(x_test)


train_acc=np.mean(train_pred==y_train)
train_acc

test_acc=np.mean(test_pred==y_test)
test_acc

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, test_pred)
confusion_matrix

#calculating the accuracy of this model w.r.t. this dataset
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,test_pred))


## Kernel='rbf
from sklearn.svm import SVC
model=SVC(kernel='rbf')

#Fitting as well as Predicting the model
train_pred=model.fit(x_train,y_train).predict(x_train)
test_pred=model.fit(x_train,y_train).predict(x_test)


train_acc=np.mean(train_pred==y_train)
train_acc

test_acc=np.mean(test_pred==y_test)
test_acc

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, test_pred)
confusion_matrix

#calculating the accuracy of this model w.r.t. this dataset
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,test_pred))


#In Both(Kernel='linear','rbf').. we got accuracy of 80%.which is good.