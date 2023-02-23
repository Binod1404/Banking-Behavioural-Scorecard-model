#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


train=pd.read_csv('Train.csv')     #here we are having separate dataset for train and test
test=pd.read_csv('Test.csv')


# In[3]:


train.head()           #Col1 contains unique id and Col2 is our target variable 


# In[4]:


test.head()       #so here Col2 is missing and our aim is to pridict the elements of Col2. Col1 contains unique id 


# In[5]:


print(train.shape)
print(test.shape)  #the difference in number of columns is due to the col2


# In[6]:


train.info()    


# In[7]:


test.info()   #the number of columns containing object values are 3 and 12 respectively i think they should be equal 


# In[8]:


train.isna().sum()


# In[9]:


test.isna().sum()


# In[10]:


train.select_dtypes(include = 'object')       #lets check the object columns of train dataset


# In[11]:


test.select_dtypes(include='object')  #object columns of test dataset


# In[12]:


train.describe()


# In[13]:


test.describe()


# In[14]:


for columns in train.select_dtypes('object').columns:
    print(columns)
    print(train[columns].unique())


# In[15]:


for tcolumns in test.select_dtypes('object').columns:
    print(tcolumns)
    print(test[tcolumns].unique())              #we will separate the unique id col1 and treat the other columns


# In[16]:


X = train.drop(['Col1','Col2'], axis=1)        #lets separate the target variable and remove the unique id that is col1
y = train.Col2

XTest = test.drop(['Col1'], axis=1)    #all columns except Col1 goes in XTest


# # data mining
# 

# In[17]:


def data_treatment(row):    #lets convert the sring and '-' to float and nan respectively
    if row == '-':            #first lets create a user defined function that will convert '-'to np.nan and string to float
        return np.nan
    else:
        return float(row)


# In[18]:


treat_columns = list(X.select_dtypes('object').columns) + list(XTest.select_dtypes('object').columns)
print(len(treat_columns))
print(treat_columns)       #lets check the columns that need treatment


# In[19]:


for col in treat_columns:
    X[col] = X[col].apply(data_treatment)
    XTest[col] = XTest[col].apply(data_treatment)   #lets treat the columns 


# In[20]:


X.info()    #lets check the information after data mining


# In[21]:


XTest.info()


# In[22]:


X.isna().sum()


# In[23]:


X=X.fillna(X.mean())


# In[24]:


XTest=XTest.fillna(X.mean())


# In[25]:


X.isna().sum()


# In[26]:


XTest.isna().sum()


# In[27]:


y.info()


# In[28]:


y


# In[29]:


y.value_counts()


# # lets apply the models  

# In[30]:


#applying classification models since the target variable contains only binary numbers 0 and 1


# decision tree classifier

# In[31]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X,y)


# In[32]:


y_pred_dt=dt.predict(XTest)


# In[33]:


y_pred_dt


# In[34]:


train_accuracy_dt=dt.score(X,y)
train_accuracy_dt                     #model has learnt 99 percent which is very good


# random forest classifier

# In[35]:


from sklearn.ensemble import RandomForestClassifier
Rf=RandomForestClassifier()
Rf.fit(X,y)


# In[36]:


y_pred_Rf=Rf.predict(XTest)


# In[37]:


y_pred_Rf


# In[38]:


train_accuracy_Rf=Rf.score(X,y)
train_accuracy_Rf                  #even here the model has learnt 99 percent


# k neighbours classifier

# In[39]:


from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knc.fit(X,y)


# In[40]:


y_pred_knc=knc.predict(XTest)
y_pred_knc


# In[41]:


train_accuracy_knc=knc.score(X,y)
train_accuracy_knc


# In[42]:


#the train accuracy of the diffirent models are as follows:
print("the accuracy of decision tree classifier is",train_accuracy_dt)
print("the accuracy of random forest classifier is",train_accuracy_Rf)
print("the accuracy of kneighbours classifier is",train_accuracy_knc)


# # result

# the aim was to predict the values of Col2 for the test dataset 

# In[43]:


#the predicted value of Col2 using decision tree is :
y_pred_dt


# In[44]:


#all the values cannot be seen so
for i in range(0, len(y_pred_dt)):    
    print(y_pred_dt[i])


# In[45]:


#the predicted values of Col2 using random forest classifier is:
y_pred_Rf


# In[46]:


#to see all the values:
for i in range(0, len(y_pred_Rf)):    
    print(y_pred_Rf[i])


# In[47]:


#the pedicted values using kneighbours classifier is as follows:
y_pred_knc


# In[48]:


for i in range(0, len(y_pred_knc)):    
    print(y_pred_knc[i])

