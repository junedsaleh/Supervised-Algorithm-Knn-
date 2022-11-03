#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[3]:


df = pd.read_csv("F:/Data Analytics for Business/DAB/Sem 3/DAB-300-Machine Learning 2/Assignment/Week 8/endangeredLang.csv")


# In[4]:


df


# In[5]:


print(df.isnull())


# In[6]:


df.dropna()


# In[7]:


print(df.dropna(axis=1))


# In[8]:


print(df.fillna(0))


# In[10]:


print(df.fillna(method="ffill",axis=1))


# In[14]:


dfk = pd.read_csv("F:/Data Analytics for Business/DAB/Sem 3/DAB-300-Machine Learning 2/Assignment/Week 8/loan_train.csv")
dfk


# In[15]:


testData = dfk[['Gender','Principal','terms','age']]


# In[16]:


X= testData.drop("Gender", axis=1) 
Y = testData["terms"]  


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .2, random_state=25) #


# In[18]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,Y)


# In[19]:


Y_predict = knn.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, Y_predict)
confusion_matrix


# In[22]:


from sklearn.metrics import accuracy_score


# In[23]:


print(accuracy_score(y_test, Y_predict))


# In[25]:


from sklearn.model_selection import cross_val_score


# In[27]:


# creating  list of K/neighbours for KNN
myList = list(range(1,50))

# empty list that will keep cv scores
cv_scores = []

#   perform 10 fold cross validation
for k in myList:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())


# In[5]:


# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = myList[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(myList, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


# In[ ]:





# In[ ]:




