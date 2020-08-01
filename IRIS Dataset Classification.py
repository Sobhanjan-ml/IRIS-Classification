#!/usr/bin/env python
# coding: utf-8

# Importing Liabraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Importing Dataset

# In[2]:


dataset = pd.read_csv('IRIS Dataset.csv')


# In[4]:


dataset.head()


# Checking for null-values

# In[5]:


dataset.isnull().any()


# Checking the Data types of the variables

# In[48]:


dataset.dtypes


# In[6]:


dataset.describe()


# Counting the number species

# In[19]:


dataset['species'].value_counts()


# Checking the co-relation between the different variables

# In[20]:


dataset.corr()


# In[21]:


sns.heatmap(dataset.corr(),annot=True)


# Separating Independent and Dependent Variables

# In[24]:


x = dataset.iloc[:,0:4].values


# In[25]:


x


# In[26]:


y = dataset.iloc[:,4].values


# In[27]:


y


# Train and Test Splits

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[30]:


x_train


# In[31]:


x_test


# In[32]:


y_train


# In[33]:


y_test


# Importing Algorithm

# In[34]:


from sklearn.neighbors import KNeighborsClassifier


# In[35]:


kn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)


# In[36]:


kn.fit(x_train,y_train)


# In[37]:


y_pred = kn.predict(x_test)


# In[38]:


y_pred


# Checking accuracy score

# In[39]:


from sklearn.metrics import accuracy_score


# In[41]:


accuracy_score(y_test,y_pred)*100


# Testing Model

# In[42]:


kn.predict([[5.1,3.5,1.4,0.2]])


# In[46]:


kn.predict([[6.7,3.0,5.2,2.3]])

