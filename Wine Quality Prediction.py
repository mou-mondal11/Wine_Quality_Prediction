#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction :
# * Machine Learning model to predict the quality of wine using linear regression

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 


# In[121]:


data = pd.read_csv(r'C:\Users\DELL\Desktop\Data\WineQT.csv')
data


# In[122]:


data.describe()


# In[123]:


data.info()


# In[124]:


data.duplicated().sum()


# # Data Visualization & Data analysis

# **Count of wine by quality**

# In[125]:


sns.catplot(x='quality', data= data, kind = 'count')


# * If the quality of wine score near to 0 then it is bad quality wine
# * If the quality of wine score near to 10 then it is good quality wine
# * here maximum wine quality is medium

# In[126]:


plt.figure(figsize=(15,15))
sns.heatmap(data.corr(),linewidths=0.5,annot=True,cmap='rainbow')
plt.show()


# In[127]:


sns.barplot(x='quality',y='fixed acidity',data=data)


# * High amount of fixed acidity is required for good quality wine

# In[128]:


sns.barplot(x='quality',y='volatile acidity',data=data)


# * Less amount of volatile acidity is required for good quality wine

# In[129]:


sns.barplot(x='quality',y='citric acid',data=data)


# * High amount of citric acid is required for good quality wine

# In[130]:


sns.barplot(x='quality',y='residual sugar',data=data)


# * Midium quantity of residual sugar is best for good qualit of wine

# In[131]:


sns.barplot(x='quality',y='chlorides',data=data)


# * Less amount of chlorides is required for good quality wine

# In[132]:


sns.barplot(x='quality',y='free sulfur dioxide',data=data)


# * Free sulfur dioxide quantity must be within 10 to 12 for good quality wine

# In[133]:


sns.barplot(x='quality',y='total sulfur dioxide',data=data)


# * Total sulfur dioxide quantity must be within 30 to 35 not less than that for good quality wine

# In[134]:


sns.barplot(x='quality',y='density',data=data)


# In[135]:


sns.barplot(x='quality',y='pH',data=data)


# In[136]:


sns.barplot(x='quality',y='sulphates',data=data)


# * High amount of sulphates is required for good quality wine

# In[137]:


sns.barplot(x='quality',y='alcohol',data=data)


# * High amount of alchohol is required for good quality wine

# # *Data preprocessing*

# In[156]:


data


# In[157]:


#splitting into dependent and independent variable
x = data.iloc[:,:11].values
y = data.iloc[:,-2].values


# In[158]:


#Splitting into Training And Testing set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .30, random_state=10)


# In[160]:


#Feature Scalling
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 


# # *Data Modeling*

# In[161]:


linear_model = LinearRegression()
model = linear_model.fit(x_train,y_train)
pred = model.predict(x_test)
print(pred)


# In[162]:


# rounding off the predicted values for test set
predicted_data = np.round_(pred)
print(predicted_data)


# In[163]:


#accuracy Check
from sklearn.metrics import r2_score
r2_score(y_test,pred)


# In[166]:


#value Prediction
value=model.predict([[7.4,0.700,0.00,1.9,0.076,11.0,34.0,0.99780,3.51,0.56,9.4]])
print(np.round_(value))


# In[172]:


value=model.predict([[0.02,0.10,1.,0.076,11.0,34.0,0.99780,3.51,0.56,2.4,7.4]])
print(np.round_(value))


# # ________________________________________________________________

# In[ ]:




