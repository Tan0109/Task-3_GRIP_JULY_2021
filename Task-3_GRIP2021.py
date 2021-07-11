#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation : #GRIP( JULY 2021)
# ## Data Science And Business Analytics -----------Task-3 Exploratory Data Ananlysis - Retail

# ## Dataset used : ‘SampleSuperstore’ -----------Tool used: 'Jupyter Notebook'

# ## By: Tanvee Gandhi

# In[1]:


#importing necessary libraries for analysis.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#directory is updated as the location of the dataset.

import os
os.chdir("C:/Users/hp.LAPTOP-AHHO3OKQ/Downloads")


# In[2]:


#loading of dataset.

data=pd.read_csv("SampleSuperstore.csv")


# In[3]:


#First 10 rows are returned to get an idea of the attributes and records of dataset.

data.head(10)


# In[4]:


#to get the exact dimensions of the given dataset.

data.shape


# In[5]:


#to get a list of attributes.

data.columns


# In[6]:


#now let's have a look at a count of null values in the given dataset.

data.isnull().sum()


# In[7]:


#we can even sort the values on the basis of profit to understand which conditions give rise to maximum profit.
#The data is being sorted into descending order.

data_sorted=data.sort_values(by=['Profit'], ascending=False)


# In[8]:


data_sorted.head(10)


# In[9]:


#we must check presence of duplicate entries, as they may cause abnormalities.

data_sorted.duplicated().sum()


# In[10]:


#removing duplicate entries.

data_sorted.drop_duplicates(inplace=True)


# In[11]:


#verifying the dimensions of dataset after removal of duplicate entries.

data_sorted.shape


# In[12]:


#now we shall look onto the statistical aspect of the numerical attributes of given dataset. We now use the sorted dataset.

data_sorted.describe()


# In[13]:


#now we shall check the inter-relationship between corresponding attributes of the given dataset.

data_sorted.corr()


# In[14]:


#visuaization of this correlation using heatmap.

sns.heatmap(data_sorted.corr(),annot=True)


# #### So we observe  the following:---
# #### 1.) The correlation of Sales and profit is positive and less than 0.5. Thus they are weakly correlated. Also that higher Profit can be achieved with higher Sales.
# 
# #### 2.) The correlation of Quantity and profit is positive and closer to zero. Thus correlation is negligible or 0 . Also that higher quantity is ineffictive for higher profits.
# 
# #### 3.) The correlation of Discount and profit is negative and closer to zero. Thus correlation is negligible or 0 on the negative scale . Also that higher Discount may pose a threat for Profits.

# In[15]:


#let's see how profit behaves with rise in sales.

plt.scatter(data_sorted['Sales'],data_sorted['Profit'],color="red",marker="*")
plt.title('Scatter plot to visualize the dependancy of profits on Sales')
plt.xlabel('Sales in US$')
plt.ylabel('Profit in US$')
plt.show()


# In[16]:


#let's look at the distribution of entries in Ship Mode(Categorical attribute).

plt.pie(data_sorted['Ship Mode'].value_counts(),labels=data_sorted['Ship Mode'].unique())
plt.show()


# In[17]:


#And the distribution of entries in Segment(Categorical attribute).

sns.countplot(x="Segment",data=data_sorted)


# In[18]:


#we shall check the joint frequency of Sub-category of products with Regions in US.

plt.figure(figsize=(17,5))
sns.countplot(x="Sub-Category", data=data_sorted, hue="Region")
plt.show()


# #### This means that we have a high ferquency of 'Binders' and 'Paper' products sold majorly in West and East.

# In[19]:


#now it is time to analyse which category of products give rise to increased profits.

t1=data_sorted.groupby('Category').mean()
t1


# In[20]:


t1['Profit'].plot.bar()
plt.title("Category vs Profit")
plt.xlabel("Category")
plt.ylabel("Profits")


# #### A major part of higher profit is occupied by products which falls under 'Technology'. Furthermore, products categorized as 'Furniture' shall be strictly be eradicated, while products of 'Office Supplies' could be continued with limited quantity.

# In[21]:


#and to analyse which sub-category of products give rise to increased profits.

t2=data_sorted.groupby('Sub-Category').mean()
t2


# In[22]:


plt.figure(figsize=(15,10))
t2['Profit'].plot.barh(color="purple")
plt.title("Sub-Category vs Profit")
plt.xlabel("Sub-Category")
plt.ylabel("Profits")
plt.show()


# #### A major part of higher profit is occupied by sales of 'Copiers', which is a technological product. It must be noted that, products listed as: Bookcases, Supplies, Tables (under category 'Furniture') must be strictly eradicated. Sale of products with very low profits such as: Art, Fastners, Binders, Furnishings, Labels contribute the least, and shall either be eradicated, or continued with acute quantity.

# In[23]:


#Evaluation of relationship between profit and shipment mode of products

t3=data_sorted.groupby('Ship Mode').mean()
t3


# In[24]:


plt.figure(figsize=(7,4))
t3['Profit'].plot.bar(color="green")
plt.title("Ship Mode vs Profit")
plt.xlabel("Ship Mode")
plt.ylabel("Profits")
plt.show()


# #### The above table illustrates that higher profits can be achieved with Ship Mode='First Class' 

# In[25]:


#Evaluation of relationship between profit and Segment/Working mode of products

t4=data_sorted.groupby('Segment').mean()
t4


# In[26]:


plt.figure(figsize=(7,3))
t4['Profit'].plot.bar(color="yellow")
plt.title("Segment vs Profit")
plt.xlabel("Segment")
plt.ylabel("Profits")
plt.show()


# #### Similarly, one can observe that higher rate of Segment='Home Office' leads to higher Profits.

# In[27]:


#Also we must see which region of the country is responsible for more profits.

t5=data_sorted.groupby('Region').mean()
t5


# In[28]:


plt.figure(figsize=(10,5))
t5['Profit'].plot.barh(color="brown")
plt.title("Region vs Profit")
plt.xlabel("Region")
plt.ylabel("Profits")
plt.show()


# #### Also, West and East Regions contribute the most to higher profits, and sales in Central US shall be decreased.

# In[29]:


#And finally we must see which cities and states have overall profits at height.

data_sorted[['State','City']].head(15)


# ### After successful analysis of dataset: 'SampleStore' using EDA(Exploratory Data Analysis), we can conclude the following points:--
# #### 1.] Sales should be increased significantly, removing the idea of discount and excess quantity.
# #### 2.] Products sale in East and West US shall be increased.
# #### 3.] The Shipment mode 'First class' can be used for better rate of profit.
# #### 4.] 'Home Office' Segment significantly increases profit.
# #### 5.] Products which fall under category 'Technology' such as 'Copiers' and 'Phones' can successfully boost up Profits.
# #### 6.] The traced out list of locations of maximum profit shall be acknowledged and sales there should be increased.
