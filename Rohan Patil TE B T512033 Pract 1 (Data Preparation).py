#!/usr/bin/env python
# coding: utf-8

# # Rohan Patil TE B T512033 Pract 1 (Data Preparation)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


dataFrame=pd.read_csv('Heart.csv')


# In[17]:


dataFrame.shape


# In[18]:


dataFrame.head()


# In[19]:


dataFrame.tail()


# In[20]:


dataFrame=dataFrame.drop("Unnamed: 0",axis =1)


# In[21]:


dataFrame.dtypes


# In[22]:


dataFrame.describe()


# In[23]:


dataFrame.info()


# In[24]:


dataFrame.Ca.value_counts()


# In[25]:


dataFrame.Sex.value_counts()


# In[26]:


dataFrame.isnull()


# In[27]:


dataFrame.isnull().sum()


# In[28]:


dataFrame.Age.mean()


# In[29]:


dataFrame.describe()


# In[30]:


dataFrame["Age"].mean(axis=0)


# In[31]:


var=dataFrame.loc[:,['Age','Sex','ChestPain','RestBP','Chol']]


# In[32]:


var


# In[33]:


# Splitting the dataset into train and test sets: 75-25 split
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(var, test_size = 0.25, random_state = 42)
X_train.shape, X_test.shape


# In[34]:


tp=90
fp=11
fn=19
tn=40
acc=(tp+tn)/(tp+fp+fn+tn)
pre=tp/(tp+fp)
rec=tp/(tp+fn)
print("Accuracy is : {}".format(acc))
print("Precision is : {}".format(pre))
print("Recall is : {}".format(rec))
print("F1-Score is : {}".format((2*pre*rec)/(pre+rec)))


# In[35]:


plt.figure(figsize=(10,10))
sns.heatmap(dataFrame.corr(), annot=True)
plt.show()


# In[36]:


dataFrame.plot();


# In[33]:


dataFrame.hist(bins = 10,figsize = (15,11));


# In[34]:


sns.pairplot(var)


# In[35]:


import matplotlib.pyplot as plt

labels = ['Male', 'Female']
dataFrame['Sex'].value_counts().plot(kind="pie", labels=labels, startangle=90, explode=(0, 0), autopct='%1.1f%%')
plt.show()


# In[42]:


plt.hist(dataFrame["Age"],bins=15,label="Age Count")
plt.title("Age Histogram")
plt.xlabel("Age range")
plt.ylabel("Count according to Age")
plt.legend(loc="upper left");


# sns.barplot(x = "Slope", y = "Age", hue = "Sex", data = dataFrame)
# plt.title("Slope Group - Count Bar Plot Grouped by Sex")
# plt.show()

# In[47]:


sns.barplot(x = "Ca", y = "Chol", hue = "Slope", data = dataFrame)
plt.title("Ca Group - Count Bar Plot Grouped by Slope")
plt.show()


# In[51]:


plt.scatter(dataFrame["RestBP"],dataFrame["Age"],label="BP level according to Age")
plt.title("BP vs Age Scatterplot")
plt.xlabel("Rest BP level")
plt.ylabel("Age")
plt.legend(loc="lower right");


# In[ ]:




