#!/usr/bin/env python
# coding: utf-8

# # importing essential libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 



# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Dataset read

# In[3]:


df = pd.read_csv('3ff3fc72-ead7-4271-b693-88751a81b2c7_Data.csv')


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.head()


# In[9]:


df.describe()


# In[10]:


X = df
y = df['Series Name']


# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Series Name'] = le.fit_transform(X['Series Name'])
y = le.transform(y)


# In[12]:


X.info()


# In[13]:


X.head()


# In[14]:


df.dropna(inplace=True)


# # Null value checking

# In[15]:


df.isnull().sum()


# In[16]:


cols = X.columns


# # clustering 

# In[17]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
plt.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=0)
km.fit(X)
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='black', s=300)
plt.xlabel("X")
plt.ylabel("Y")
plt.title('Clustering')
plt.show()


# In[18]:


df


# In[19]:


df = df.fillna(0)


# In[20]:


df.head()


# # pie plot

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
data = [130, 125, 120, 110, 100, 100, 100, 100, 100]
keys=['2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]']
palette_color = sns.color_palette('pastel')
plt.figure(figsize=(10, 5))  
plt.pie(data, labels=keys, colors=palette_color,
        autopct='%.0f%%')
plt.show()


# # Bar plot

# In[22]:


years = [2012, 2013, 2014, 2015, 2016]
values = [47.8, 46.5, 45.2, 43.9, 42.6]
plt.figure(figsize=(10, 5))
plt.bar(years, values, color = 'green')
plt.xlabel('Years')
plt.ylabel('Values')
plt.title('Values by 5-year Intervals')
plt.show()


# In[23]:


data_1 = {'years': ['2012', '2013', '2014', '2015', '2016'],
        'values': [47.8, 46.5, 45.2, 43.9, 42.6]}
df_1= pd.DataFrame(data_1)
plt.figure(figsize=(10, 5))
plt.bar(df_1['years'], df_1['values'], color = 'orange')
plt.yscale("log")
plt.show()


# In[24]:


df_1['values'] = df_1['values']/df_1['values'].mean()*100
plt.figure(figsize=(10, 5))
plt.bar(df_1['years'], df_1['values'], color = 'pink')
plt.show()


# In[25]:


df


# In[26]:


print(df.isnull().sum())


# In[27]:


df = df.fillna(0)


# In[28]:


df.head()


# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[30]:


columns_to_drop = ['Series Name', 'Series Code', 'Country Name', 'Country Code']
data_2 = df.drop(columns_to_drop, axis=1)


# In[31]:


data_2.info()


# In[32]:


data_2.dtypes


# In[33]:


data_2 = data_2.replace(['..', 'nan'], [0, 0])


# In[34]:


data_2 = data_2.fillna(0)


# In[35]:


data_2.info()


# In[36]:


data_2


# In[37]:


X = data_2[['2012 [YR2012]', '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]', '2016 [YR2016]', '2017 [YR2017]']]
y = data_2['2015 [YR2015]']


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# # Random forest

# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rf = RandomForestRegressor(n_estimators=100, random_state=0)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R^2 score : ", r2)

