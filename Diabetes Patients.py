#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np # for linear algebra
import pandas as pd # data processing, CSV file I/O, etc
import seaborn as sns # for plots
import plotly.graph_objects as go # for plots
import plotly.express as px #for plots
import matplotlib.pyplot as plt # for visualizations and plots
import missingno as msno # for plotting missing data


# In[9]:


df = pd.read_csv("C:/Users/THILAK.S/Downloads/Project 2 MeriSKILL/diabetes.csv")
df.head() # displays the top 5 values in the dataset


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.isnull().sum()


# In[13]:


df["Glucose"] = df["Glucose"].apply(lambda x: np.nan if x == 0 else x)
df["BloodPressure"] = df["BloodPressure"].apply(lambda x: np.nan if x == 0 else x)
df["SkinThickness"] = df["SkinThickness"].apply(lambda x: np.nan if x == 0 else x)
df["Insulin"] = df["Insulin"].apply(lambda x: np.nan if x == 0 else x)
df["BMI"] = df["BMI"].apply(lambda x: np.nan if x == 0 else x)


# In[14]:


df.isnull().sum()


# In[43]:


px.pie(df, names="Outcome")


# In[16]:


sns.countplot(x="Outcome", data=df, palette=random.choice(pallete))


# In[17]:


sns.countplot(x="Pregnancies", hue = "Outcome", data=df, palette=random.choice(pallete))


# In[18]:


sns.histplot(x="Pregnancies", hue="Outcome", data=df, kde=True, palette=random.choice(pallete))


# In[19]:


sns.histplot(x="BloodPressure", hue="Outcome", data=df, kde=True, palette=random.choice(pallete))


# In[20]:


sns.histplot(x="Glucose", hue="Outcome", data=df, kde=True, palette=random.choice(pallete))


# In[21]:


sns.histplot(x="SkinThickness", hue="Outcome", data=df, kde=True, palette=random.choice(pallete))


# In[22]:


sns.histplot(x="Insulin", hue="Outcome", data=df, kde=True, palette=random.choice(pallete))


# In[23]:


sns.histplot(x="Age", hue="Outcome", data=df, kde=True, palette=random.choice(pallete))


# In[24]:


sns.histplot(x="BMI", hue="Outcome", data=df, kde=True, palette=random.choice(pallete))


# In[25]:


sns.histplot(x="DiabetesPedigreeFunction", hue="Outcome", data=df, kde=True, palette=random.choice(pallete))


# In[26]:


sns.pairplot(df, hue='Outcome',palette=random.choice(pallete))


# In[27]:


fig, axs = plt.subplots(4, 2, figsize=(20,20))
axs = axs.flatten()
for i in range(len(df.columns)-1):
    sns.boxplot(data=df, x=df.columns[i], ax=axs[i], palette=random.choice(pallete))


# In[28]:


sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True, cmap='coolwarm', linecolor='white', annot=True).set_title("Correlation Map")


# In[29]:


df.isnull().sum()


# In[30]:


msno.bar(df)


# In[31]:


msno.matrix(df, figsize=(20,35))


# In[32]:


msno.heatmap(df, cmap=random.choice(pallete))


# In[33]:


msno.dendrogram(df)


# In[34]:


df.isnull().sum()/len(df)*100


# In[35]:


df.drop(columns=["Insulin"], inplace=True)


# In[36]:


df.describe()


# In[37]:


df.skew()


# In[38]:


# Highly skewed
df["BMI"].replace(to_replace=np.nan,value=df["BMI"].median(), inplace=True)
df["Pregnancies"].replace(to_replace=np.nan,value=df["Pregnancies"].median(), inplace=True)

# Normal
df["Glucose"].replace(to_replace=np.nan,value=df["Glucose"].mean(), inplace=True)
df["BloodPressure"].replace(to_replace=np.nan,value=df["BloodPressure"].mean(), inplace=True)
df["SkinThickness"].replace(to_replace=np.nan,value=df["SkinThickness"].mean(), inplace=True)


# In[39]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[40]:


df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f'Before: {df.shape}, After: {df_out.shape}')


# In[41]:


for col in df.columns[:-1]:
    up_out = df[col].quantile(0.90)
    low_out = df[col].quantile(0.10)
    med = df[col].median()
#     print(col, up_out, low_out, med)
    df[col] = np.where(df[col] > up_out, med, df[col])
    df[col] = np.where(df[col] < low_out, med, df[col])


# In[42]:


df.describe()

