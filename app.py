#!/usr/bin/env python
# coding: utf-8

# ### Import library

# In[17]:


from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ### Load Weather Data
# #### weather.csv is produced from weatherPreprocess.ipynb, containing temperature 20170301~20170430 and 20180301~20870430

# In[18]:


def Read_WeatherData(filename):
    dataset = pd.read_csv(filename,sep=',')
       
    return dataset


# In[19]:


weather_train = Read_WeatherData("weather.csv")


# ### Load Training Data
# #### power data of 20170101~20181130, download from https://data.gov.tw/dataset/19995

# In[20]:


def Read_TrainData(filename):
    dataset = pd.read_csv(filename,sep=',')
    dataset.rename(columns={'日期':'date', '尖峰負載(MW)':'peak','淨尖峰供電能力(MW)':'provide','備轉容量(MW)':'volumn'}, inplace=True)

        
    return dataset


# In[21]:


dataset = Read_TrainData("20170101_20181130.csv")


# ### Load Testing Data
# ##### predicted power data of 20190401~20190408, download from https://data.gov.tw/dataset/33462

# In[22]:


def Read_TestData(filename):
    dataset = pd.read_csv(filename,sep=',')
    dataset.rename(columns={'日期(年/月/日)':'date', '預估淨尖峰供電能力(萬瓩)':'provide','預估尖峰備轉容量(萬瓩)':'volumn'}, inplace=True)
    future_temp = [20,20.5,22,22,23,24,23,24]
    dataset['temperature'] = future_temp
    return dataset


# In[23]:


new_data = Read_TestData("20190401.csv")
# select feature
new_feature = new_data.iloc[1:, [1,4]]*10
new_feature['temperature'] = new_data['temperature']
# new_feature


# ### Select features and predict column

# In[24]:


march_data = dataset.iloc[59:90, 0:4]
march_data.index = range(len(march_data))
march_data['temperature'] = weather_train['2017-03'][:]
april_data = dataset.iloc[90:120, 0:4]
april_data.index = range(len(april_data))
april_data['temperature'] = weather_train['2017-04'][:]
# next_march_data = march_data.append(dataset.iloc[425:455, 0:4]) #2018
# next_april_data = april_data.append(dataset.iloc[456:463, 0:4]) #2018


# In[25]:


march_feature = march_data.iloc[:, [1,3,4]]
march_peak = march_data.iloc[:, 2]
# new_feature = april_data.iloc[:, [1,3]]
# new_peak = april_data.iloc[:, 2]
# new_feature
train_feature = march_feature.append(april_data.iloc[:, [1,3,4]])
train_peak = march_peak.append(april_data.iloc[:, 2])
train_feature


# ### Build regression model

# In[26]:


from sklearn.datasets import make_regression
# fit final model
model = LinearRegression()
model.fit(train_feature, train_peak)


# ### Make prediction
# ##### 因供電能力和備轉容量這兩個 feature在regression過程中權重較重，
# ##### 但目前這兩個值也是被預測出來的，所以在最後將氣溫較接近的2017尖峰負載再加入平均。

# In[27]:


# make a prediction
ynew = model.predict(new_feature)
# predicted outputs
prediction = {}
prediction['date'] = new_data.iloc[1:, 0].tolist()
# add 2017 data to average, since this year's weather is similiar
val_data = dataset.iloc[91:98, 2]

prediction['peak_load(MW)'] = np.average([val_data,ynew], axis=0).astype(int)

# convert dict to dataframe
pd.DataFrame.from_dict(prediction)


# ### Write out predition
# ##### eg. date,peak_load(MW)
# #####       20190402,22905

# In[28]:


def Write_Out(data):
    output_file = "submission.csv"
    f_out = open(output_file,'w')
    
    data.to_csv(output_file,index=False,sep=',')


# In[29]:


Write_Out(pd.DataFrame.from_dict(prediction))


# In[ ]:




