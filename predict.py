# -*- coding: utf-8 -*-
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
#不平衡数据处理
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

#忽略警告信息
import warnings
warnings.filterwarnings("ignore")

def make_pridict_data(dataframe, batch_size=128):
  dataframe = dataframe.copy()
  #使用 tf.data.Dataset.from_tensor_slices 进行加载
  ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
  ds = ds.batch(batch_size)
  return ds

def get_data():
    data_matrix = [[0] for i in range(1,5)]
    data_matrix[0][0] = float(input('请输入Recency:'))
    data_matrix[1][0] = float(input('请输入Frequency:'))
    data_matrix[2][0] = float(input('请输入Monetary:'))
    data_matrix[3][0] = float(input('请输入Time:'))
    return data_matrix

def get_result(pridict_result):
    if(pridict_result>=0.8):
        print('预测会来,几率为:'+str(pridict_result*100)+'%')
    else:
        print('预测不会来,几率为:'+str((1-pridict_result)*100)+'%')

URL = './data.csv'
#读取数据
dataframe = pd.read_csv(URL)
# 使用z-score标准化数据
scale_features = ['Recency','Frequency','Monetary','Time']
#实例化标准化方法
ss = StandardScaler()
#通过标准化方法标准化数据
dataframe[scale_features] = ss.fit_transform(dataframe[scale_features])
#选取标准化后的结果标签列为变量：labels
labels = dataframe['whether_he/she_donated_blood_in_March_2007']
#在训练集中pop出标签列 
#除去label列之外的所有feature值
trainset = dataframe.drop(['whether_he/she_donated_blood_in_March_2007'], axis=1)
#使用imblearn进行随机过采样
dataframe.pop("whether_he/she_donated_blood_in_March_2007")
#不平衡数据集处理方法：过采样（over-sampling）
#对小类的数据样本进行过采样来增加小类的数据样本个数，即采样的个数大于该类样本的个数。
#实例化过采样方法
ros = RandomOverSampler(random_state=0)
#通过剥离标签列的数据集和标签列进行过采样，得到[(0, 570), (1, 570)]：标签为0和1的数量相同的数据集
X_resampled, y_resampled = ros.fit_resample(trainset,labels)
#将数据集和标签列合并
X_resampled['whether_he/she_donated_blood_in_March_2007'] = y_resampled
#将数据集赋值为新变量dataframe，用于结构化数据集的机器学习算法
dataframe = X_resampled
#print(dataframe)

num = get_data()
d = dict(zip(scale_features, num))
df=pd.DataFrame(d,index=['0'])
df[scale_features] = ss.transform(df[scale_features])

model = keras.models.load_model("./model/")
pridict_result = model.predict(make_pridict_data(df))[0][0]
#print(pridict_result)
print('\n\n\n\n\n\n\n\n\n\n\n\n')
print('+++++++++++++++++++++++++++++++++++++++++++++')
print('\n',df,'\n\n')
get_result(pridict_result)
print('\n+++++++++++++++++++++++++++++++++++++++++++++')

input('输入任何值以停止运行')
















