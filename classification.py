# -*- coding: utf-8 -*-

#数学运算
import numpy as np
import pandas as pd

#tensorflow机器学习框架
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#skit_learn机器学习数学运算
from sklearn import preprocessing
import keras
#skit_learn数据标准化
from sklearn.preprocessing import StandardScaler
#不平衡数据处理
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
#忽略警告信息
import warnings
warnings.filterwarnings("ignore")
#一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
def df_to_dataset(dataframe, shuffle=True, batch_size=128):
    #拷贝dataframe的值
    dataframe = dataframe.copy()
    #取出标签列
    labels = dataframe.pop('whether_he/she_donated_blood_in_March_2007')
    #将标签列映射到特征列
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    #打乱顺序
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
    return ds

#导入数据包
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
#将数据集分为训练集，测试集和验证集，分配比例为8:2
train, test = train_test_split(dataframe, test_size=0.2)
train1, val = train_test_split(train, test_size=0.2)
#打印集合形状
print('训练集形状:', train.shape)
print('验证集形状:', val.shape)
print('测试集形状:', test.shape)
#定义标签列
feature_columns = []
#使用数值列作为输入流，scale_features为特征列
#一个特征列的输出将成为模型的输入。
for header in scale_features:
  #feature_column用于表示实数特征。使用此列时，模型将从 dataframe 中接收未更改的列值。
  feature_columns.append(feature_column.numeric_column(header))

#使用密集特征（DenseFeatures）层将特征列输入到Keras 模型中。
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
#创建一个输入流水线。
train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val, shuffle=False)
test_ds = df_to_dataset(test, shuffle=False)

#创建，编译和训练模型
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

#设置keras.optimizers.Adam()优化器
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

checkpoint_save_path = "./checkpoint.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')
#fit函数返回一个History的对象
history = model.fit(train_ds,      #输入数据，特征列
          validation_data=val_ds,  #输入验证集
          epochs=9500,             #训练的总轮数
          callbacks=[cp_callback]) #回调函数

#输入测试集，得到准确率
loss, accuracy = model.evaluate(test_ds)

#取出accuracy（列表）
acc = history.history['accuracy']
#绘制loss和acc曲线
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()

val_accuracy = history.history['val_accuracy']
plt.subplot(1, 2, 1)
plt.plot(val_accuracy, label='Training val_accuracy')
plt.title('Training val_accuracy')
plt.legend()

loss = history.history['loss']
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.title('Training Loss')
plt.legend()

val_loss = history.history['val_loss']
plt.subplot(1, 2, 2)
plt.plot(val_loss, label='Training val_loss')
plt.title('Training val_loss')
plt.legend()
plt.savefig('./confusion_matrix.png',dpi=350)
plt.show()

print("-------------")
#label = ['Recency','Frequency','Monetary','Time']
#num = [[1],[24],[6000],[77]]
#d = dict(zip(label, num))
#df=pd.DataFrame(d,index=['0'])
#dl = ss.transform(df)
#pr = make_pridict_data(dl)
#model.predict(pr)
print("-------------")
#保存模型
model.save("./model/")
#绘制网络模型图
from keras.utils import plot_model
plot_model(model, to_file='./model.png')
#打印预测准确率
print("Accuracy", accuracy)





























