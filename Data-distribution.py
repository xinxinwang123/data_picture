# 各个特征的概率密度分布图
import pandas as pd  # 数据分析
import numpy as np  # 科学计算
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

data_train = pd.read_csv("C:\Users\kudong\Desktop\data\927.csv")
data_train1 = pd.read_csv("C:\Users\kudong\Desktop\data\929.csv")
data_train2 = pd.read_csv("C:\Users\kudong\Desktop\data\data_cmp1.csv")
data_train3 = pd.read_csv("C:\Users\kudong\Desktop\data\data_cmp2.csv")
# data_train = csv.reader(open('C:\Users\kudong\Desktop\data\data(2).csv'))
# data_train.info()#查看数据是否缺失
# details=data_train.describe()
# print data_train
fig = plt.figure(figsize=(10, 15), dpi=150)
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
plt.subplot2grid((4, 1), (0, 0))
data_train.cpu.plot(kind='kde')
data_train1.cpu.plot(kind='kde')
data_train2.cpu.plot(kind='kde')
data_train3.cpu.plot(kind='kde')
plt.xlabel(u'cpu')  # plots an axis lable
plt.ylabel(u'density')
plt.legend((u'0927', u'0929', u'1010cmp1', u'1010cmp2'), loc='best')  # sets our legend for our graph.

plt.subplot2grid((4, 1), (1, 0))
data_train.memory.plot(kind='kde')
data_train1.memory.plot(kind='kde')
data_train2.memory.plot(kind='kde')
data_train3.memory.plot(kind='kde')
plt.xlabel(u'mem')  # plots an axis lable
plt.ylabel(u'density')
plt.legend((u'0927', u'0929', u'1010cmp1', u'1010cmp2'), loc='best')  # sets our legend for our graph.

plt.subplot2grid((4, 1), (2, 0))
data_train.mutex.plot(kind='kde')
data_train1.mutex.plot(kind='kde')
data_train2.mutex.plot(kind='kde')
data_train3.mutex.plot(kind='kde')
plt.xlabel(u'mutex')  # plots an axis lable
plt.ylabel(u'density')
plt.legend((u'0927', u'0929', u'1010cmp1', u'1010cmp2'), loc='best')  # sets our legend for our graph.

plt.subplot2grid((4, 1), (3, 0))
data_train.threads.plot(kind='kde')
data_train1.threads.plot(kind='kde')
data_train2.threads.plot(kind='kde')
data_train3.threads.plot(kind='kde')
plt.xlabel(u'threads')  # plots an axis lable
plt.ylabel(u'density')
plt.legend((u'0927', u'0929', u'1010cmp1', u'1010cmp2'), loc='best')  # sets our legend for our graph.

plt.show() 