# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:10:41 2021

@author: MaYiming
"""
#导入库
import numpy as np
import random
import matplotlib.pyplot as plt
from KNN import KNN
#参数设置
params = {}
params['filename'] = 'Iris.csv'
#归一化
def autoNorm(dataset):
    lenDataset = len(dataset)
    mini = dataset.min(0)
    maxi = dataset.max(0)
    ranges = (maxi - mini)
    rangesMatrix = np.tile(ranges,(lenDataset,1))
    normDataset = dataset/rangesMatrix
    return normDataset
#导入csv文件
data = np.loadtxt(params['filename'],delimiter=',',dtype=str)       #由于数据含有字符串标签，需要以str形式读取数据
data = data[1:]                                              #第一行为数据标识信息 需要删去
random.shuffle(data)                                      #数据打乱
dataset = data[:, 1:-1]                                            #读取数据，读取全部的行，读取第一列到最后一列之前，为X
label = data[:, -1]                                              #读取标签

for i,x in enumerate(label):                                     #将字符串标签更改为数字标签
    if x == 'Iris-setosa':
        label[i] = 1
    if x == 'Iris-versicolor':
        label[i] = 2
    if x == 'Iris-virginica':
        label[i] = 3
dataset = autoNorm(np.float64(dataset))                          #格式转换
label = np.int16(label)
#划分数据集和测试集
testRate = 0.85
testNum = int(testRate * len(dataset))
Traindataset = dataset[testNum:len(dataset),:]
Trainlabel = label[testNum:len(dataset)]
Testdataset = dataset[0:testNum,:]
Testlabel = label[0:testNum]
#记载错误率
Error = []
#预测种类数
classNum = 3
#训练
for i in range(1,15):
    knn = KNN(Traindataset,Trainlabel,classNum,i)
    Error.append(knn.testClass(Testdataset,Testlabel))
lis_x = list(range(1,15))
#画图
plt.figure()
plt.xlabel("iter")
plt.ylabel("error%")
plt.plot(lis_x,Error,color = 'blue')
plt.show()