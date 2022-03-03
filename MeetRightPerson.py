# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:55:44 2021

@author: MaYiming
"""
#导入库
import numpy as np
import matplotlib.pyplot as plt
from KNN import KNN
#设置参数
params = {}
params["data_path"] = 'datingTestSet.txt'
#从文件中获取数据
def GetData(params):
    #打开
    with open(params["data_path"]) as p:
        #读取每一行
        lines = p.readlines()
        #所有的行数
        numberOfLine = len(lines)
        #对数据数组和标签数组进行初始化
        lisMatrix = np.zeros((numberOfLine, 3))
        label = np.zeros((numberOfLine,1))
        #将数据导入初始化数组
        for index,line in enumerate(lines):
            #一些处理技巧
            li = line.strip("\n")
            lis_split = (li.strip().split("\t"))
            #前三列为数据
            lisMatrix[index:] = lis_split[0:3]
            #最后一列为标签，这里分别标记为0、1、2
            if lis_split[-1] == 'didntLike':
                label[index] = 0
            elif lis_split[-1] == 'smallDoses':
                label[index] = 1
            else:
                label[index] = 2
    #返回
    return lisMatrix, label
#归一化
def autoNorm(dataset):
    #总数
    lenDataset = len(dataset)
    #平均值
#    mean = dataset.mean(0)
#    meanMatrix = np.tile(mean,(lenDataset,1))
#    dataset -= meanMatrix
    #求每一列的最大最小
    mini = dataset.min(0)
    maxi = dataset.max(0)
    #差值
    ranges = (maxi - mini)
    #归一化
    rangesMatrix = np.tile(ranges,(lenDataset,1))
    normDataset = dataset/rangesMatrix
    return normDataset
#加载数据
dataset ,label = GetData(params)
#数据的图像化输出
#三个变量两两一对进行图像化处理
plt.figure()
for index,lab in enumerate(label):
    #三种lab分别输出不同的颜色
    if lab == 0:
        plt.scatter(dataset[index][0],dataset[index][1],c="red")
    elif lab == 1:
        plt.scatter(dataset[index][0],dataset[index][1],c="blue")
    else:
        plt.scatter(dataset[index][0],dataset[index][1],c="black")
plt.show()
plt.figure()
for index,lab in enumerate(label):
    if lab == 0:
        plt.scatter(dataset[index][0],dataset[index][2],c="red")
    elif lab == 1:
        plt.scatter(dataset[index][0],dataset[index][2],c="blue")
    else:
        plt.scatter(dataset[index][0],dataset[index][2],c="black")
plt.show()
plt.figure()
for index,lab in enumerate(label):
    if lab == 0:
        plt.scatter(dataset[index][1],dataset[index][2],c="red")
    elif lab == 1:
        plt.scatter(dataset[index][1],dataset[index][2],c="blue")
    else:
        plt.scatter(dataset[index][1],dataset[index][2],c="black")
plt.show()
#归一化
dataset = autoNorm(dataset)
#数据集和测试集的数据划分
testRate = 0.2
testNum = int(testRate * len(dataset))
Traindataset = dataset[testNum:len(dataset),:]
Trainlabel = label[testNum:len(dataset),:]
Testdataset = dataset[0:testNum,:]
Testlabel = label[0:testNum,:]
#错误记载
Error = []
#分类数
classNum = 3
#K从1~15进行变换
for i in range(1,15):
    #knn训练
    knn = KNN(Traindataset,Trainlabel,classNum,i)
    #返回预测错误率
    Error.append(knn.testClass(Testdataset,Testlabel))
#显示不同K的测试效果
lis_x = list(range(1,15))
plt.figure()
plt.xlabel("iter")
plt.ylabel("error%")
plt.plot(lis_x,Error,color = 'blue')
plt.show()