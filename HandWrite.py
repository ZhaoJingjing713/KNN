# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:00:37 2021

@author: MaYiming
"""
#导入库和KNN
from KNN import KNN
import os
import numpy as np
import matplotlib.pyplot as plt
#需要用到的路径和文件名
params = {}
params["train_path"] = "trainingDigits//"
params["test_path"] = "testDigits//"
params['trainDirList'] = os.listdir(params["train_path"])
params['testDirList'] = os.listdir(params["test_path"])

#列数32
#行数32
TrainImageNum = len(params['trainDirList'])
TrainImageMatrix = [[[0] for i in range(32*32)]for j in range(TrainImageNum)]
TrainLabel = [0]*TrainImageNum
#遍历每一个文件
for ImageIndex in range(TrainImageNum):
    with open(os.path.join(params["train_path"],params["trainDirList"][ImageIndex])) as p:
        #每一行
        for i,line in enumerate(p):
            for j in range(32):
                #每一个进行转化
                TrainImageMatrix[ImageIndex][32*i+j] = int(line[j])
        #标签利用第一个字符处理
        TrainLabel[ImageIndex] = int(params['trainDirList'][ImageIndex][0])
TrainImageMatrix = np.array(TrainImageMatrix)
#测试集同上
TestImageNum = len(params['testDirList'])
TestImageMatrix = [[[0] for i in range(32*32)]for j in range(TestImageNum)]
TestLabel = [0]*TestImageNum
for ImageIndex in range(TestImageNum):
    with open(os.path.join(params["test_path"],params["testDirList"][ImageIndex])) as p:
        for i,line in enumerate(p):
            for j in range(32):
                TestImageMatrix[ImageIndex][32*i+j] = int(line[j])
        TestLabel[ImageIndex] = int(params['testDirList'][ImageIndex][0])
TestImageMatrix = np.array(TestImageMatrix)
#进行预测
Error = []
for i in range(1,15):
    knn = KNN(TrainImageMatrix,TrainLabel,10,i)
    Error.append(knn.testClass(TestImageMatrix,TestLabel))
#图像输出
lis_x = list(range(1,15))
plt.figure()
plt.xlabel("iter")
plt.ylabel("error%")
plt.plot(lis_x,Error,color = 'blue')
plt.show()