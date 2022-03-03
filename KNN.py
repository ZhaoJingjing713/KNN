# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:56:36 2021

@author: MaYiming
"""
#导入库
import numpy as np
#KNN模型
class KNN:
    def __init__(self,dataset,label,classes,k):
        #数据集、数据标签、预测种类多少、K
        self.dataset = dataset
        self.label = label
        self.classes = classes
        self.k = k
        #数据集数据大小
        self.dataLen = len(self.dataset)
    def classify(self, test,dataset,label,k):
        #训练集数据大小
        datasize = dataset.shape[0]
        #利用如下的技巧求出测试数据和每个的距离，即每个都减去数据集然后平方和再开方
        testMatrix = np.tile(test,(datasize,1)) - dataset
        distance = ((testMatrix**2).sum(axis=1))**0.5
        #print(distance)
        #排序
        sortedDistance = distance.argsort()
        K_label = []
        #选出K个距离最近的
        for i in range(k):
            K_label.append(label[sortedDistance[i]])
        #求出距离最近的几个点里边的哪个标签最多
        number = [0]*self.classes
        for i in range(self.classes):
            number[i] = K_label.count(i)
        #求最多
        maxi = np.argmax(np.array(number))
        Res = maxi
        return Res
    def testClass(self, testDataset,testlabel):
        #测试集数量
        testNum = len(testDataset)
        error = 0.0
        #预测
        for i in range(testNum):
            Res = self.classify(testDataset[i,:],self.dataset,self.label,self.k)
            #print("分类结果:%d\t真实类别:%d" % (Res, label[i]))
            if Res != testlabel[i]:
                error += 1.0
        print("错误率:%f%%" %(error/float(testNum)*100))
        #返回百分比
        return error/float(testNum)*100