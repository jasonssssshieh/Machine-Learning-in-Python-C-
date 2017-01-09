# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:40:40 2017

@author: Jason
"""

import numpy as np
#假如原始数据集为矩阵dataMat，dataMat中每一行代表一个样本，每一列代表同一个特征。
def zeromean(datamat):
    meanval = np.mean(datamat, axis = 0)  #按列求均值，即求各个特征的均值，axis=0表示按列求均值。
    newdata = datamat - meanval
    return newdata, meanval

newdata, meanval = zeromean(datamat)
covmat = np.cov(newdata, rowvar = 0) 
#若rowvar=0，说明传入的数据一行代表一个样本，
#若非0，说明传入的数据一列代表一个样本。因为newData每一行代表一个样本，所以将rowvar设置为0。

eigvals, eigvectors = np.linalg.eig(np.mat(covmat))
eigvalindice = np.argsort(eigvals)#对特征值从小到大排序  
n_eigvalindice = eigvalindice[-1:-(n+1):-1]#最大的n个特征值的下标  
#首先argsort对特征值是从小到大排序的，那么最大的n个特征值就排在后面，
#所以eigValIndice[-1:-(n+1):-1]就取出这个n个特征值对应的下标。
#【python里面，list[a:b:c]代表从下标a开始到b，步长为c。】
n_eigvect = eigvectors[:, n_eigvalindice]#最大的n个特征值对应的特征向量  
lowddatamat = newdata*n_eigvect#低维特征空间的数据  
reconmat = (lowddatamat * n_eigvect.T) + meanval#重构数据  
return lowddatamat, reconmat

#可以写个函数，函数传入的参数是百分比percentage和特征值向量，然后根据percentage确定n，
def percentage2n(eigvals, percentage):
    sortarray = np.sort(eigvals)#升序
    sortarray = sortarray[-1::-1]#逆序，即降序
    arraysum = sum(sortarray)
    tmpsum = 0
    num = 0
    for i in sortarray:
        tmpsum += i
        num += 1
        if tmpsum >= arraysum*percentage:
            return num

def pca(datamat, percentage=0.99):
    newdata,meanval = zeromean(datamat)
    covmat = np.cov(newdata, rowvar = 0)#若rowvar非0，一列代表一个样本，为0，一行代表一个样本 
    eigvals, eigvects = np.linalg.eig(np.mat(covmat))
    n = percentage2n(eigvals, percentage)
    eigvalindice = np.argsort(eigvals)
    n_eigvalindice = eigvalindice[-1:-(n+1):-1]
    n_eigvect = eigvalindice[:,n_eigvalindice]
    lowddata = newdata * n_eigvect
    reconmat = (lowddata*n_eigvect.T)+meanval
    return lowddata, reconmat
