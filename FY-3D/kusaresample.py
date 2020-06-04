# -*- coding:utf-8 -*-

import h5py as hdf
import numpy as np
import math
#import cv2 as cv
#from NMI import NMI, H_Matrix, H_Joint, shift
from time import *
import pandas as pd
from multiprocessing import Pool, Process
import os

btime = time()
ctime = time()
#####读取HDF文件及数据组模块
#f = hdf.File("D:/1000M/20191221/FY3D_MERSI_GBAL_L1_20191221_0005_1000M_MS.HDF", 'r')  # 读取文件
f0155 = hdf.File("C:/fy3d/FY3D_MERSI_GBAL_L1_20191221_0155_1000M_MS.HDF", 'r')  # 读取文件
f0150 = hdf.File("C:/fy3d/FY3D_MERSI_GBAL_L1_20191221_0150_1000M_MS.HDF", 'r')  # 读取文件
f0510 = hdf.File("C:/fy3d/FY3D_MERSI_GBAL_L1_20191221_0510_1000M_MS.HDF", 'r')  # 读取文件
f0845 = hdf.File("C:/fy3d/FY3D_MERSI_GBAL_L1_20191221_0845_1000M_MS.HDF", 'r')  # 读取文件
f1020 = hdf.File("C:/fy3d/FY3D_MERSI_GBAL_L1_20191221_1020_1000M_MS.HDF", 'r')  # 读取文件
f1025 = hdf.File("C:/fy3d/FY3D_MERSI_GBAL_L1_20191221_1025_1000M_MS.HDF", 'r')  # 读取文件
f1205 = hdf.File("C:/fy3d/FY3D_MERSI_GBAL_L1_20191221_1205_1000M_MS.HDF", 'r')  # 读取文件
f2220 = hdf.File("C:/fy3d/FY3D_MERSI_GBAL_L1_20191221_2220_1000M_MS.HDF", 'r')  # 读取文件
flink = [f0155, f0150, f0510, f0845, f1020, f1025, f1205, f2220]
#print(flink)
#print(f)


def preprocessing(f):
    data = f.require_group("Data")
    定标 = f.require_group("Calibration")
    #EV_250_Aggr.1KM_RefSB:1,2,3,4
    #EV_1KM_RefSB:5-19
    #EV_1KM_Emissive:20,21,22,23
    #EV_250_Aggr.1KM_Emissive:24,25
    定标系数 = 定标['VIS_Cal_Coeff']
    L1_ = data['EV_250_Aggr.1KM_RefSB']
    L5_ = data['EV_1KM_RefSB']
    L1_L4 = L1_[0::, ::, ::]
    L5_L19 = L5_[0::, ::, ::]
    L1_L4[np.logical_or(L1_L4 > 60000, L1_L4 == 1)] = 0
    L5_L19[np.logical_or(L5_L19 > 60000, L5_L19 == 1)] = 0
    for i in np.arange(4):
        L1_L4[i] = L1_L4[i] * 定标系数[i, 1]+定标系数[i, 0]
    #print(L1_L4)
    L1_L4[np.logical_or(L1_L4 < 0, L1_L4 > 60000)] = 0

    #各通道定标
    for i in np.arange(15):
        L5_L19[i] = L5_L19[i] * 定标系数[i+4, 1] + 定标系数[i+4, 0]
    L5_L19[np.logical_or(L5_L19 < 0, L5_L19 > 60000)] = 0
    L1_L4[L1_L4 < 0] = 0
    L5_L19[L5_L19 < 0] = 0
    L20_L23 = data['EV_1KM_Emissive']               #已定标
    L24_L25 = data['EV_250_Aggr.1KM_Emissive']     #已定标
    #c = data.keys()
    L = L5_L19[7]  # 基准图层--12通道
    Lall = np.zeros((19, L.shape[0], L.shape[1]), dtype=np.int)  # 后置图层
    # print(L1_L4[1:3:, ::, ::].shape)
    Lall[:4:, ::, ::] = L1_L4[0:4:, ::, ::]
    Lall[4::, ::, ::] = L5_L19  # Lall储存了所有可见光波段的信息(2-19)
    # print(Lall)
    # print("np.max(Lall) =", np.max(Lall))
    # print("L", L)
    # print("Lall", Lall[11, ::, ::])
    #L12 = resample(L, 5, 2).astype(np.int)
    #print(定标系数)
    #print(c)
    '''真彩色波段 = data['EV_250_Aggr.1KM_RefSB']
    b = 真彩色波段[0]
    b[np.logical_or(b > 60000, b == 1)] = 0
    b = b*定标系数[0, 1]+定标系数[0, 0]
    b[b < 0] = 0
    g = 真彩色波段[1]
    g[np.logical_or(g > 60000, g == 1)] = 0
    g = g*定标系数[1, 1]+定标系数[1, 0]
    g[g < 0] = 0'''
    return [L, Lall]    #基准通道,所有通道


def resample(M, p, h):      #重采样函数
    retime = time()
    #print("resample", p, h)
    #p = 5   #放大倍率power
    M1 = np.zeros((M.shape[0]*p-(p-1), M.shape[1]*p-(p-1)))   #目标数组
    M1[0::p, 0::p] = M   #/100    #写入原信息点
    #print("M1.shape", M1.shape)
    #print("1", time()-retime)


    mid = np.zeros((M.shape[0]*(p-1)-(p-1), M.shape[1]*(p-1)-(p-1)))   #中心点数组
    edge = np.zeros((M.shape[0]*(p-1)-(p-1), M.shape[1]*(p-1)-(p-1), 4))     #参考边缘点数值数组
    dis = np.zeros((M.shape[0]*(p-1)-(p-1), M.shape[1]*(p-1)-(p-1), 5))     #四方向距离/权重数组
    #print("flag2")
    ########################################计算中心数组部分##############################
    for i in np.arange((p-1)):
        for j in np.arange((p-1)):
            edge[i::(p-1), j::(p-1), 0] = M[:-1:, :-1:]  #M为原图像数组
            edge[i::(p-1), j::(p-1), 1] = M[:-1:, 1::]      #在中心数组edge的参考边缘点数组中填值
            edge[i::(p-1), j::(p-1), 2] = M[1::, :-1:]
            edge[i::(p-1), j::(p-1), 3] = M[1::, 1::]
            dis[i::(p-1), j::(p-1), 0] = 1/math.sqrt((i % (p-1) + 1) ** 2 + (j % (p-1) + 1) ** 2)
            dis[i::(p-1), j::(p-1), 1] = 1/math.sqrt((i % (p-1) + 1) ** 2 + ((p-1) - j % (p-1)) ** 2)     #直接计算距离倒数
            dis[i::(p-1), j::(p-1), 2] = 1/math.sqrt(((p-1) - i % (p-1)) ** 2 + (j % (p-1) + 1) ** 2)
            dis[i::(p-1), j::(p-1), 3] = 1/math.sqrt(((p-1) - i % (p-1)) ** 2 + ((p-1) - j % (p-1)) ** 2)

    #h = 2   #邻点权重指数
    dis[:, :, 4] = dis[:, :, 0]**h + dis[:, :, 1]**h + dis[:, :, 2]**h + dis[:, :, 3]**h
    dis[:, :, 0] = dis[:, :, 0]**h / dis[:, :, 4]
    dis[:, :, 1] = dis[:, :, 1]**h / dis[:, :, 4]      #计算权重数组
    dis[:, :, 2] = dis[:, :, 2]**h / dis[:, :, 4]
    dis[:, :, 3] = dis[:, :, 3]**h / dis[:, :, 4]

    mid[:, :] = dis[:, :, 0] * edge[:, :, 0] + dis[:, :, 1] * edge[:, :, 1] + dis[:, :, 2] * edge[:, :, 2] + dis[:, :, 3] * edge[:, :, 3]
    #mid = mid/100
    mid = mid.reshape(-1)
    #print("mid.shape", mid.shape)
    #################################计算行列数组部分################################
    # row横整十行数组
    row = np.zeros((M.shape[0], (M1.shape[1]-M.shape[1])))     #行插值数组
    #print("row.shape", row.shape)
    row_edge = np.zeros((row.shape[0], row.shape[1], 2))          #行对应信息点数组
    #print("row_edge.shape", row_edge.shape)
    row_dis = np.zeros((row.shape[0], row.shape[1], 3))              #信息点距离/权重数组
    #print("row_dis.shape", row_dis.shape)
    for i in np.arange((p - 1)):
        row_edge[::, i::(p - 1), 0] = M[::, :-1:]
        row_edge[::, i::(p - 1), 1] = M[::, 1::]
        row_dis[::, i::(p - 1), 0] = (1/(i % (p - 1) + 1))**h
        row_dis[::, i::(p - 1), 1] = (1/(p-(i % (p-1) + 1)))**h
    #print("row distance", row_dis)
    #print("dis0 shape", row_dis[::, ::, 0].shape)
    row_dis[::, ::, 2] = row_dis[::, ::, 0] + row_dis[::, ::, 1]
    row_dis[::, ::, 0] = row_dis[::, ::, 0] / row_dis[::, ::, 2]
    row_dis[::, ::, 1] = row_dis[::, ::, 1] / row_dis[::, ::, 2]    #转换成权重数组
    #print("row distance0", row_dis)
    row[::, ::] = row_dis[::, ::, 0] * row_edge[::, ::, 0] + row_dis[::, ::, 1] * row_edge[::, ::, 1]
    #row = row/100
    #print("row", row)

    ############################竖行数组################################
    col = np.zeros(((M1.shape[0]-M.shape[0]), M.shape[1]))
    #print("col.shape", col.shape)
    col_edge = np.zeros((col.shape[0], col.shape[1], 2))  # 列对应信息点数组
    #print("col_edge.shape", col_edge.shape)
    col_dis = np.zeros((col.shape[0], col.shape[1], 3))  # 信息点距离/权重数组
    #print("col_dis.shape", col_dis.shape)
    for i in np.arange((p - 1)):
        col_edge[i::(p - 1), ::, 0] = M[:-1:, ::]
        col_edge[i::(p - 1), ::, 1] = M[1::, ::]
        col_dis[i::(p - 1), ::, 0] = (1 / (i % (p - 1) + 1))**h
        col_dis[i::(p - 1), ::, 1] = (1 / (p - (i % (p - 1) + 1)))**h
    # print("row distance", row_dis)
    #print("col0 shape", col_dis[::, ::, 0].shape)
    col_dis[::, ::, 2] = col_dis[::, ::, 0] + col_dis[::, ::, 1]
    col_dis[::, ::, 0] = col_dis[::, ::, 0] / col_dis[::, ::, 2]
    col_dis[::, ::, 1] = col_dis[::, ::, 1] / col_dis[::, ::, 2]  # 转换成权重数组
    # print("row distance0", row_dis)
    col[::, ::] = col_dis[::, ::, 0] * col_edge[::, ::, 0] + col_dis[::, ::, 1] * col_edge[::, ::, 1]
    #col = col / 100


    #以下掩膜部分,计算后赋值用
    nrow, ncol = M1.shape
    x, y = np.mgrid[0:nrow, 0:ncol]
#    print("flag2")

    mask1 = (np.logical_and(y % p == 0, np.logical_not(x % p == 0)))
#    print("mask1", mask1)     #竖整十列网格点bool掩膜,不含原信息点
    mask4 = (np.logical_and(x % p == 0, np.logical_not(y % p == 0)))
#    print("mask4", mask4)       #横整十行普通网格点
    mask2 = (np.logical_and(x % p != 0, y % p != 0))
#    print("mask2", mask2)        #普通网格点
#    mask3 = np.logical_and(x % p == 0, y % p == 0)    #信息点
#    print("mask3", mask3)
    '''for index, item in enumerate(M1):
        print(index, item)'''
    M1[mask2] = mid
    M1[np.where(mask4)] = row.reshape(-1)
    M1[mask1] = col.reshape(-1)
    #cv.imshow("1", M1)
    print("resample time", time() - retime)
    return M1


def H_Matrix(M):
#    print("H_Matrix")
    #Mhist = np.zeros(np.max(M)+1, dtype=np.int64)       #计入0灰度
    # Mhist  矩阵M直方图
    #print("直方图大小Mhist.size=", Mhist.size)
    #for x in np.nditer(M):
    #    Mhist[x] = Mhist[x] + 1       #计入0灰度
    #Mhist[M.reshape(-1)] += 1
    s = pd.Series(M.reshape(-1))
    KKSK = s.value_counts(normalize=True)
    #print("KKSK=", KKSK)
    M_array = KKSK.values.tolist()
    M_array = np.array(M_array)
#    Mhist[M[:, 0].reshape(-1), M[:, 1].reshape(-1)] += 1
#    print(Mhist)
    HM = 0  # HM  熵 H(Matrix)
    HM = HM - M_array * np.log2(M_array)
    #HM = HM - (Mhist[Mhist != 0] / (M.shape[0] * M.shape[1]) * np.log2(Mhist[Mhist != 0] / (M.shape[0] * M.shape[1])))
    HM = np.sum(HM)
    #print("HM", HM)
    return HM


def H_Joint(M1):  # M1是两层数组,图像重叠部分
#    print("H_Joint")
    #Mhist = np.zeros(((np.max(M1[0])+1), (np.max(M1[1])+1)), dtype=np.int64)  # 建立联合灰度直方数组 #计入0灰度
    M1 = M1.reshape(2, (M1.shape[1] * M1.shape[2]))  # 重构数组到二维
    #M1 = np.transpose(M1)  # 反转数组维度
    M1 = M1.T
#    print("M1", M1)
    M2 = M1[::, 0]*np.max(M1[::, 1]) + M1[::, 1]
#    print("M2", M2)
    #print("M2.SHAPE", M2.shape)
    s = pd.Series(M2)
    #print("s", s)
    #M1 = np.array([[62, 0], [62, 0]])
    '''s0 = pd.Series(M1[::, 0])
    s1 = pd.Series(M1[::, 1])
    s = pd.concat([s0, s1], axis=1)'''
    #print("s\n", s)
    Mhist = s.value_counts(normalize=True)
    #print("s.value_counts(normalize=True)\n", s.value_counts(normalize=True))
#    print(M1.shape)
#    print(M1)
#    print(M1[::, 1].reshape(-1).shape)
    #print("Mhist.shape=", Mhist.shape)
    #print("M1=", M1)
    #print("M1.shape,Mhist.shape", M1.shape, Mhist.shape)
    #print("M1[::, 0]", M1[::, 0])
    #print("Mhist",Mhist)
    #print("Mhist[62,0]", Mhist[62, 0])

    #Mhist[M1[::, 0], M1[::, 1]] += 1
    #print("Mhist[62,0]", Mhist[62, 0])
    '''for i in M1:
        #print("1")
        Mhist[i[0],i[1]] += 1      #计入0灰度
        #Mhist[M1[:, 0].reshape(-1), M1[:, 1].reshape(-1)] += 1
        #print("Mhist[62,0]", Mhist[62, 0])'''
    #print("sum=", Mhist.sum())
    #print("Mhist", Mhist)
    a = M1.shape[0]
    HM = 0

    HM = HM - (Mhist * np.log2(Mhist))
    HM = np.sum(HM)
    #print("HM=", HM)
    return HM


def NMI(M):     #定义NMI  #M为两层数组
#    print("NMI")
    return (H_Matrix(M[0])+H_Matrix(M[1]))/H_Joint(M)


def shift(M, x, y):        #定义位移函数,参数:双层矩阵,x位移,y位移
    global btime
    btime = time()
    #k = 0
    M1 = M[0]               #位移是M1相对于M2的位移
    M2 = M[1]
    if x > 0:
        M1 = M1[::, x::]
        M2 = M2[::, :-x:]
    elif x == 0:
        k = 1
        #print("\r")
    else:
        M1 = M1[::, :x:]
        M2 = M2[::, -x::]
    if y > 0:
        M1 = M1[y::, ::]
        M2 = M2[:-y:, ::]
    elif ｙ == 0:
        k = 1
        #print("\r")
    else:
        M1 = M1[:y:, ::]
        M2 = M2[-y::, ::]
    Matrix = np.stack((M1, M2), axis=0)
    #print("shift time = ", time() - btime); btime = time()
    return Matrix


def resampleNMI(i, j):
    print("resampleNMI", i, j)
    L_ = L12
    Lall_ = Lall[j]
    L2 = np.stack((resample(L_, 5, 2).astype(np.int), resample(Lall_, 5, 2).astype(np.int)), axis=0)
    #print("L2", L2)
    k = np.zeros((7, 7))
    #print("k=", k)
    for x in np.arange(-3, 4):
        for y in np.arange(-3, 3):
            k[x + 3, y + 3] = NMI(shift(L2, x, y))
            #print(k[x + 3, y + 3], "x=", x, "y=", y)
    k = k - k.max()
    print("文件序数=", i, "j=", j, k)
    return k


fprocessed = [preprocessing(flink[0]), preprocessing(flink[1]), preprocessing(flink[2]), preprocessing(flink[3]), preprocessing(flink[4]), preprocessing(flink[5]), preprocessing(flink[6]), preprocessing(flink[7])]
#print("fprocessed", fprocessed)
print("预处理time = ", time()-btime); btime = time()
'''for i in np.arange(8):
    print("i=", i)'''
    #print(flink[i])
L12 = fprocessed[0][0]
Lall = fprocessed[0][1]
i = 0
print("i=", i)
i = i+1
#for j in np.arange(19):
#    resampleNMI(0, j)
#print("L12, Lall", L12, Lall)
if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for j in np.arange(19):

        #print(j)
        p.apply_async(resampleNMI, args=(0, j))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


    #L2 = np.stack((resample(L, 5, 2).astype(np.int), resample(Lall[j], 5, 2).astype(np.int)), axis=0)
    #print("L2", L2)
    #print("NMI(L2)", NMI(L2))
    k = np.zeros((7, 7))
    for x in np.arange(-3, 4):
        for y in np.arange(-3, 4):
            k[x + 3, y + 3] = NMI(shift(L2, x, y))
            #print(k[x + 3, y + 3], "x=", x, "y=", y)'''
    #print("k", k)
    #print(k.max() - k)



dtime = time() - ctime
print("total time", dtime)
#resample(b, 5, 1)
#cv.imshow("1", resample(b, 5, 1))
#cv.imshow("2", resample(b, 5, 2))
#cv.imshow("3", resample(b, 5, 3))
#cv.imshow("4", resample(b, 5, 4))
#cv.imshow("before", b/100)
#cv.waitKey()
#print(resample(b))




