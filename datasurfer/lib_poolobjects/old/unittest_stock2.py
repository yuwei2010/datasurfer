# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:11:26 2019

@author: WEI
"""
import numpy as np
import matplotlib.pyplot as plt

from stock import fidat
from scipy.spatial.distance import cdist



def diag_sum(dis, k):
    
    diag = np.diag(dis, k=k)
    
#    print(diag.size)
    
    return diag.sum() / diag.size





fd = fidat(js='IFX.DE')

arr1 = np.atleast_2d(np.array([s.median() for s in fd.split(fd.close, 'd')])).T

arr1 = (arr1 - arr1.min()) / arr1.ptp()

arr2 = arr1[-30:-10][::-1]


x1 = np.arange(arr1.size)


dis = cdist(arr2, arr1)

r, c = dis.shape

dis_vet = dis.sum(axis=0) / c

min1 = dis_vet.min()

dis_dia = np.array([diag_sum(dis, k) for k in range(c)], dtype=float)

offset = dis_dia.argmin()

dis_diare =  np.array([diag_sum(np.fliplr(dis), k) for k in range(c)], dtype=float)[::-1]

print(dis_diare.argmin())


fig, ax = plt.subplots()

ax.plot(x1, arr1)

ax.plot(np.arange(arr2.size)+52-arr2.size, arr2, lw=5, )
#
ax.plot(dis_diare)




#
#
#
#arr0 = np.fromiter((a.median() for a in fd.split('close', 'd')), dtype=float)
#
#
#plt.close('all')

#arr0 = np.atleast_2d(np.sin(np.arange(0, 2*np.pi, np.pi/10))).T
#arr1 = np.atleast_2d(np.cos(np.arange(0, np.pi/2, np.pi/10))[::-1]).T
#
#dis = cdist(arr1, arr0)
#
#print(dis.shape)
#print(arr1)
#
#r, c = dis.shape
#
#
#dis1 = dis.sum(axis=0) / c
#
#dis2 = np.array([get(dis, k) for k in range(c)])
#
#dis3 = np.array([get(np.fliplr(dis), k) for k in range(c)])[::-1]
#
#print(dis3.argmin())
#
#fig, ax = plt.subplots()
#
#ax.plot(arr0)
#ax.plot(np.arange(5)+5, arr1[::-1])
#
#ax = ax.twinx()
#
#ax.plot(dis3, color='r')

