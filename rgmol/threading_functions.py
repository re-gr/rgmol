#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Notes
-----

This script adds functions to do quicker calculations using multiprocessing
"""

import numpy as np
import threading as th
from rgmol.objects import *

#################################
## OVERLAP MATRIX CALCULATIONS ##
#################################

def _overlap_matrix(overlap_matrix,Psi,length,start_i,start_j,dV):
    """Calculates the overlap matrix as
    (S)ij = \\int Psi_i(r) Psi_j(r) dr
    on a part of the grid :
    from start_i to start_i + length and start_j to start_j + length
    """
    for i in range(length):
        for j in range(i+1):
            index_i,index_j = i + start_i, j + start_j
            overlap = np.einsum("ijk,ijk->",Psi[index_i],Psi[index_j])*dV
            overlap_matrix[index_i,index_j] = overlap
            overlap_matrix[index_j,index_i] = overlap


def _overlap_matrix_double(overlap_matrix,Psi,list_div_element,dV):
    """Just do the calculation for multiple ones. This is used when the number of processors is N = 2^k with k odd"""
    for (start_i,start_j),length in list_div_element:
        _overlap_matrix(overlap_matrix,Psi,length,start_i,start_j,dV)



def _divide_triangle(N,K,l):
    """
    This functions divides the rectangle triangle to be computed into K almost equal rectangle triangle.
    This is used to properly divide the calculation between the threads.
    """
    if K == 0:
        return [[l,N]]
    #Divides into 2*2 parts or 4 parts
    l1 = l + np.array([0,0]) #topleft
    l2 = l + np.array([0,N//2]) #topright
    l3 = l + np.array([N//2,0]) #botleft, adds one if odd dim
    l4 = l + np.array([N//2,N//2]) #botright

    if K==1:
        #Divides into 2*2 parts
        return [[[l1,N//2],[l3,N//2+N%2]]] + [[[l2,N//2+N%2],[l4,N//2+N%2]]]

    if K>=2:
        #Divides into 4 parts, not equals if the length is odd
        return _divide_triangle(N//2,K-2,l1) + _divide_triangle(N//2+N%2,K-2,l2) + _divide_triangle(N//2+N%2,K-2,l3) + _divide_triangle(N//2+N%2,K-2,l4)


def calculate_overlap_matrix_multithread(Psi,nprocs,dV):
    """
    This functions separates the calculation into multiple threads automatically.
    As the overlap matrix is symmetric, only the rectangle triangle needs to be computed.
    The triangle is divided into 2^K smaller triangles which are all computed in separate threads.
    K is equal to the highest power of 2 that is lower that the number maximum of threads
    """

    N = len(Psi)
    overlap_matrix = np.zeros((N,N))

    L = 2**np.arange(10)
    K = np.argmin(nprocs>=L)-1 #Get the higher number of proc that is a power of 2 that will be used

    list_div = _divide_triangle(N,K,np.array([0,0]))
    list_th = []

    if K%2==0:
        for (start_i,start_j),length in list_div:
            kk = th.Thread(target=_overlap_matrix,args=(overlap_matrix,Psi,length,start_i,start_j,dV))
            list_th.append(kk)
            kk.start()

    else:
        for list_div_element in list_div:
            thread = th.Thread(target=_overlap_matrix_double,args=(overlap_matrix,Psi,list_div_element,dV))
            list_th.append(thread)
            thread.start()

    for thread in list_th:
        thread.join()

    return overlap_matrix

