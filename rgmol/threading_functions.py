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
import rgmol.molecular_calculations

#################################
## OVERLAP MATRIX CALCULATIONS ##
#################################

def _overlap_matrix(mol,overlap_matrix,Psi,length,start_i,start_j):
    """Calculates the overlap matrix as
    (S)ij = \\int Psi_i(r) Psi_j(r) dr
    on a part of the grid :
    from start_i to start_i + length and start_j to start_j + length
    """
    N_grids = len(mol.mol_grids.grids)
    for i in range(length):
        for j in range(i+1):
            index_i,index_j = i + start_i, j + start_j
            overlap = 0
            for grid,index_grid in zip(mol.mol_grids.grids,range(N_grids)):
                overlap += grid.integrate(Psi[index_grid,index_i]*Psi[index_grid,index_j])
            overlap_matrix[index_i,index_j] = overlap
            overlap_matrix[index_j,index_i] = overlap


def _overlap_matrix_double(mol,overlap_matrix,Psi,list_div_element):
    """Just do the calculation for multiple ones. This is used when the number of processors is N = 2^k with k odd"""
    for (start_i,start_j),length in list_div_element:
        _overlap_matrix(mol,overlap_matrix,Psi,length,start_i,start_j)



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


def calculate_overlap_matrix_multithread(mol,Psi,nprocs):
    """
    This function separates the calculation into multiple threads automatically.
    As the overlap matrix is symmetric, only the rectangle triangle needs to be computed.
    The triangle is divided into 2^K smaller triangles which are all computed in separate threads.
    K is equal to the highest power of 2 that is lower that the number maximum of threads
    """

    N = len(Psi[0])
    overlap_matrix = np.zeros((N,N))

    L = 2**np.arange(10)
    K = np.argmin(nprocs>=L)-1 #Get the higher number of proc that is a power of 2 that will be used

    list_div = _divide_triangle(N,K,np.array([0,0]))
    list_th = []

    if K%2==0:
        for (start_i,start_j),length in list_div:
            kk = th.Thread(target=_overlap_matrix,args=(mol,overlap_matrix,Psi,length,start_i,start_j))
            list_th.append(kk)
            kk.start()

    else:
        for list_div_element in list_div:
            thread = th.Thread(target=_overlap_matrix_double,args=(mol,overlap_matrix,Psi,list_div_element))
            list_th.append(thread)
            thread.start()

    for thread in list_th:
        thread.join()

    return overlap_matrix

#####################################
## TRANSITION DENSITY CALCUALTIONS ##
#####################################



def _product_MO_calc(mol,MO,list_add_transition,transitions,nocc,nvirt,index_move,transition_density_coefficients):

    num_transitions = len(transition_density_coefficients)

    for index in range(len(transitions)):
        if transitions[index]:
            occ,virt = np.unravel_index(index+index_move,(nocc,nvirt))

            MO_product = MO[occ] * MO[virt]

            transition_coeffs = transition_density_coefficients[:,occ,virt]
            for transition in range(num_transitions):
                coeff = transition_coeffs[transition]
                if coeff!=0:
                    list_add_transition[transition] += coeff * MO_product

def _divide_bool(arr,nprocs):
    """
    Divides an array composed of 0 and 1 into multiple fragments that have the same numbers of 1
    """

    somm_bool = np.sum(arr)

    num_per_procs = somm_bool // nprocs
    list_num_per_procs = np.array([num_per_procs for k in range(nprocs)])
    num_procs_add = int(somm_bool % nprocs)

    list_X = [0 for k in range(nprocs+1)]
    list_num_per_procs[:num_procs_add] += 1
    num_proc = 0
    count = 0

    for index_bool in range(len(arr)):
        count += arr[index_bool]

        if count == list_num_per_procs[num_proc]:
            list_X[num_proc+1] = index_bool+1
            num_proc += 1
            count = 0
            if num_proc == nprocs:
                break

    list_bools = []
    for k in range(nprocs):
        if k!=nprocs-1:
            list_bools.append(arr[list_X[k]:list_X[k+1]])
        else:
            list_bools.append(arr[list_X[k]:])

    return list_bools,list_X


def calculate_transition_density_multithread(mol,transitions,transition_density_coefficients,grid_points,nprocs):
    """
    This function divides the virtual coefficients into multiple parts
    """
    #This hard cap is because it is actually slower to put too much processors
    if nprocs > 4:
        nprocs = 4

    num_transitions = len(transition_density_coefficients)
    nx,ny,nz = grid_points

    list_add_transitions = np.zeros((nprocs,num_transitions,nx,ny,nz))
    list_th = []
    nocc,nvirt = np.shape(transitions)


    list_transitions,list_X = _divide_bool(transitions.ravel(),nprocs)

    for list_transition,index_move,index_proc in zip(list_transitions,list_X,range(nprocs)):
        thread = th.Thread(target=_product_MO_calc,args=(mol,mol.properties["Reconstructed_MO"],list_add_transitions[index_proc],list_transition,nocc,nvirt,index_move,transition_density_coefficients))
        list_th.append(thread)
        thread.start()

    for thread in list_th:
        thread.join()
    transition_density_list = np.einsum("ijklm->jklm",list_add_transitions)

    return transition_density_list


####################################
## Reconstruction of eigenvectors ##
####################################


def _reconstruct(reconstructed_eigenvectors,transitions,eigenvectors,index):

    for eigenvector,index_eig in zip(eigenvectors,range(len(eigenvectors))):
        reconstructed_eigenvector = np.einsum("i,ijkl->jkl",eigenvector,transitions)
        # reconstructed_eigenvector_norm = mol.mol_grids.(reconstructed_eigenvector,reconstructed_eigenvector)
        reconstructed_eigenvectors[index+index_eig] = reconstructed_eigenvector


def _divide_eigenvectors(eigenvectors,nprocs):

    num_transitions = len(eigenvectors)

    num_per_procs = num_transitions // nprocs
    list_num_per_procs = np.array([num_per_procs for k in range(nprocs)])
    num_procs_add = int(num_transitions % nprocs)
    list_num_per_procs[:num_procs_add] += 1


    list_eigenvectors = []
    list_index = []
    count = 0
    for index in range(nprocs):
        list_eigenvectors.append(eigenvectors[count:count+list_num_per_procs[index]])
        list_index.append(count)
        count += list_num_per_procs[index]

    return list_eigenvectors,list_index


def multithreading_reconstruct_eigenvectors(transitions,eigenvectors,nprocs):
    """REconstructs the eigenvectors"""

    #This hard cap is because it is actually slower to put too much processors
    if nprocs > 4:
        nprocs = 4

    N_trans,nx,ny,nz = np.shape(transitions)

    list_eigenvectors,list_index = _divide_eigenvectors(eigenvectors,nprocs)
    reconstructed_eigenvectors = [np.zeros((nx,ny,nz)) for k in range(N_trans)]

    list_th = []
    for eigenvectors,index in zip(list_eigenvectors,list_index):
        thread = th.Thread(target=_reconstruct,args=(reconstructed_eigenvectors,transitions,eigenvectors,index))
        list_th.append(thread)
        thread.start()

    for thread in list_th:
        thread.join()

    return reconstructed_eigenvectors

##
def _reconstruct_atomic_grids(mol,reconstructed_eigenvectors,reconstructed_eigenvector_norms,transitions,eigenvectors,index):

    N_grids,N_trans,N_r,N_ang = np.shape(transitions)

    for eigenvector,index_eig in zip(eigenvectors,range(len(eigenvectors))):
        reconstructed_eigenvector = np.einsum("j,ijkl->ikl",eigenvector,transitions)
        reconstructed_eigenvector_norm = mol.mol_grids.integrate_product(reconstructed_eigenvector,reconstructed_eigenvector)
        reconstructed_eigenvector_norms[index+index_eig] = reconstructed_eigenvector_norm
        reconstructed_eigenvectors[index+index_eig] = reconstructed_eigenvector/reconstructed_eigenvector_norm**(1/2)



def multithreading_reconstruct_eigenvectors_atomic_grids(mol,transitions,eigenvectors,nprocs):
    """REconstructs the eigenvectors"""

    #This hard cap is because it is actually slower to put too much processors
    if nprocs > 4:
        nprocs = 4

    N_grids,N_trans,N_r,N_ang = np.shape(transitions)

    list_eigenvectors,list_index = _divide_eigenvectors(eigenvectors,nprocs)
    reconstructed_eigenvectors = [np.zeros((N_grids,N_r,N_ang)) for k in range(N_trans)]
    reconstructed_eigenvector_norms = [0 for k in range(N_trans)]

    list_th = []
    for eigenvectors,index in zip(list_eigenvectors,list_index):
        thread = th.Thread(target=_reconstruct_atomic_grids,args=(mol,reconstructed_eigenvectors,reconstructed_eigenvector_norms,transitions,eigenvectors,index))
        list_th.append(thread)
        thread.start()

    for thread in list_th:
        thread.join()

    return np.array(reconstructed_eigenvectors),np.array(reconstructed_eigenvector_norms)





