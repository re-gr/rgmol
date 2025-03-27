#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script adds functions, and methods to the molecule objects.
These methods allow the calculation of various chemical properties such as AO, MO, Transition densities or linear response function.
"""

import numpy as np
import scipy as sp
from rgmol.objects import *


def calculate_fukui(self,mol_p=None,mol_m=None,grid_points=(100,100,100),delta=5):
    """Calculates fukui function"""

    if not(mol_p) and not(mol_m):
        raise TypeError("Need at least the molecule with N+1 or N-1 number of electrons")

    if not "electron_density" in self.properties:
        self.calculate_electron_density(grid_points=grid_points,delta=delta)
    rhoN = self.properties["electron_density"]

    if mol_p:
        #calculate f+
        if not "electron_density" in mol_p.properties:
            mol_p.calculate_electron_density(grid_points=grid_points,delta=delta)

        rhoNp1 = mol_p.properties["electron_density"]

        fp = rhoNp1 - rhoN
    else:
        fp = None

    if mol_m:
        #calculate f-
        if not "electron_density" in mol_m.properties:
            mol_m.calculate_electron_density(grid_points=grid_points,delta=delta)

        rhoNm1 = mol_m.properties["electron_density"]

        fm = rhoN - rhoNm1

    else:
        fm = None

    if mol_p and mol_m:
        f0 = (fp+fm)/2
    else: f0 = None

    self.properties["f0"] = f0
    self.properties["f+"] = fp
    self.properties["f-"] = fm

    return f0,fp,fm


def calculate_hardness(self):
    """
    Calculates hardness using Koopmans theorem
    """

    if not "MO_energy" in self.properties:
        raise TypeError("The Energy of the MO has not been found. Cannot calculate the hardness")

    MO_occupancy = self.properties["MO_occupancy"]
    MO_energy =  self.properties["MO_energy"]

    LUMO = np.argmin(MO_occupancy)
    HOMO = LUMO - 1

    hardness = MO_energy[LUMO] - MO_energy[HOMO]
    self.properties["hardness"] = hardness
    return hardness


def calculate_softness_kernel_eigenmodes(self,fukui_type="0",mol_p=None,mol_m=None,grid_points=(100,100,100),delta=10):
    """Calculates the softness kernel eigenmodes"""


    nx,ny,nz = grid_points

    if not "transition_density_list" in self.properties:
        self.calculate_transition_density(grid_points,delta=delta)

    transition_density_list = self.properties["transition_density_list"]
    transition_energy = np.array(self.properties['transition_energy'])

    if not "f+" in self.properties and not "f-" in self.properties:
        self.calculate_fukui(mol_p=mol_p,mol_m=mol_m,grid_points=grid_points,delta=delta)

    if "0" in fukui_type:
        fukui = self.properties["f0"]
    elif "+" in fukui_type or "p" in fukui_type:
        fukui = self.properties["f+"]
    elif "-" in fukui_type or "m" in fukui_type:
        fukui = self.properties["f-"]

    if not 'hardness' in self.properties:
        hardness = self.calculate_hardness()


    voxel_matrix = self.properties["voxel_matrix"]
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]


    basis = np.array(transition_density_list + [fukui]) #Append the fukui function
    factor = np.array((transition_energy/2).tolist() + [hardness])

    #Append the hardness and divide the energy by 2 to take into account the 2 in the formula of the LRF
    length_basis = len(basis)

    overlap_matrix = np.zeros((length_basis,length_basis))

    for first_transition in range(length_basis):
        for second_transition in range(first_transition+1):
            overlap_integral = np.sum(basis[first_transition] * basis[second_transition]) * dV
            overlap_matrix[first_transition,second_transition] = overlap_integral
            overlap_matrix[second_transition,first_transition] = overlap_integral


    #following dimensions : i,k,j
    #Calculates <rho_0^i|rho_0^k>
    overlap_matrix_reshaped = overlap_matrix.reshape((length_basis,length_basis,1))
    #Calculates <rho_0^k|rho_0^j>
    overlap_matrix_reshaped_2 = overlap_matrix.reshape((1,length_basis,length_basis))
    #Used to divide by the energy for k
    factor_reshaped = factor.reshape(1,length_basis,1)

    s_matrix = np.sum(1/factor_reshaped* overlap_matrix_reshaped * overlap_matrix_reshaped_2,axis=1)


    overlap_matrix_inv = np.linalg.inv(overlap_matrix)
    diag_matrix = overlap_matrix_inv.dot(s_matrix)

    eigenvalues, eigenvectors = np.linalg.eigh(diag_matrix)


    eigenvalues, eigenvectors = sp.linalg.eigh(s_matrix,overlap_matrix)
    eigenvectors = eigenvectors.transpose()
    reconstructed_eigenvectors = []


    for eigenvector in zip(eigenvectors,eigenvalues,range(len(eigenvectors))):
        eigenvectors[eigenvector[2]] = eigenvectors[eigenvector[2]]/np.sum(abs(eigenvectors[eigenvector[2]]))

        reconstructed_eigenvector = np.zeros((nx,ny,nz))

        for transition in range(len(eigenvector[0])):
            reconstructed_eigenvector += eigenvector[0][transition] * basis[transition]
        reconstructed_eigenvector = reconstructed_eigenvector/np.sum(reconstructed_eigenvector**2*dV)**(1/2)
        reconstructed_eigenvectors.append(reconstructed_eigenvector)

    self.properties["softness_kernel_eigenvalues"] = eigenvalues
    self.properties["softness_kernel_eigenvectors"] = reconstructed_eigenvectors
    self.properties["contribution_softness_kernel_eigenvectors"] = eigenvectors

    return eigenvalues, reconstructed_eigenvectors



molecule.calculate_fukui = calculate_fukui
molecule.calculate_hardness = calculate_hardness
molecule.calculate_softness_kernel_eigenmodes = calculate_softness_kernel_eigenmodes









