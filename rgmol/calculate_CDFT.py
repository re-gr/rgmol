#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script adds functions, and methods to the molecule objects.
These methods allow the calculation of various chemical properties such as AO, MO, Transition densities or linear response function.
"""

import time
import numpy as np
import scipy as sp
import rgmol
from rgmol.objects import *
from rgmol.threading_functions import *


def calculate_fukui_function(self,mol_p=None,mol_m=None,grid_points=(100,100,100),delta=10):
    """
    calculate_fukui_function(mol_p=None,mol_m=None,grid_points=(100,100,100),delta=10)

    Calculates the fukui function using finite differences between electron density.
    If mol_p is provided, f+ will be computed.
    If mol_m is provided, f- will be computed.
    If both are provided, f+, f- and f0 will be computed.
    The fukui functions are automatically added to the molecule.properties with the keys : f0 f+ and f-

    Parameters
    ----------
    grid_points : list of 3
    delta : float, optional
        the length added on all directions of the box containing all atomic centers

    Returns
    -------
    f0
        the f0 fukui function
    fp
        the f+ fukui function
    fm
        the f- fukui function


    Notes
    -----

    :math:`f^+ = \\rho(N+1)-\\rho(N)`
    :math:`f^- = \\rho(N)-\\rho(N-1)`
    :math:`f^0 = (f^+ + f^-)/2`
    """

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
    calculate_hardness()

    This method calculates the global hardness using Koopmans theorem.
    The hardness is therefore just the difference LUMO - HOMO
    The hardness is directly added into the molecule.properties with the key "hardness"

    Parameters
    ----------
    None

    Returns
    -------
    hardness
        the hardness of the molecule

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


def calculate_eigenmodes_linear_response_function(self,grid_points=(100,100,100),delta=10):
    """
    calculate_eigenmodes_linear_response_function(grid_points=(100,100,100),delta=10)

    Calculates the linear response function from the transition densities.
    This method does not calculate directly the linear response, but only the eigenmodes.
    The mathematics behind this function will soon be available somewhere...


    Parameters
    ----------
    grid_points : list of 3
    delta : float, optional
        the length added on all directions of the box containing all atomic centers

    Returns
    -------
    linear_response_eigenvalues
        the eigenvalues of the linear response function
    linear_response_eigenvectors
        the eigenvectors of the linear response function

    Notes
    -----
    The linear response function kernel can be computed as :

    :math:`\\chi(r,r') = -2\\sum_{k\\neq0} \\frac{\\rho_0^k(r) \\rho_0^k(r')}{E_k-E_0}`

    With :math:`\\rho_0^k` the transition density, and :math:`E_k` the energy of the transition k.

    Therefore, the molecule needs the transition properties that can be extracted from a TD-DFT calculation, and the MO extracted from a molden file. More details can be found :doc:`here<../tuto/orbitals>`.
    """

    nx,ny,nz = grid_points

    if not "transition_density_list" in self.properties:
        self.calculate_transition_density(grid_points,delta=delta)

    #Dummy implementation for now
    transition_density_list = np.array(self.properties["transition_density_list"])
    transition_energy = np.array(self.properties['transition_energy'])

    number_transition = len(transition_density_list)

    voxel_matrix = self.properties["voxel_matrix"]
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]

    nprocs = rgmol.nprocs

    print("#################################")
    print("# Calculating Eigenmodes of LRF #")
    print("#################################")
    time_before_calc = time.time()

    if nprocs >1:
        transition_matrix = calculate_overlap_matrix_multithread(transition_density_list,nprocs,dV)
    else:
        transition_matrix = np.zeros((number_transition,number_transition))
        for first_transition in range(len(transition_density_list)):
            for second_transition in range(first_transition+1):
                overlap_integral = np.sum(transition_density_list[first_transition] * transition_density_list[second_transition]) * dV
                transition_matrix[first_transition,second_transition] = overlap_integral
                transition_matrix[second_transition,first_transition] = overlap_integral

    LR_matrix_in_TDB = np.zeros((number_transition,number_transition))
    #linear response matrix in transition densities basis


    #following dimensions : i,j,k
    #Calculates <rho_0^i|rho_0^j>
    transition_matrix_reshaped = transition_matrix.reshape((number_transition,number_transition,1))
    #Calculates <rho_0^j|rho_0^k>
    transition_matrix_reshaped_2 = transition_matrix.reshape((1,number_transition,number_transition))
    #Used to divide by the energy for j
    transition_energy_reshaped = transition_energy.reshape(1,number_transition,1)

    LR_matrix_in_TDB = np.einsum("ijk,ijk,ijk->ik",-2/transition_energy_reshaped,transition_matrix_reshaped,transition_matrix_reshaped_2)


    # transition_matrix_inv = np.linalg.inv(transition_matrix)
    # diag_matrix = transition_matrix_inv.dot(LR_matrix_in_TDB)

    # eigenvalues, eigenvectors = np.linalg.eigh(diag_matrix)
    eigenvalues, eigenvectors = sp.linalg.eigh(LR_matrix_in_TDB,transition_matrix)
    eigenvectors = eigenvectors.transpose()

    if nprocs >1:
        reconstructed_eigenvectors = reconstruct_eigenvectors(transition_density_list,eigenvectors,dV,nprocs)
    else:
        reconstructed_eigenvectors = []
        for eigenvector in eigenvectors:
            eigenvector_reshaped = eigenvector.reshape((number_transition,1,1,1))
            reconstructed_eigenvector = np.einsum("ijkl,ijkl->jkl",eigenvector_reshaped,transition_density_list)
            reconstructed_eigenvector = reconstructed_eigenvector/(np.einsum("ijk,ijk->",reconstructed_eigenvector,reconstructed_eigenvector)*dV)**(1/2)
            reconstructed_eigenvectors.append(reconstructed_eigenvector)


    self.properties["linear_response_eigenvalues"] = eigenvalues
    self.properties["linear_response_eigenvectors"] = reconstructed_eigenvectors
    self.properties["contribution_linear_response_eigenvectors"] = eigenvectors

    time_taken = time.time() - time_before_calc

    print("##########################################")
    print("# Finished calculating eigenmodes of LRF #")
    print("# in {:3.3f} s #".format(time_taken))
    print("##########################################")
    return eigenvalues, reconstructed_eigenvectors


def calculate_softness_kernel_eigenmodes(self,fukui_type="0",mol_p=None,mol_m=None,grid_points=(100,100,100),delta=10):
    """
    calculate_softness_kernel_eigenmodes(fukui_type="0",mol_p=None,mol_m=None,grid_points=(100,100,100),delta=10)

    Calculates the softness kernel from the transition densities and the fukui function using the Parr-Berkowitz relation.
    For that, a calculation adding (mol_p) or removing an electron (mol_m) needs to be done with the same geometry.
    This method does not calculate directly the softness kernel, but only the eigenmodes.
    The mathematics behind this function will soon be available somewhere...


    Parameters
    ----------
    mol_p : molecule, optional
        The molecule with an electron added. Needed for calculating the softness kernel with a fukui_type of "0" or "+"
    mol_n : molecule, optional
        The molecule with an electron removed. Needed for calculating the softness kernel with a fukui_type of "0" or "-"
    fukui_type : molecule, optional
        The type of fukui function used to calculate the softness kernel. The available types are "0", "+" or "-"
    grid_points : list of 3
    delta : float, optional
        the length added on all directions of the box containing all atomic centers

    Returns
    -------

    linear_response_eigenvalues
        the eigenvalues of the linear response function
    linear_response_eigenvectors
        the eigenvectors of the linear response function

    Notes
    -----
    The linear response function kernel can be computed as :

    :math:`\\chi(r,r') = -2\\sum_{k\\neq0} \\frac{\\rho_0^k(r) \\rho_0^k(r')}{E_k-E_0}`

    With :math:`\\rho_0^k` the transition density, and :math:`E_k` the energy of the transition k.

    Therefore, the molecule needs the transition properties that can be extracted from a TD-DFT calculation, and the MO extracted from a molden file. More details can be found :doc:`here<../tuto/orbitals>`.
    """

    nx,ny,nz = grid_points

    if not "transition_density_list" in self.properties:
        self.calculate_transition_density(grid_points,delta=delta)

    transition_density_list = self.properties["transition_density_list"]
    transition_energy = np.array(self.properties['transition_energy'])

    if not fukui_type in self.properties:
        self.calculate_fukui_function(mol_p=mol_p,mol_m=mol_m,grid_points=grid_points,delta=delta)

    if "0" in fukui_type:
        fukui = self.properties["f0"]
    elif "+" in fukui_type or "p" in fukui_type:
        fukui = self.properties["f+"]
    elif "-" in fukui_type or "m" in fukui_type:
        fukui = self.properties["f-"]

    if not 'hardness' in self.properties:
        self.calculate_hardness()
    hardness = self.properties["hardness"]


    voxel_matrix = self.properties["voxel_matrix"]
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]


    nprocs = rgmol.nprocs

    print("#############################################")
    print("# Calculating Eigenmodes of Softness Kernel #")
    print("#############################################")


    time_before_calc = time.time()
    fukui_reshaped = fukui.reshape((1,nx,ny,nz))
    basis = np.append(transition_density_list,fukui_reshaped,axis=0) #Append the fukui function
    factor = np.array((transition_energy/2).tolist() + [hardness])

    #Append the hardness and divide the energy by 2 to take into account the 2 in the formula of the LRF
    length_basis = len(basis)



    if nprocs >1:
        overlap_matrix = calculate_overlap_matrix_multithread(basis,nprocs,dV)
    else:
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


    # overlap_matrix_inv = np.linalg.inv(overlap_matrix)
    # diag_matrix = overlap_matrix_inv.dot(s_matrix)

    # eigenvalues, eigenvectors = np.linalg.eigh(diag_matrix)


    eigenvalues, eigenvectors = sp.linalg.eigh(s_matrix,overlap_matrix)
    eigenvectors = eigenvectors.transpose()
    reconstructed_eigenvectors = []

    if nprocs >1:
        reconstructed_eigenvectors = reconstruct_eigenvectors(basis,eigenvectors,dV,nprocs)
    else:
        reconstructed_eigenvectors = []
        for eigenvector in eigenvectors:
            eigenvector_reshaped = eigenvector.reshape((length_basis,1,1,1))
            reconstructed_eigenvector = np.einsum("ijkl,ijkl->jkl",eigenvector_reshaped,basis)
            reconstructed_eigenvector = reconstructed_eigenvector/(np.einsum("ijk,ijk->",reconstructed_eigenvector,reconstructed_eigenvector)*dV)**(1/2)
            reconstructed_eigenvectors.append(reconstructed_eigenvector)

    self.properties["softness_kernel_eigenvalues"] = eigenvalues[::-1]
    self.properties["softness_kernel_eigenvectors"] = reconstructed_eigenvectors[::-1]
    self.properties["contribution_softness_kernel_eigenvectors"] = eigenvectors[::-1]

    time_taken = time.time() - time_before_calc

    print("######################################################")
    print("# Finished calculating eigenmodes of Softness Kernel #")
    print("# in {:3.3f} s #".format(time_taken))
    print("######################################################")

    return eigenvalues, reconstructed_eigenvectors



molecule.calculate_fukui_function = calculate_fukui_function
molecule.calculate_hardness = calculate_hardness
molecule.calculate_eigenmodes_linear_response_function = calculate_eigenmodes_linear_response_function
molecule.calculate_softness_kernel_eigenmodes = calculate_softness_kernel_eigenmodes






