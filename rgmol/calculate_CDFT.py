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


def calculate_eigenmodes_linear_response_function(self):
    """
    calculate_eigenmodes_linear_response_function()

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

    if not "transition_density_list" in self.properties:
        self.calculate_transition_density()

    #Dummy implementation for now
    transition_density_list = self.properties["transition_density_list"]
    transition_energy = np.array(self.properties['transition_energy'])

    N_grids,N_trans,N_r,N_ang = np.shape(transition_density_list)

    nprocs = rgmol.nprocs

    print("#################################")
    print("# Calculating Eigenmodes of LRF #")
    print("#################################")
    time_before_calc = time.time()

    if nprocs >1 and 0:
        transition_matrix = calculate_overlap_matrix_multithread(transition_density_list,nprocs,dV)
    else:
        transition_matrix = np.zeros((N_trans,N_trans))
        for first_transition in range(N_trans):
            for second_transition in range(first_transition+1):
                overlap_integral = 0
                for grid,index_grid in zip(self.mol_grids.grids,range(N_grids)):
                    overlap_integral += grid.integrate(transition_density_list[index_grid,first_transition] * transition_density_list[index_grid,second_transition])

                transition_matrix[first_transition,second_transition] = overlap_integral
                transition_matrix[second_transition,first_transition] = overlap_integral
    LR_matrix_in_TDB = np.zeros((N_trans,N_trans))
    #linear response matrix in transition densities basis


    #following dimensions : i,j,k
    #Calculates <rho_0^i|rho_0^j>
    transition_matrix_reshaped = transition_matrix.reshape((N_trans,N_trans,1))
    #Calculates <rho_0^j|rho_0^k>
    transition_matrix_reshaped_2 = transition_matrix.reshape((1,N_trans,N_trans))
    #Used to divide by the energy for j
    transition_energy_reshaped = transition_energy.reshape(1,N_trans,1)

    # LR_matrix_in_TDB = -2*np.einsum("ijk,ijk->ik",transition_matrix_reshaped,transition_matrix_reshaped_2)
    LR_matrix_in_TDB = np.einsum("ijk,ijk,ijk->ik",-2/transition_energy_reshaped,transition_matrix_reshaped,transition_matrix_reshaped_2)


    # transition_matrix_inv = np.linalg.inv(transition_matrix + np.eye(N_trans)*1e-5)
    # diag_matrix = transition_matrix_inv.dot(LR_matrix_in_TDB)
    #
    # eigenvalues, eigenvectors = np.linalg.eigh(diag_matrix)
    eigenvalues, eigenvectors = sp.linalg.eigh(LR_matrix_in_TDB,transition_matrix)
    # eigenvalues, eigenvectors = sp.linalg.eig(LR_matrix_in_TDB,transition_matrix)

    eigenvectors = eigenvectors.transpose()

    # eigvalB,eigvecB = np.linalg.eigh(transition_matrix)
    # eigvecBtilde = eigvecB * eigvalB**(-1/2)
    # Atilde = eigvecBtilde.transpose() @ LR_matrix_in_TDB @ eigvecBtilde
    # eigenvalues, eigvecA = np.linalg.eigh(Atilde)
    # eigenvectors = eigvecBtilde @ eigvecA

    # eigenvectors = eigenvectors.transpose()
    #
    # if nprocs >1 and 0:
    #     reconstructed_eigenvectors = reconstruct_eigenvectors(transition_density_list,eigenvectors,dV,nprocs)
    # else:
    #     reconstructed_eigenvectors = []
    #     for eigenvector in eigenvectors:
    #         eigenvector_reshaped = eigenvector.reshape((N_trans,1,1,1))
    #         reconstructed_eigenvector = np.einsum("ijkl,ijkl->jkl",eigenvector_reshaped,transition_density_list)
    #         # reconstructed_eigenvector = reconstructed_eigenvector/(np.einsum("ijk,ijk->",reconstructed_eigenvector,reconstructed_eigenvector)*dV)**(1/2)
    #         reconstructed_eigenvectors.append(reconstructed_eigenvector)


    # contribution_eigenvectors = eigenvectors / np.einsum("ij,ij->i",eigenvectors,eigenvectors)**(1/2)
    # for k in contribution_eigenvectors:
    #     print(np.sum(k**2))
    contribution_eigenvectors = []
    for k in eigenvectors:
        n = np.sum(k**2)**(1/2)
        contribution_eigenvectors.append(k/n)

    self.properties["linear_response_eigenvalues"] = eigenvalues
    # self.properties["linear_response_eigenvectors"] = reconstructed_eigenvectors
    self.properties["contribution_linear_response_eigenvectors"] = np.array(contribution_eigenvectors)

    time_taken = time.time() - time_before_calc

    print("##########################################")
    print("# Finished calculating eigenmodes of LRF #")
    print("# in {:3.3f} s #".format(time_taken))
    print("##########################################")
    return eigenvalues, np.array(contribution_eigenvectors)


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
                overlap_integral = np.einsum('jkl,jkl->',basis[first_transition],basis[second_transition]) * dV
                overlap_matrix[first_transition,second_transition] = overlap_integral
                overlap_matrix[second_transition,first_transition] = overlap_integral

    #following dimensions : i,k,j
    #Calculates <rho_0^i|rho_0^k>
    overlap_matrix_reshaped = overlap_matrix.reshape((length_basis,length_basis,1))
    #Calculates <rho_0^k|rho_0^j>
    overlap_matrix_reshaped_2 = overlap_matrix.reshape((1,length_basis,length_basis))
    #Used to divide by the energy for k
    factor_reshaped = factor.reshape(1,length_basis,1)

    s_matrix = np.einsum("ikj,ikj,ikj->ij",1/factor_reshaped,overlap_matrix_reshaped,overlap_matrix_reshaped_2)

    # eigvalB,eigvecB = np.linalg.eigh(overlap_matrix)
    # eigvecBtilde = eigvecB * eigvalB**(-1/2)
    # Atilde = eigvecBtilde.transpose() @ s_matrix @ eigvecBtilde
    # eigenvalues, eigvecA = np.linalg.eigh(Atilde)
    # eigenvectors = eigvecBtilde @ eigvecA
    # overlap_matrix_inv = np.linalg.inv(overlap_matrix)
    # diag_matrix = overlap_matrix_inv.dot(s_matrix)
    #
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


#####################################
## Functions to analyze eigenmodes ##
#####################################


def analysis_eigenmodes(self,kernel="linear_response_function",list_vectors=[1,2,3,4,5]):
    """
    do analysis
    """

    if kernel == "linear_response_function":
        eigenvectors = self.properties["linear_response_eigenvectors"]
        eigenvalues = self.properties["linear_response_eigenvalues"]
        contrib_eigenvectors = self.properties["contribution_linear_response_eigenvectors"]
        transition_list = self.properties["transition_list"]
        transition_factor_list = self.properties["transition_factor_list"]

    elif kernel == "softness_kernel":
        eigenvectors = self.properties["softness_kernel_eigenvectors"]
        eigenvalues = self.properties["softness_kernel_eigenvalues"]
        contrib_eigenvectors = self.properties["contribution_softness_kernel_eigenvectors"]
        transition_list = self.properties["transition_list"] + [[[-1,-1]]]
        transition_factor_list = self.properties["transition_factor_list"] + [[[1]]]

    else:
        raise ValueError("The kernel {} is not known or is not implemented. The kernels available are linear_response_function and softness_kernel".format(kernel))

    lin_transition = np.array([transition for transitions in transition_list for transi in transitions for transition in transi])

    occ = lin_transition[::2]
    virt = lin_transition[1::2]
    max_occupied = np.max(occ)
    max_virtual= np.max(virt)


    from_occ_list = []
    to_virt_list = []
    for vector_number in list_vectors:
        if kernel=="softness_kernel":
            #To add fukui function
            transition_contrib = np.zeros((max_occupied+2,max_virtual+2))
            from_occ = np.zeros((max_occupied+2))
        else:
            transition_contrib = np.zeros((max_occupied+1,max_virtual+1))
            from_occ = np.zeros((max_occupied+1))

        contributions = contrib_eigenvectors[vector_number-1]

        for contrib in range(len(contributions)):
            transitions = transition_list[contrib]
            transition_factors = transition_factor_list[contrib]
            for transi,transi_factor in zip(transitions,transition_factors):
                transition_contrib[transi[0],transi[1]] += contributions[contrib] * transi_factor[0]

        for k in range(len(from_occ)):
            for j in range(len(transition_contrib[0])):
                from_occ[k] += transition_contrib[k,j]**2
        from_occ_list.append(from_occ)



        to_virt = np.zeros((max_virtual+1))
        for k in range(len(to_virt)):
            for j in range(len(transition_contrib)):
                to_virt[k] += transition_contrib[j,k]**2
        to_virt_list.append(to_virt)

    fukui_decomposition = []
    if kernel == "softness_kernel":
        for vector_number in range(len(eigenvalues)):
            contributions = contrib_eigenvectors[vector_number-1]

            fukui_decomposition.append(contributions[-1]**2 * self.properties["hardness"] * self.properties["softness_kernel_eigenvalues"][vector_number-1])


    self.properties["from_occ"] = from_occ_list
    self.properties["to_virt"] = to_virt_list
    if kernel == "softness_kernel":
        self.properties["fukui_decomposition"] = fukui_decomposition




molecule.analysis_eigenmodes = analysis_eigenmodes

##TEMPO

def tempo_calculate_overlap_matrix(self,grid_points=(100,100,100),delta=10):
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

    if nprocs > 1:
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

    # LR_matrix_in_TDB = -2*np.einsum("ijk,ijk->ik",transition_matrix_reshaped,transition_matrix_reshaped_2)
    LR_matrix_in_TDB = np.einsum("ijk,ijk,ijk->ik",-2/transition_energy_reshaped,transition_matrix_reshaped,transition_matrix_reshaped_2)


    transition_matrix_inv = np.linalg.inv(transition_matrix + np.eye(number_transition)*1e-10)
    diag_matrix = transition_matrix_inv.dot(LR_matrix_in_TDB)

    eigenvalues, eigenvectors = np.linalg.eigh(diag_matrix)
    # eigenvalues, eigenvectors = sp.linalg.eigh(LR_matrix_in_TDB,transition_matrix)

    return transition_matrix,LR_matrix_in_TDB,eigenvalues,eigenvectors


molecule.tempo_calculate_overlap_matrix = tempo_calculate_overlap_matrix
