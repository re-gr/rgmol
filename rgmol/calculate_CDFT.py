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


def calculate_fukui_function(self,mol_p=None,mol_m=None):
    """
    calculate_fukui_function(mol_p=None,mol_m=None)

    Calculates the fukui function using finite differences between electron density.
    If mol_p is provided, f+ will be computed.
    If mol_m is provided, f- will be computed.
    If both are provided, f+, f- and f0 will be computed.
    The fukui functions are automatically added to the molecule.properties with the keys : f0 f+ and f-

    Parameters
    ----------
    mol_p : molecule, optional
        A molecule with the same atoms and positions, but with an extra electron. By default None
    mol_m : molecule, optional
        A molecule with the same atoms and positions, but with an electron less. By default None

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
        self.calculate_electron_density()
    rhoN = self.properties["electron_density"]

    if mol_p:
        #calculate f+
        if not "electron_density" in mol_p.properties:
            mol_p.calculate_electron_density()

        rhoNp1 = mol_p.properties["electron_density"]

        fp = rhoNp1 - rhoN
    else:
        fp = None

    if mol_m:
        #calculate f-
        if not "electron_density" in mol_m.properties:
            mol_m.calculate_electron_density()

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
        None
            All the data should be inside the molecule. Needs TD-DFT and molden properties

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
    if nprocs >1:
        transition_matrix = calculate_overlap_matrix_multithread(self,transition_density_list,nprocs)
    else:
        transition_matrix = np.zeros((N_trans,N_trans))
        for first_transition in range(N_trans):
            for second_transition in range(first_transition+1):
                overlap_integral = self.mol_grids.integrate_product(transition_density_list[:,first_transition],transition_density_list[:,second_transition])

                transition_matrix[first_transition,second_transition] = overlap_integral
                transition_matrix[second_transition,first_transition] = overlap_integral
    #linear response matrix in transition densities basis
    #following dimensions : i,j,k
    # LR_matrix_in_TDB = np.einsum("k,ik,kj->ij",-2/transition_energy,transition_matrix,transition_matrix)

    # eigenvalues, eigenvectors = np.linalg.eigh(diag_matrix)
    # eigenvalues, eigenvectors = sp.linalg.eigh(LR_matrix_in_TDB,transition_matrix)
    # eigenvectors = eigenvectors.transpose()

    if 0:
        diag,transf = np.linalg.eigh(transition_matrix)
        pij = transf.dot(np.diag(diag)**(1/2)).dot(transf.transpose())

        D = -2*np.einsum("ik,jk,k->ij",pij,pij,1/transition_energy)
        eigenvalues, eigenvectors = np.linalg.eigh(D)
        inv_pij = transf.dot(np.diag(1/diag).dot(transf.transpose()))
        # inv_pij = np.linalg.inv(pij).transpose()
        # eigenvectors = (inv_pij.dot(eigenvectors.transpose()))
        # eigenvectors = (eigenvectors.dot(inv_pij)).transpose()
        eigenvectors = eigenvectors.transpose()



    D = np.diag(-2/transition_energy).dot(transition_matrix)
    eigenvalues,eigenvectors = np.linalg.eig(D)
    eigenvectors = eigenvectors.transpose()


    # return D,transition_matrix

    # eigvalB,eigvecB = np.linalg.eigh(transition_matrix)
    # eigvecBtilde = eigvecB * eigvalB**(-1/2)
    # Atilde = eigvecBtilde.transpose() @ LR_matrix_in_TDB @ eigvecBtilde
    # eigenvalues, eigvecA = np.linalg.eigh(Atilde)
    # eigenvectors = eigvecBtilde @ eigvecA

    # eigenvectors = eigenvectors.transpose()

    if nprocs >1:
        reconstructed_eigenvectors, reconstructed_eigenvector_norms = multithreading_reconstruct_eigenvectors_atomic_grids(self,transition_density_list,eigenvectors,nprocs)
    else:
        reconstructed_eigenvectors = []
        reconstructed_eigenvector_norms = []
        for eigenvector in eigenvectors:
            reconstructed_eigenvector = np.einsum("j,ijkl->ikl",eigenvector,transition_density_list)
            #all eigenvectors are defined at a scalar and the square should integrate to 1
            reconstructed_eigenvector_norm = self.mol_grids.integrate_product(reconstructed_eigenvector,reconstructed_eigenvector)
            reconstructed_eigenvector_norms.append(reconstructed_eigenvector_norm)
            reconstructed_eigenvectors.append(reconstructed_eigenvector/reconstructed_eigenvector_norm**(1/2))

    eigenvectors = np.einsum("ij,i->ij",eigenvectors,1/np.array(reconstructed_eigenvector_norms)**(1/2))
    reconstructed_eigenvectors = np.array(reconstructed_eigenvectors)

    eig_sort = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[eig_sort]
    eigenvectors = eigenvectors[eig_sort]
    reconstructed_eigenvectors = reconstructed_eigenvectors[eig_sort]


    self.properties["linear_response_eigenvalues"] = eigenvalues
    self.properties["contribution_linear_response_eigenvectors"] = np.array(eigenvectors)
    self.properties["linear_response_eigenvectors"] = reconstructed_eigenvectors

    time_taken = time.time() - time_before_calc

    print("##########################################")
    print("# Finished calculating eigenmodes of LRF #")
    print("# in {:3.3f} s #".format(time_taken))
    print("##########################################")
    return eigenvalues, np.array(eigenvectors)


def calculate_softness_kernel_eigenmodes(self,fukui_type="0",mol_p=None,mol_m=None):
    """
    calculate_softness_kernel_eigenmodes(fukui_type="0",mol_p=None,mol_m=None)

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

    transition_density_list = self.properties["transition_density_list"]
    transition_energy = np.array(self.properties['transition_energy'])

    N_grids,N_trans,N_r,N_ang = np.shape(transition_density_list)

    if not fukui_type in self.properties:
        self.calculate_fukui_function(mol_p=mol_p,mol_m=mol_m)

    if "0" in fukui_type:
        fukui = self.properties["f0"]
    elif "+" in fukui_type or "p" in fukui_type:
        fukui = self.properties["f+"]
    elif "-" in fukui_type or "m" in fukui_type:
        fukui = self.properties["f-"]

    if not 'hardness' in self.properties:
        self.calculate_hardness()
    hardness = self.properties["hardness"]


    nprocs = rgmol.nprocs

    print("#############################################")
    print("# Calculating Eigenmodes of Softness Kernel #")
    print("#############################################")


    time_before_calc = time.time()
    fukui_reshaped = fukui.reshape((N_grids,1,N_r,N_ang))
    basis = np.append(transition_density_list,fukui_reshaped,axis=1) #Append the fukui function
    factor = np.array((transition_energy/2).tolist() + [hardness])

    #Append the hardness and divide the energy by 2 to take into account the 2 in the formula of the LRF
    length_basis = len(basis[0])




    if nprocs >1:
        overlap_matrix = calculate_overlap_matrix_multithread(self,basis,nprocs)
    else:
        overlap_matrix = np.zeros((length_basis,length_basis))
        for first_transition in range(length_basis):
            for second_transition in range(first_transition+1):
                overlap_integral = self.mol_grids.integrate_product(basis[:,first_transition],basis[:,second_transition])
                overlap_matrix[first_transition,second_transition] = overlap_integral
                overlap_matrix[second_transition,first_transition] = overlap_integral


    # s_matrix = np.einsum("k,ik,kj->ij",1/factor,overlap_matrix,overlap_matrix)


    # eigvalB,eigvecB = np.linalg.eigh(overlap_matrix)
    # eigvecBtilde = eigvecB * eigvalB**(-1/2)
    # Atilde = eigvecBtilde.transpose() @ s_matrix @ eigvecBtilde
    # eigenvalues, eigvecA = np.linalg.eigh(Atilde)
    # eigenvectors = eigvecBtilde @ eigvecA
    # overlap_matrix_inv = np.linalg.inv(overlap_matrix)
    # diag_matrix = overlap_matrix_inv.dot(s_matrix)
    #
    # eigenvalues, eigenvectors = np.linalg.eigh(diag_matrix)

    # eigenvalues, eigenvectors = sp.linalg.eigh(s_matrix,overlap_matrix)
    D = np.diag(1/factor).dot(overlap_matrix)
    eigenvalues,eigenvectors = np.linalg.eig(D)
    eigenvectors = eigenvectors.transpose()


    if nprocs >1:
        reconstructed_eigenvectors, reconstructed_eigenvector_norms = multithreading_reconstruct_eigenvectors_atomic_grids(self,basis,eigenvectors,nprocs)
    else:
        reconstructed_eigenvectors = []
        reconstructed_eigenvector_norms = []
        for eigenvector in eigenvectors:
            reconstructed_eigenvector = np.einsum("j,ijkl->ikl",eigenvector,basis)
            reconstructed_eigenvector_norm = self.mol_grids.integrate_product(reconstructed_eigenvector,reconstructed_eigenvector)
            reconstructed_eigenvector_norms.append(reconstructed_eigenvector_norm)
            reconstructed_eigenvectors.append(reconstructed_eigenvector/reconstructed_eigenvector_norm**(1/2))
        reconstructed_eigenvectors = np.array(reconstructed_eigenvectors)
        reconstructed_eigenvector_norms = np.array(reconstructed_eigenvector_norms)

    eigenvectors = np.einsum("ij,i->ij",eigenvectors,1/np.array(reconstructed_eigenvector_norms)**(1/2))

    eig_sort = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[eig_sort]
    eigenvectors = eigenvectors[eig_sort]
    reconstructed_eigenvectors = reconstructed_eigenvectors[eig_sort]
    self.properties["softness_kernel_eigenvectors"] = reconstructed_eigenvectors[::-1]
    self.properties["softness_kernel_eigenvalues"] = eigenvalues[::-1]
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
    analysis_eigenmodes(self,kernel="linear_response_function",list_vectors=[1,2,3,4,5])

    Computes some analysis of the eigenmodes of the linear_response_function or the softness_kernel.
    The kernels must be computed beforehand.
    The analysis are : Percentage of occupied MO in the mode, percentage of virtual MO in the mode.
    For the softness kernel, the proportion of electron transferred in each mode.
    These quantities will be stored in mol.properties["from_occ"], "to_virt", and "fukui_decomposition"
    :doc:`plot_analysis<../plot/plot_analysis>` can be used to plot these quantities

    Parameters
    ----------
        kernel : str
            The kernel on which the analysis will be made. Either "linear_reponse_function" or "softness_kernel"
        list_vectors : list, optional
            The list of the vectors that will be decomposed on Occupied Virtual orbitals. Starting to 1 instead of 0. By default [1,2,3,4,5]

    Returns
    -------
        None

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
        from_occ = from_occ/np.sum(from_occ)
        from_occ_list.append(from_occ)


        to_virt = np.zeros((max_virtual+1))
        for k in range(len(to_virt)):
            for j in range(len(transition_contrib)):
                to_virt[k] += transition_contrib[j,k]**2
        to_virt = to_virt/np.sum(to_virt)
        to_virt_list.append(to_virt)

    fukui_decomposition = []
    if kernel == "softness_kernel":
        for vector_number in range(len(eigenvalues)):
            contributions = contrib_eigenvectors[vector_number-1]
            fukui_decomposition.append(contributions[-1]**2 * self.properties["hardness"] * self.properties["softness_kernel_eigenvalues"][vector_number-1])
    f = np.array(fukui_decomposition)



    self.properties["from_occ"] = from_occ_list
    self.properties["to_virt"] = to_virt_list
    if kernel == "softness_kernel":
        self.properties["fukui_decomposition"] = fukui_decomposition

    return


def calculate_polarization(self,kernel,mol_p=None,mol_m=None,fukui_type="0",try_reading_diagonalized=True,save_diagonalized=True,reconstruct=False,new_mo=False):
    """
    """

    if not "external_potential" in self.properties:
        raise TypeError("No external potential found. Must be calculated using mol.set_external_potential or mol.set_external_potential_from_file")

    ext_pot = self.properties["external_potential"]

    if kernel == "linear_response_function":
        # if not "linear_response_eigenvectors" in self.properties:
        #     self.plot_diagonalized_kernel(kernel,number_plotted_eigenvectors=0,try_reading=try_reading_diagonalized,save=save_diagonalized)
        eigenvalues = self.properties["linear_response_eigenvalues"]
        eigenvectors = self.properties["linear_response_eigenvectors"]
        if reconstruct:
            Reconstructed_eigenvectors = self.properties["Reconstructed_linear_response_eigenvectors"]
        contributions = self.properties["contribution_linear_response_eigenvectors"]

    elif kernel == "softness_kernel":
        # if not "softness_kernel_eigenvectors" in self.properties:
        #     self.plot_diagonalized_kernel(kernel,number_plotted_eigenvectors=0,mol_p=mol_p,mol_m=mol_m,fukui_type=fukui_type,try_reading=try_reading_diagonalized,save=save_diagonalized)
        eigenvalues = self.properties["softness_kernel_eigenvalues"]
        eigenvectors = self.properties["softness_kernel_eigenvectors"]
        if reconstruct:
            Reconstructed_eigenvectors = self.properties["Reconstructed_softness_kernel_eigenvectors"]
        contributions = self.properties["contribution_softness_kernel_eigenvectors"]

    else:
        raise ValueError("The kernel {} has not been implemented. The available kernels are : linear_response_function and softness_kernel".format(kernel))



    mol_grids = self.mol_grids
    dV = mol_grids.get_dV()


    transition_list = self.properties["transition_list"]
    transition_factor_list = self.properties["transition_factor_list"]

    transi_occ_max = 0
    transi_virt_max = 0
    num_transition = len(transition_list)
    if new_mo:
        for transition,transition_factor in zip(transition_list,transition_factor_list):
            # print(transition)
            transi_occ = np.max(transition[:,0])
            transi_virt = np.max(transition[:,1])

            if transi_occ_max < transi_occ:
                transi_occ_max = transi_occ
            if transi_virt_max < transi_virt:
                transi_virt_max = transi_virt

        transi_occ_max += 1
        transi_virt_max += 1
        transition_density_coefficients = np.zeros((num_transition,transi_occ_max,transi_virt_max))
        for transition in range(len(transition_list)):
            for factor_index in range(len(transition_list[transition])):
                occ,virt = transition_list[transition][factor_index]
                transition_density_coefficients[transition,occ,virt] = transition_factor_list[transition][factor_index]

        transition_density_coefficients = transition_density_coefficients[:len(contributions)]
        coeffs_transitions = np.einsum("jia,kj->kia",transition_density_coefficients,contributions)


        coef_MO = np.einsum("k,k,kia->ia",eigenvalues,projections,coeffs_transitions)/2

        c, MO = rgmol.rectilinear_grid_reconstruction.reconstruct_MO(self)
        nx,ny,nz = np.shape(MO[0])
        MO_reconstructed = np.zeros((transi_occ_max,nx,ny,nz))
        MO_virt = MO[:transi_virt_max]
        for k in range(transi_occ_max):
            # print(np.shape(coef_MO[k]),np.shape(MO_virt),np.shape(MO_reconstructed),np.shape(MO[k]))
            MO_reconstructed[k] = MO[k] + np.einsum("i,ijkl->jkl",coef_MO[k],MO_virt)

    projections = np.einsum("ijkl,jkl,jkl->i",eigenvectors,ext_pot,dV)
    polarization = np.einsum("i,i,ijkl->jkl",eigenvalues,projections,eigenvectors)

    if reconstruct:
        Reconstructed_polarization = np.einsum("i,i,ijkl->jkl",eigenvalues[:len(Reconstructed_eigenvectors)],projections[:len(Reconstructed_eigenvectors)],Reconstructed_eigenvectors)

    if kernel == "linear_response_function":
        self.properties["linear_response_projection_external_potential"] = projections
        self.properties["linear_response_polarization"] = polarization
        if reconstruct:
            self.properties["Reconstructed_linear_response_polarization"] = Reconstructed_polarization
        if new_mo:
            self.properties["newmo"] = MO_reconstructed

    elif kernel == "softness_kernel":
        self.properties["softness_kernel_projection_external_potential"] = projections
        self.properties["softness_kernel_polarization"] = polarization
        if reconstruct:
            self.properties["Reconstructed_softness_kernel_polarization"] = Reconstructed_polarization
        if new_mo:
            self.properties["newmo"] = MO_reconstructed


    if reconstruct:
        return projections, polarization, Reconstructed_polarization
    return projections, polarization, None






molecule.analysis_eigenmodes = analysis_eigenmodes
molecule.calculate_polarization = calculate_polarization




##TEMPO

#
# def calculate_overlaps(self):
#     """
#     calculate_overlaps()
#
#     Calculates the linear response function from the transition densities.
#     This method does not calculate directly the linear response, but only the eigenmodes.
#     The mathematics behind this function will soon be available somewhere...
#
#
#     Parameters
#     ----------
#     grid_points : list of 3
#     delta : float, optional
#         the length added on all directions of the box containing all atomic centers
#
#     Returns
#     -------
#     linear_response_eigenvalues
#         the eigenvalues of the linear response function
#     linear_response_eigenvectors
#         the eigenvectors of the linear response function
#
#     Notes
#     -----
#     The linear response function kernel can be computed as :
#
#     :math:`\\chi(r,r') = -2\\sum_{k\\neq0} \\frac{\\rho_0^k(r) \\rho_0^k(r')}{E_k-E_0}`
#
#     With :math:`\\rho_0^k` the transition density, and :math:`E_k` the energy of the transition k.
#
#     Therefore, the molecule needs the transition properties that can be extracted from a TD-DFT calculation, and the MO extracted from a molden file. More details can be found :doc:`here<../tuto/orbitals>`.
#     """
#
#     if not "transition_density_list" in self.properties:
#         self.calculate_transition_density()
#
#     #Dummy implementation for now
#     transition_density_list = self.properties["transition_density_list"]
#     transition_energy = np.array(self.properties['transition_energy'])
#
#     N_grids,N_trans,N_r,N_ang = np.shape(transition_density_list)
#
#     nprocs = rgmol.nprocs
#
#     print("#################################")
#     print("# Calculating Eigenmodes of LRF #")
#     print("#################################")
#     time_before_calc = time.time()
#
#     if nprocs >1:
#         transition_matrix = calculate_overlap_matrix_multithread(self,transition_density_list,nprocs)
#     else:
#         transition_matrix = np.zeros((N_trans,N_trans))
#         for first_transition in range(N_trans):
#             for second_transition in range(first_transition+1):
#                 overlap_integral = 0
#                 for grid,index_grid in zip(self.mol_grids.grids,range(N_grids)):
#                     overlap_integral += grid.integrate(transition_density_list[index_grid,first_transition] * transition_density_list[index_grid,second_transition])
#
#                 transition_matrix[first_transition,second_transition] = overlap_integral
#                 transition_matrix[second_transition,first_transition] = overlap_integral
#     D = np.diag(-2/transition_energy).dot(transition_matrix)
#
#     return D
#
# molecule.calculate_overlaps = calculate_overlaps