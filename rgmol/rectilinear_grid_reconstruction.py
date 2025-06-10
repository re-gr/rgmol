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
from rgmol.molecular_calculations import *
from rgmol.grid import *



def reconstruct_AO(mol,grid_points=(80,80,80),delta=5):
    """
    calculate_AO(mol,grid_points=(80,80,80),detla=5)

    Calculate all atomic orbitals for a molecule on the representation grid
    If no representation grids were associated with the molecule, it will automatically create it

    Parameters
    ----------
        mol : molecule
            the molecule
        grid_points : list of 3, optional
            the number of points on the representation grid
        delta : float, optional
            the length added on all directions to the box containing all atomic centers

    Returns
    -------
        coords, AO_calculated
    """

    if "Reconstructed_AO" in mol.properties:
        return mol.properties["Reconstructed_coords"],mol.properties["Reconstructed_AO"]

    # r,c = create_rectilinear_grid_from_molecule(mol,grid_points=grid_points)
    r,c = create_cubic_grid_from_molecule(mol,grid_points=grid_points,delta=delta)


    if not "AO_list" in mol.properties:
        raise TypeError("The Atomic Orbital functions were not found.")

    AO_list = mol.properties["AO_list"]
    AO_type_list = mol.properties["AO_type_list"]

    AO_calculated = []
    print("##################################")
    print("# Reconstructing Atomic Orbitals #")
    print("#       On Rectilinear Grid      #")
    print("##################################")
    time_before_calc = time.time()

    for AO_atom in zip(mol.atoms,AO_list,AO_type_list):
        r0 = AO_atom[0].pos
        for AO in zip(AO_atom[1],AO_atom[2]):
            exponent_primitives,contraction_coefficients=np.array(AO[0]).transpose()
            if AO[1] == "s":
                AO_calc = gaussian_s(r,contraction_coefficients,exponent_primitives,r0)
            elif AO[1] == "p":
                AO_calc = gaussian_p(r,contraction_coefficients,exponent_primitives,r0)
            elif AO[1] == "d":
                AO_calc = gaussian_d(r,contraction_coefficients,exponent_primitives,r0)
            elif AO[1] == "f":
                AO_calc = gaussian_f(r,contraction_coefficients,exponent_primitives,r0)
            elif AO[1] == "g":
                AO_calc = gaussian_g(r,contraction_coefficients,exponent_primitives,r0)
            else: raise ValueError("This type of orbital has not been yet implemented")

            for ao in AO_calc:
                AO_calculated.append(ao)

    time_taken = time.time() - time_before_calc
    print("###########################################")
    print("# Finished Reconstructing Atomic Orbitals #")
    print("#               in {:3.3f} s              #".format(time_taken))
    print("###########################################")

    mol.properties["Reconstructed_coords"] = c
    mol.properties["Reconstructed_AO"] = np.array(AO_calculated)

    return c,np.array(AO_calculated)


def reconstruct_chosen_MO(mol,MO_chosen,grid_points=(80,80,80),delta=5):
    """
    calculate_MO(mol,grid_points=(80,80,80),delta=5)

    Calculate all molecular orbitals for a molecule on the representation grid
    If no representation grids were associated with the molecule, it will automatically create it
    If the AO were not calculated on this grid, it will also calculate them

    Parameters
    ----------
        mol : molecule
            the molecule
        grid_points : list of 3, optional
            the number of points on the representation grid
        delta : float, optional
            the length added on all directions to the box containing all atomic centers

    Returns
    -------
        coords, MO_calculated
    """

    if "Reconstructed_MO" in mol.properties:
        return mol.properties["Reconstructed_coords"],mol.properties["Reconstructed_MO"]

    coords,AO_calculated = reconstruct_AO(mol,grid_points=grid_points,delta=delta)

    MO_list = np.array(mol.properties["MO_list"])
    MO = MO_list[MO_chosen]
    MO_not_normalized = np.einsum("ijkl,i->jkl",AO_calculated,MO)

    return coords,MO_not_normalized


def reconstruct_MO(mol,grid_points=(80,80,80),delta=5):
    """
    calculate_MO(mol,grid_points=(80,80,80),delta=5)

    Calculate all molecular orbitals for a molecule on the representation grid
    If no representation grids were associated with the molecule, it will automatically create it
    If the AO were not calculated on this grid, it will also calculate them

    Parameters
    ----------
        mol : molecule
            the molecule
        grid_points : list of 3, optional
            the number of points on the representation grid
        delta : float, optional
            the length added on all directions to the box containing all atomic centers

    Returns
    -------
        coords, MO_calculated
    """

    if "Reconstructed_MO" in mol.properties:
        return mol.properties["Reconstructed_coords"],mol.properties["Reconstructed_MO"]

    coords,AO_calculated = reconstruct_AO(mol,grid_points=grid_points,delta=delta)

    occ = mol.properties["MO_occupancy"][0]

    print('#####################################')
    print("# Reconstructing Molecular Orbitals #")
    print("#        On Rectilinear Grid        #")
    print('#####################################')

    time_before_calc = time.time()

    MO_list = np.array(mol.properties["MO_list"])
    MO_calculated = np.einsum("ij,jklm->iklm",MO_list,AO_calculated)
    MO_norm = np.einsum("ijkl,ijkl->i", MO_calculated, MO_calculated) / occ
    MO_norm = MO_norm.reshape((len(MO_norm),1,1,1))
    MO_calculated = MO_calculated / MO_norm **(1/2)


    time_taken = time.time() - time_before_calc

    print("##############################################")
    print("# Finished Reconstructing Molecular Orbitals #")
    print("#                in {:3.3f} s                #".format(time_taken))
    print("##############################################")

    mol.properties["Reconstructed_coords"] = coords
    mol.properties["Reconstructed_MO"] = np.array(MO_calculated)

    return coords,np.array(MO_calculated)


def reconstruct_electron_density(mol,grid_points=(80,80,80),delta=5):
    """
    calculate_electron_density(mol,grid_points=(80,80,80),delta=5)

    Calculates the electron density for a molecule on the representation grid
    If no representation grids were associated with the molecule, it will automatically create it
    If the MO were not calculated on this grid, it will also calculate them

    Parameters
    ----------
        mol : molecule
            the molecule
        grid_points : list of 3, optional
            the number of points on the representation grid
        delta : float, optional
            the length added on all directions to the box containing all atomic centers

    Returns
    -------
        coords, electron_density
    """

    coords,MO_calculated = reconstruct_MO(mol,grid_points=grid_points,delta=delta)


    MO_occ = np.array(mol.properties["MO_occupancy"])
    MO_occ_arr = MO_occ>0
    MO_occ_index = np.argmin(MO_occ)

    MO_occ = MO_occ[MO_occ_arr]

    MO_occupied = MO_calculated[:MO_occ_index]
    MO_occ_occupied = MO_occ[:MO_occ_index]


    electron_density = np.einsum("ijkl,ijkl->jkl",MO_occupied,MO_occupied)

    mol.properties["Reconstructed_electron_density"] = electron_density

    return coords, electron_density



def reconstruct_fukui_function(mol,mol_p=None,mol_m=None,grid_points=(80,80,80),delta=5):
    """
    calculate_fukui_function(mol,mol_p=None,mol_m=None,grid_points=(80,80,80),delta=5)

    Calculates the fukui function using finite differences between electron density on the representation grid.
    If mol_p is provided, f+ will be computed.
    If mol_m is provided, f- will be computed.
    If both are provided, f+, f- and f0 will be computed.
    The fukui functions are automatically added to the molecule.properties with the keys : f0 f+ and f-

    Parameters
    ----------
        mol : molecule
            the molecule
        grid_points : list of 3, optional
            the number of points on the representation grid
        delta : float, optional
            the length added on all directions to the box containing all atomic centers

    Returns
    -------
        coords, f0, fp, fm

    Notes
    -----

    :math:`f^+ = \\rho(N+1)-\\rho(N)`
    :math:`f^- = \\rho(N)-\\rho(N-1)`
    :math:`f^0 = (f^+ + f^-)/2`
    """

    if not(mol_p) and not(mol_m):
        raise TypeError("Need at least the molecule with N+1 or N-1 number of electrons")

    if not "Reconstructed_electron_density" in mol.properties:
        coords,elec_dens = reconstruct_electron_density(mol,grid_points=grid_points,delta=delta)
    rhoN = mol.properties["Reconstructed_electron_density"]

    if mol_p:
        #calculate f+
        if not "Reconstructed_electron_density" in mol_p.properties:
            reconstruct_electron_density(mol_p,grid_points=grid_points,delta=delta)

        rhoNp1 = mol_p.properties["Reconstructed_electron_density"]

        fp = rhoNp1 - rhoN
    else:
        fp = None

    if mol_m:
        #calculate f-
        if not "Reconstructed_electron_density" in mol_m.properties:
            reconstruct_electron_density(mol_m,grid_points=grid_points,delta=delta)

        rhoNm1 = mol_m.properties["Reconstructed_electron_density"]

        fm = rhoN - rhoNm1

    else:
        fm = None

    if mol_p and mol_m:
        f0 = (fp+fm)/2
    else: f0 = None

    return coords,f0,fp,fm


####################
## EXCITED STATES ##
####################


def reconstruct_transition_density(mol,grid_points=(80,80,80),delta=5):
    """
    calculate_transition_density(mol,grid_points,grid_points=(80,80,80),delta=5)

    Calculates all the transition densities for a molecule on the representation grid

    Parameters
    ----------
        mol : molecule
            the molecule
        grid_points : list of 3, optional
            the number of points on the representation grid
        delta : float, optional
            the length added on all directions to the box containing all atomic centers

    Returns
    -------
        coords, transition_density_list

    Notes
    -----
    The transition densities are defined as

    | :math:`\\rho_0^k = \\sum_i c_i (Occ_i(r) * Virt_i(r))`
    | With the sum being on all the transitions of the excitation, :math:`Occ_i(r)` and :math:`Virt_i(r)` being respectively the occupied and the virtual molecular orbitals considered in the transition, and :math:`c_i` the coefficient of the transition.

    Examples
    --------
    For a TD-DFT calculation output :

    STATE  1:  E=   0.148431 au      4.039 eV
       | 18a ->  20a  :     0.5000 (c= 0.7071)
       | 19a ->  21a  :     0.5000 (c= -0.7071)

    The transition density will be :
        | :math:`\\rho_0^1 =  c_1 (\\psi_{18}(r) * \\psi_{20}(r)) + c_2 (\\psi_{19}(r) * \\psi_{21}(r))`
        | with :math:`c_1 = 0.7071` and :math:`c_2 = -0.7071`
    """

    if not "transition_list" in mol.properties:
        raise ValueError("No transitions were found, one should use rgmol.extract_excited_states.extract_transition to extract transition")

    transition_list = mol.properties["transition_list"]
    transition_factor_list = mol.properties["transition_factor_list"]

    coords,MO_calculated = reconstruct_MO(mol,grid_points=grid_points,delta=delta)

    nprocs = rgmol.nprocs
    nx,ny,nz = grid_points


    print("#####################################")
    print("# Reconstructing Transition Density #")
    print("#####################################")

    time_before_calc = time.time()

    #There should be a way to speed up the calculation
    #This can be done by geting an array of the coeffs with 0 on those who are not there and then doing the einsum
    transi_occ_max = 0
    transi_virt_max = 0
    num_transition = len(transition_list)

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
    transitions = np.zeros((transi_occ_max,transi_virt_max))
    transition_density_coefficients = np.zeros((num_transition,transi_occ_max,transi_virt_max))
    count = 0
    for transition in range(len(transition_list)):
        for factor_index in range(len(transition_list[transition])):
            occ,virt = transition_list[transition][factor_index]
            transitions[occ,virt] = 1
            transition_density_coefficients[transition,occ,virt] = transition_factor_list[transition][factor_index]
    MO_occ = MO_calculated[:transi_occ_max]
    print(np.shape(transition_density_coefficients),np.shape(MO_occ),np.shape(MO_calculated))


    # if nprocs > 1 and 0:
    if nprocs > 1:
        transition_density_list = calculate_transition_density_multithread(mol,transitions,transition_density_coefficients,grid_points,nprocs)
    else:
        # transition_density_list = np.einsum("ijk,jlmn,klmn->ilmn",transition_density_coefficients,MO_occ,MO_calculated,optimize=True)


        transition_density_list = np.zeros((len(transition_list),nx,ny,nz))
        for occ in range(transi_occ_max):
            for virt in range(transi_virt_max):
                if transitions[occ,virt]:

                    MO_OCC = MO_calculated[occ]
                    MO_VIRT = MO_calculated[virt]
                    MO_product = MO_OCC * MO_VIRT

                    transition_coeffs = transition_density_coefficients[:,occ,virt]
                    for transition in range(num_transition):
                        coeff = transition_coeffs[transition]
                        if coeff!=0:
                            transition_density_list[transition] = transition_density_list[transition] + coeff * MO_product


    mol.properties["Reconstructed_transition_density_list"] = transition_density_list


    time_taken = time.time()-time_before_calc
    print("##############################################")
    print("# Finished Reconstructing Transition Density #")
    print('# in {:3.3f} s #'.format(time_taken))
    print("##############################################")
    return coords,transition_density_list


def reconstruct_chosen_transition_density(mol,chosen_transition_density,grid_points=(80,80,80),delta=5):
    """
    calculate_chosen_transition_density(mol,chosen_transition_density,grid_points=(80,80,80),delta=5)

    Calculates a chosen transition density for a molecule on the representation grid

    Parameters
    ----------
        mol : molecule
            the molecule
        chosen_transition_density : int
            the index of the chosen transition_density
        grid_points : list of 3, optional
            the number of points on the representation grid
        delta : float, optional
            the length added on all directions to the box containing all atomic centers

    Returns
    -------
        transition_density

    Notes
    -----
    The transition densities are defined as :

    | :math:`\\rho_0^k = \\sum_i c_i (Occ_i(r) * Virt_i(r))`
    | With the sum being on all the transitions of the excitation, :math:`Occ_i(r)` and :math:`Virt_i(r)` being respectively the occupied and the virtual molecular orbitals considered in the transition, and :math:`c_i` the coefficient of the transition.

    Examples
    --------
    For a TD-DFT calculation output :

    STATE  1:  E=   0.148431 au      4.039 eV
       | 18a ->  20a  :     0.5000 (c= 0.7071)
       | 19a ->  21a  :     0.5000 (c= -0.7071)

    The transition density will be :
        | :math:`\\rho_0^1 =  c_1 (\\psi_{18}(r) * \\psi_{20}(r)) + c_2 (\\psi_{19}(r) * \\psi_{21}(r))`
        | with :math:`c_1 = 0.7071` and :math:`c_2 = -0.7071`
    """

    if not "transition_list" in mol.properties:
        raise ValueError("No transitions were found, one should use rgmol.extract_excited_states.extract_transition to extract transition")

    coords,MO_calculated = reconstruct_MO(mol,grid_points=grid_points,delta=delta)

    #Initialize transition density list
    if not "Reconstructed_transition_density_list" in mol.properties:
        mol.properties["Reconstructed_transition_density_list"] = [[] for k in range(len(mol.properties["transition_list"]))]

    if type(mol.properties["Reconstructed_transition_density_list"][chosen_transition_density]) is not list:
        return mol.properties["Reconstructed_transition_density_list"][chosen_transition_density]


    transition_list = mol.properties["transition_list"][chosen_transition_density]
    transition_factor_list = mol.properties["transition_factor_list"][chosen_transition_density]


    transition_factor_list = np.array(transition_factor_list)
    transition_factor_list /= np.sum(transition_factor_list**2)**(1/2)

    nx,ny,nz = grid_points
    transition_density = np.zeros((nx,ny,nz))


    for transition_MO in zip(transition_list, transition_factor_list):

        MO_OCC = MO_calculated[transition_MO[0][0]]
        MO_VIRT = MO_calculated[transition_MO[0][1]]

        transition_density += transition_MO[1][0] * MO_OCC * MO_VIRT

    mol.properties["Reconstructed_transition_density_list"][chosen_transition_density] = transition_density
    return coords,transition_density


def reconstruct_eigenvectors(mol,kernel,grid_points=(80,80,80),delta=5,mol_p=None,mol_m=None,fukui_type="0"):
    """
    reconstruct_eigenvectors(mol,kernel,grid_points=(80,80,80),delta=5,mol_p=None,mol_m=None,fukui_type="0")

    Calculates the eigenmodes of a kernel for a molecule on the representation grid

    Parameters
    ----------
        mol : molecule
            the molecule
        grid_points : list of 3, optional
            the number of points on the representation grid
        delta : float, optional
            the length added on all directions to the box containing all atomic centers
        mol_p : molecule, optional
            The molecule with an electron added. Needed for calculating the softness kernel with a fukui_type of "0" or "+"
        mol_m : molecule, optional
            The molecule with an electron removed. Needed for calculating the softness kernel with a fukui_type of "0" or "-"
        fukui_type : molecule, optional
            The type of fukui function used to calculate the softness kernel. The available types are "0", "+" or "-"

    Returns
    -------
        coords, reconstructed_eigenvectors
    """
    nprocs = rgmol.nprocs
    nx,ny,nz = grid_points
    if kernel == "linear_response_function":
        if "Reconstructed_linear_response_eigenvectors" in mol.properties:
            return mol.properties["Reconstructed_coords"],mol.properties["Reconstructed_linear_response_eigenvectors"]
        contribution_eigenvectors = mol.properties["contribution_linear_response_eigenvectors"]

        coords,Reconstructed_transition_density_list = reconstruct_transition_density(mol,grid_points=grid_points,delta=delta)
        N_trans,nx,ny,nz = np.shape(Reconstructed_transition_density_list)

    if kernel == "softness_kernel":
        if "Reconstructed_softness_kernel_eigenvectors" in mol.properties:
            return mol.properties["Reconstructed_coords"],mol.properties["Reconstructed_softness_kernel_eigenvectors"]
        contribution_eigenvectors = mol.properties["contribution_softness_kernel_eigenvectors"]

        coords,Reconstructed_transition_density_list = reconstruct_transition_density(mol,grid_points=grid_points,delta=delta)
        cooords, f0, fp, fm = reconstruct_fukui_function(mol,mol_p=mol_p,mol_m=mol_m,grid_points=grid_points,delta=delta)

        if "0" in fukui_type:
            Reconstructed_transition_density_list = np.append(Reconstructed_transition_density_list,f0.reshape((1,nx,ny,nz)),axis=0)
        elif "+" in fukui_type or "p" in fukui_type:
            Reconstructed_transition_density_list = np.append(Reconstructed_transition_density_list,fp.reshape((1,nx,ny,nz)),axis=0)
        elif "-" in fukui_type or "m" in fukui_type:
            Reconstructed_transition_density_list = np.append(Reconstructed_transition_density_list,fm.reshape((1,nx,ny,nz)),axis=0)

        N_trans,nx,ny,nz = np.shape(Reconstructed_transition_density_list)


    print("###############################")
    print("# Reconstructing Eigenvectors #")
    print("###############################")
    time_before_calc = time.time()

    if nprocs >1:
        reconstructed_eigenvectors = multithreading_reconstruct_eigenvectors(Reconstructed_transition_density_list,contribution_eigenvectors,nprocs)
    else:
        reconstructed_eigenvectors = []
        for eigenvector in contribution_eigenvectors:
            eigenvector_reshaped = eigenvector.reshape((N_trans,1,1,1))
            reconstructed_eigenvector = np.einsum("ijkl,ijkl->jkl",eigenvector_reshaped,Reconstructed_transition_density_list)
            reconstructed_eigenvectors.append(reconstructed_eigenvector)
    if kernel == "linear_response_function":
        mol.properties["Reconstructed_linear_response_eigenvectors"] = np.array(reconstructed_eigenvectors)
    elif kernel == "softness_kernel":
        mol.properties["Reconstructed_softness_kernel_eigenvectors"] = np.array(reconstructed_eigenvectors)


    time_taken = time.time()-time_before_calc
    print("########################################")
    print("# Finished Reconstructing Eigenvectors #")
    print('# in {:3.3f} s #'.format(time_taken))
    print("########################################")

    return coords,np.array(reconstructed_eigenvectors)