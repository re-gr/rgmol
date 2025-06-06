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
from rgmol.grid import *

def gaussian_s(r,contraction_coefficients,exponent_primitives,r0):
    """
    gaussian_s(r,contraction_coefficients,exponent_primitives,r0)

    Returns the calculations for the gaussian s atomic orbital

    Parameters
    ----------
        r : 3d ndarray
            the volume on which the function is calculated
        contraction_coefficients : list
            the contraction coefficient
        exponent_primitives : list
            the exponential factor
        r0 : ndarray of length 3
            the center of the orbital

    Returns
    -------
        s_orbital : ndarray same shape as r

    """

    sum_gaussian = np.zeros(np.shape(r)[1:])
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0
    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian +=  coefs[0] * np.exp(-coefs[1]*r_r)


    return sum_gaussian,

def gaussian_p(r,contraction_coefficients,exponent_primitives,r0):
    """
    gaussian_p(r,contraction_coefficients,exponent_primitives,r0)

    Returns the calculations for the gaussian p atomic orbital
    The order of the orbitals follows the molden format

    Parameters
    ----------
        r : 3d ndarray
            the volume on which the function is calculated
        contraction_coefficients : list
            the contraction coefficients
        exponent_primitives : list
            the exponential factors
        r0 : ndarray of length 3
            the center of the orbital

    Returns
    -------
        p_x_orbital,
        p_y_orbital,
        p_z_orbital
    """

    sum_gaussian_x = np.zeros(np.shape(r)[1:])
    sum_gaussian_y = np.zeros(np.shape(r)[1:])
    sum_gaussian_z = np.zeros(np.shape(r)[1:])
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0

    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian_x += x*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_y += y*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_z += z*coefs[0] * np.exp(-coefs[1]*r_r)


    return sum_gaussian_x,sum_gaussian_y,sum_gaussian_z


def gaussian_d(r,contraction_coefficients,exponent_primitives,r0):
    """
    gaussian_d(r,contraction_coefficients,exponent_primitives,r0)

    Returns the calculations for the gaussian d atomic orbital
    The order of the orbitals follows the molden format

    Parameters
    ----------
        r : 3d ndarray
            the volume on which the function is calculated
        contraction_coefficients : list
            the contraction coefficients
        exponent_primitives : list
            the exponential factors
        r0 : ndarray of length 3
            the center of the orbital

    Returns
    -------
        d_zz_orbital,
        d_xz_orbital,
        d_yz_orbital,
        d_xx_yy_orbital,
        d_xy_orbital
    """

    sum_gaussian_xy = np.zeros(np.shape(r)[1:])
    sum_gaussian_xz = np.zeros(np.shape(r)[1:])
    sum_gaussian_yz = np.zeros(np.shape(r)[1:])
    sum_gaussian_xx_yy = np.zeros(np.shape(r)[1:])
    sum_gaussian_zz = np.zeros(np.shape(r)[1:])
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0
    r0_reshaped = np.array(r0).reshape(3,1,1,1)

    x_x = x*x
    y_y = y*y
    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian_xy += (x*y)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xz += (x*z)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_yz += (y*z)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xx_yy += (x_x-y_y)/2*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zz += (z*z-x_x/2-y_y/2)/3**(1/2) * coefs[0] * np.exp(-coefs[1]*r_r)

    return sum_gaussian_zz, sum_gaussian_xz, sum_gaussian_yz, sum_gaussian_xx_yy, sum_gaussian_xy



def gaussian_f(r,contraction_coefficients,exponent_primitives,r0):
    """
    gaussian_f(r,contraction_coefficients,exponent_primitives,r0)

    Returns the calculations for the gaussian f atomic orbital
    The order of the orbitals follows the molden format

    Parameters
    ----------
        r : 3d ndarray
            the volume on which the function is calculated
        contraction_coefficients : list
            the contraction coefficients
        exponent_primitives : list
            the exponential factors
        r0 : ndarray of length 3
            the center of the orbital

    Returns
    -------
        f_zzz_zrr_orbital,
        f_xzz_xrr_orbital,
        f_yzz_yrr_orbital,
        f_zxx_zyy_orbital,
        f_xyz_orbital,
        f_xxx_xyy_orbital,
        f_yyy_xxy_orbital
    """
    sum_gaussian_yyy_xxy = np.zeros(np.shape(r)[1:])
    sum_gaussian_xyz = np.zeros(np.shape(r)[1:])
    sum_gaussian_yzz_yrr = np.zeros(np.shape(r)[1:])
    sum_gaussian_zzz_zrr = np.zeros(np.shape(r)[1:])
    sum_gaussian_xzz_xrr = np.zeros(np.shape(r)[1:])
    sum_gaussian_zxx_zyy = np.zeros(np.shape(r)[1:])
    sum_gaussian_xxx_xyy = np.zeros(np.shape(r)[1:])
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0
    r0_reshaped = np.array(r0).reshape(3,1,1,1)

    x_x = x*x
    y_y = y*y
    z_z = z*z
    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian_yyy_xxy += (y_y-3*x_x)/(2*6**(1/2))*y*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xyz += (x*y*z)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_yzz_yrr += (5*z_z-r_r)/(2*10**(1/2))*y*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zzz_zrr += (5*z_z-3*r_r) /(2*15**(1/2))  *z*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xzz_xrr += (5*z_z-r_r)/(2*10**(1/2))*x*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zxx_zyy += (x_x-y_y)/2*z*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xxx_xyy += (3*y_y-x_x)/(2*6**(1/2))*x*coefs[0] * np.exp(-coefs[1]*r_r)


    return sum_gaussian_zzz_zrr, sum_gaussian_xzz_xrr, sum_gaussian_yzz_yrr, sum_gaussian_zxx_zyy, sum_gaussian_xyz, sum_gaussian_xxx_xyy, sum_gaussian_yyy_xxy




def gaussian_g(r,contraction_coefficients,exponent_primitives,r0):
    """
    gaussian_g(r,contraction_coefficients,exponent_primitives,r0)

    Returns the calculations for the gaussian g atomic orbital
    The order of the orbitals follows the molden format

    Parameters
    ----------
        r : 3d ndarray
            the volume on which the function is calculated
        contraction_coefficients : list
            the contraction coefficients
        exponent_primitives : list
            the exponential factors
        r0 : ndarray of length 3
            the center of the orbital

    Returns
    -------
        g_zzzz_orbital,
        g_zzzx_orbital,
        g_zzzy_orbital,
        g_zz_xx_yy_orbital,
        g_zzxy_orbital,
        g_zxxx_orbital,
        g_zyyy_orbital,
        g_xxxx_yyyy_orbital,
        g_xy_xx_yy_orbital
    """
    sum_gaussian_zzzz = np.zeros(np.shape(r)[1:])
    sum_gaussian_zzzy = np.zeros(np.shape(r)[1:])
    sum_gaussian_zzzx = np.zeros(np.shape(r)[1:])
    sum_gaussian_zzxy = np.zeros(np.shape(r)[1:])
    sum_gaussian_zz_xx_yy = np.zeros(np.shape(r)[1:])
    sum_gaussian_zyyy = np.zeros(np.shape(r)[1:])
    sum_gaussian_zxxx = np.zeros(np.shape(r)[1:])
    sum_gaussian_xy_xx_yy = np.zeros(np.shape(r)[1:])
    sum_gaussian_xxxx_yyyy = np.zeros(np.shape(r)[1:])
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0
    r0_reshaped = np.array(r0).reshape(3,1,1,1)

    x_x = x*x
    y_y = y*y
    z_z = z*z
    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian_zzzz += (35*z_z*z_z - 30*z_z*r_r + 3*r_r*r_r)/(2*(16*35)**(1/2))*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zzzy += y*z*(7*z_z-3*r_r)/(2*(2*7)**(1/2))*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zzzx += x*z*(7*z_z-3*r_r)/(2*(2*7)**(1/2))*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zzxy += x*y*(7*z_z-r_r)/(2*(7)**(1/2))*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zz_xx_yy += (x_x-y_y)*(7*z_z-r_r)/(2*(2*2*7)**(1/2))*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zyyy += y*z*(y_y-3*x_x)/(2*(2)**(1/2))*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zxxx += x*z*(3*y_y-x_x)/(2*(2)**(1/2))*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xy_xx_yy += x*y*(y_y-x_x)/(2)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xxxx_yyyy += (6*x_x*y_y-x_x*x_x-y_y*y_y)/(2*(16)**(1/2))*coefs[0] * np.exp(-coefs[1]*r_r)


    return sum_gaussian_zzzz, sum_gaussian_zzzx, sum_gaussian_zzzy, sum_gaussian_zz_xx_yy, sum_gaussian_zzxy, sum_gaussian_zxxx, sum_gaussian_zyyy, sum_gaussian_xxxx_yyyy, sum_gaussian_xy_xx_yy



def calculate_AO(self):
    """
    calculate_AO()

    Calculate all atomic orbitals for a molecule and puts it in molecule.properties["AO_calculated_list"]
    If no grids were associated with the molecule, it will automatically create all atomic grids

    Parameters
    ----------
        None
            If no grids were computed, a new one will be automatically created

    Returns
    -------
        AO_calculated_list : list of ndarray
    """

    if not self.mol_grids:
        create_grid_from_mol(self)

    mol_grids = self.mol_grids
    N_grids = len(mol_grids.grids)

    if not "AO_list" in self.properties:
        raise TypeError("The Atomic Orbital functions were not found.")

    AO_list = self.properties["AO_list"]
    AO_type_list = self.properties["AO_type_list"]
    AO_calculated_list = []

    print("###############################")
    print("# Calculating Atomic Orbitals #")
    print("###############################")
    time_before_calc = time.time()

    Ints = []
    for grid in mol_grids.grids:
        r = grid.xyz_coords

        AO_calculated = []

        for AO_atom in zip(self.atoms,AO_list,AO_type_list):
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
                    Ints.append(grid.integrate_product(ao,ao))
        AO_calculated_list.append(np.array(AO_calculated))

    Ints = np.array(Ints).reshape((N_grids,len(AO_calculated)))
    Ints = np.sum(Ints,axis=0).reshape((1,len(AO_calculated),1,1))
    #Renormalize the results. They should be around 1-1e-10 before normalization though
    # AO_calculated_list = AO_calculated_list / Ints**(1/2)

    self.properties["AO_calculated_list"] = np.array(AO_calculated_list)

    time_taken = time.time() - time_before_calc
    text_finished = "# Finished Calculating Atomic Orbitals #"
    text_time = "# in {:3.3f} s #".format(time_taken)
    numb_di = max(len(text_finished),len(text_time))
    print("#"*numb_di)
    print(text_finished)
    print(text_time)
    print("#"*numb_di)
    return np.array(AO_calculated_list)



def calculate_MO(self):
    """
    calculate_MO()

    Calculate all molecular orbitals for a molecule and puts it in molecule.properties["MO_calculated_list"]

    If no grids were associated with the molecule, it will automatically create all atomic grids
    If the AO were not calculated it will also calculate them

    Parameters
    ----------
        None
            If no grids were computed, a new one will be automatically created

    Returns
    -------
        MO_calculated_list : list of ndarray
    """

    if not "AO_calculated_list" in self.properties:
        self.calculate_AO()

    AO_calculated_list = self.properties["AO_calculated_list"]

    MO_calculated_list = []
    MO_list = np.array(self.properties["MO_list"])
    MO_occupancy = self.properties["MO_occupancy"][0]
    N_grids,N_AO,N_r,N_ang = np.shape(AO_calculated_list)


    print('##################################')
    print("# Calculating Molecular Orbitals #")
    print('##################################')

    time_before_calc = time.time()
    Ints = []
    for MO in MO_list:
        MO_not_normalized = np.einsum("ijkl,j->ikl",AO_calculated_list,MO)
        MO_calculated_list.append(MO_not_normalized)
        Ints.append(self.mol_grids.integrate_product(MO_not_normalized,MO_not_normalized))

    N_MO = len(MO_list)

    #Convert to array an transpose to the dimensions : grids,MO,radial,angular
    MO_calculated_list = np.array(MO_calculated_list).transpose((1,0,2,3))
    #Same for int for the two first dimensions
    Ints = np.array(Ints).reshape((1,N_MO,1,1))

    #Renormalize the results. They should be around 1-1e-10 before normalization though
    MO_calculated_list = MO_calculated_list / Ints**(1/2) * MO_occupancy**(1/2)

    self.properties["MO_calculated_list"] = np.array(MO_calculated_list)

    time_taken = time.time() - time_before_calc

    print("###########################################")
    print("# Finished Calculating Molecular Orbitals #")
    print("# in {:3.3f} s #".format(time_taken))
    print("###########################################")

    return np.array(MO_calculated_list)




def calculate_MO_chosen(self,MO_chosen):
    """
    calculate_MO_chosen(MO_chosen,grid_points,delta=3)

    Calculate a molecular orbitals for a molecule and puts it in molecule.properties["MO_calculated"][MO_chosen]

    If no voxel were associated with the molecule, it will automatically create a voxel
    If the AO were not calculated it will also calculate them


    Parameters
    ----------
        MO_chosen : int
            the number of the molecular orbital starting at 0
        grid_points : list of 3
        delta : float, optional
            the length added on all directiosn to the box containing all atomic centers

    Returns
    -------
        MO_calculated_list : ndarray
    """
    if not "AO_calculated" in self.properties:
        calculate_AO(self)

    if not "MO_calculated_list" in self.properties:
        self.properties["MO_calculated_list"] = [[] for k in range(len(self.properties["MO_list"]))]

    AO_calculated = self.properties["AO_calculated"]
    N_AO = len(AO_calculated)
    MO_chosen_calculated = self.properties["MO_calculated_list"][MO_chosen]
    if type(MO_chosen_calculated) is not list:
        return MO_chosen_calculated

    voxel_matrix = self.properties["voxel_matrix"]
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]

    MO = np.array(self.properties["MO_list"][MO_chosen])
    MO_occupancy = self.properties["MO_occupancy"][0]


    AO_contribution_reshaped = np.array(MO).reshape((N_AO,1,1,1))

    MO_chosen_calculated = np.einsum("ijkl,ijkl->jkl",AO_calculated,AO_contribution_reshaped)

    self.properties["MO_calculated_list"][MO_chosen] = MO_chosen_calculated / (np.einsum("ijk,ijk->",MO_chosen_calculated,MO_chosen_calculated)*dV/MO_occupancy)**(1/2)

    return MO_chosen_calculated / (np.sum(MO_chosen_calculated**2*dV/MO_occupancy)**(1/2))



def calculate_electron_density(self):
    """
    calculate_electron_density()

    Calculates the electron density for a molecule and puts it in molecule.properties["electron_density"]

    If no voxel were associated with the molecule, it will automatically create a voxel
    If the MO were not calculated it will also calculate them

    Parameters
    ----------
        grid_points : list of 3
        delta : float, optional
            the length added on all directiosn to the box containing all atomic centers

    Returns
    -------
        electron_density : ndarray
    """

    if not "MO_calculated_list" in self.properties:
        self.calculate_MO()

    MO = np.array(self.properties["MO_calculated_list"])

    MO_occ = np.array(self.properties["MO_occupancy"])
    MO_occ_arr = MO_occ>0
    MO_occ_index = np.argmin(MO_occ)

    MO_occupied = MO[:,:MO_occ_index]

    electron_density = np.einsum("ijkl,ijkl->ikl",MO_occupied,MO_occupied)

    self.properties["electron_density"] = electron_density
    return electron_density


molecule.calculate_AO = calculate_AO
molecule.calculate_MO = calculate_MO
# molecule.calculate_occupied_MO = calculate_occupied_MO
molecule.calculate_MO_chosen = calculate_MO_chosen
molecule.calculate_electron_density = calculate_electron_density

####################
## EXCITED STATES ##
####################


def calculate_transition_density(self):
    """
    calculate_transition_density(grid_points,delta=3)

    Calculates all the transition densities for a molecule

    Parameters
    ----------
        grid_points : list of 3
        delta : float, optional
            the length added on all directions to the box containing all atomic centers

    Returns
    -------
        transition_density_list


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

    if not "transition_list" in self.properties:
        raise ValueError("No transitions were found, one should use rgmol.extract_excited_states.extract_transition to extract transition")

    transition_list = self.properties["transition_list"]
    transition_factor_list = self.properties["transition_factor_list"]

    if not "MO_calculated_list" in self.properties:
        self.calculate_MO()

    MO_calculated_list = self.properties["MO_calculated_list"]

    nprocs = rgmol.nprocs
    N_grids,N_MO,N_r,N_ang = np.shape(MO_calculated_list)

    print("##################################")
    print("# Calculating Transition Density #")
    print("##################################")

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



    if nprocs > 1 and 0:
    # if nprocs > 1:
        transition_density_list = calculate_transition_density_multithread(self,transitions,transition_density_coefficients,grid_points,nprocs)

    else:
        transition_density_list = np.zeros((N_grids,len(transition_list),N_r,N_ang))
        for occ in range(transi_occ_max):
            for virt in range(transi_virt_max):
                if transitions[occ,virt]:

                    MO_OCC = MO_calculated_list[:,occ]
                    MO_VIRT = MO_calculated_list[:,virt]
                    MO_product = MO_OCC * MO_VIRT

                    transition_coeffs = transition_density_coefficients[:,occ,virt]
                    for transition in range(num_transition):
                        coeff = transition_coeffs[transition]
                        if coeff!=0:
                            transition_density_list[:,transition] = transition_density_list[:,transition] + coeff * MO_product


    self.properties["transition_density_list"] = transition_density_list


    time_taken = time.time()-time_before_calc
    print("###########################################")
    print("# Finished calculating Transition Density #")
    print('# in {:3.3f} s #'.format(time_taken))
    print("###########################################")
    return transition_density_list


molecule.calculate_transition_density = calculate_transition_density

