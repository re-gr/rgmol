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

def gaussian_s(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix):
    """
    gaussian_s(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)

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
        voxel_matrix : 2D ndarray
            the voxel matrix used to calculated the volume

    Returns
    -------
        s_orbital : ndarray same shape as r

    """


    n,nx,ny,nz = np.shape(r)
    sum_gaussian = np.zeros((nx,ny,nz))
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0
    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian +=  coefs[0] * np.exp(-coefs[1]*r_r)

    sum_gaussian /= (np.einsum("ijk,ijk->",sum_gaussian,sum_gaussian)*dV)**(1/2)

    return sum_gaussian,

def gaussian_p(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix):
    """
    gaussian_p(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)

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
        voxel_matrix : 2D ndarray
            the voxel matrix used to calculated the volume

    Returns
    -------
        p_x_orbital,
        p_y_orbital,
        p_z_orbital
    """


    n,nx,ny,nz = np.shape(r)
    sum_gaussian_x = np.zeros((nx,ny,nz))
    sum_gaussian_y = np.zeros((nx,ny,nz))
    sum_gaussian_z = np.zeros((nx,ny,nz))
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0

    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian_x += x*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_y += y*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_z += z*coefs[0] * np.exp(-coefs[1]*r_r)

    sum_gaussian_x /= (np.einsum("ijk,ijk->",sum_gaussian_x,sum_gaussian_x)*dV)**(1/2)
    sum_gaussian_y /= (np.einsum("ijk,ijk->",sum_gaussian_y,sum_gaussian_y)*dV)**(1/2)
    sum_gaussian_z /= (np.einsum("ijk,ijk->",sum_gaussian_z,sum_gaussian_z)*dV)**(1/2)

    return sum_gaussian_x,sum_gaussian_y,sum_gaussian_z


def gaussian_d(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix):
    """
    gaussian_d(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)

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
        voxel_matrix : 2D ndarray
            the voxel matrix used to calculated the volume

    Returns
    -------
        d_zz_orbital,
        d_xz_orbital,
        d_yz_orbital,
        d_xx_yy_orbital,
        d_xy_orbital
    """

    n,nx,ny,nz = np.shape(r)
    sum_gaussian_xy = np.zeros((nx,ny,nz))
    sum_gaussian_xz = np.zeros((nx,ny,nz))
    sum_gaussian_yz = np.zeros((nx,ny,nz))
    sum_gaussian_xx_yy = np.zeros((nx,ny,nz))
    sum_gaussian_zz = np.zeros((nx,ny,nz))
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]
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
        sum_gaussian_xx_yy += (x_x-y_y)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zz += (2*z*z-x_x-y_y)*coefs[0] * np.exp(-coefs[1]*r_r)

    sum_gaussian_xy /= (np.einsum("ijk,ijk->",sum_gaussian_xy,sum_gaussian_xy)*dV)**(1/2)
    sum_gaussian_xz /= (np.einsum("ijk,ijk->",sum_gaussian_xz,sum_gaussian_xz)*dV)**(1/2)
    sum_gaussian_yz /= (np.einsum("ijk,ijk->",sum_gaussian_yz,sum_gaussian_yz)*dV)**(1/2)
    sum_gaussian_xx_yy /= (np.einsum("ijk,ijk->",sum_gaussian_xx_yy,sum_gaussian_xx_yy)*dV)**(1/2)
    sum_gaussian_zz /= (np.einsum("ijk,ijk->",sum_gaussian_zz,sum_gaussian_zz)*dV)**(1/2)

    return sum_gaussian_zz, sum_gaussian_xz, sum_gaussian_yz, sum_gaussian_xx_yy, sum_gaussian_xy



def gaussian_f(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix):
    """
    gaussian_f(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)

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
        voxel_matrix : 2D ndarray
            the voxel matrix used to calculated the volume

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
    n,nx,ny,nz = np.shape(r)
    sum_gaussian_yyy_xxy = np.zeros((nx,ny,nz))
    sum_gaussian_xyz = np.zeros((nx,ny,nz))
    sum_gaussian_yzz_yrr = np.zeros((nx,ny,nz))
    sum_gaussian_zzz_zrr = np.zeros((nx,ny,nz))
    sum_gaussian_xzz_xrr = np.zeros((nx,ny,nz))
    sum_gaussian_zxx_zyy = np.zeros((nx,ny,nz))
    sum_gaussian_xxx_xyy = np.zeros((nx,ny,nz))
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0
    r0_reshaped = np.array(r0).reshape(3,1,1,1)

    x_x = x*x
    y_y = y*y
    z_z = z*z
    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian_yyy_xxy += (y_y-3*x_x)*y*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xyz += (x*y*z)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_yzz_yrr += (5*z_z-r_r)*y*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zzz_zrr += (5*z_z-3*r_r)*z*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xzz_xrr += (5*z_z-r_r)*x*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zxx_zyy += (x_x-y_y)*z*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xxx_xyy += (3*y_y-x_x)*x*coefs[0] * np.exp(-coefs[1]*r_r)

    sum_gaussian_yyy_xxy /= (np.einsum("ijk,ijk->",sum_gaussian_yyy_xxy,sum_gaussian_yyy_xxy)*dV)**(1/2)
    sum_gaussian_xyz /= (np.einsum("ijk,ijk->",sum_gaussian_xyz,sum_gaussian_xyz)*dV)**(1/2)
    sum_gaussian_yzz_yrr /= (np.einsum("ijk,ijk->",sum_gaussian_yzz_yrr,sum_gaussian_yzz_yrr)*dV)**(1/2)
    sum_gaussian_zzz_zrr /= (np.einsum("ijk,ijk->",sum_gaussian_zzz_zrr,sum_gaussian_zzz_zrr)*dV)**(1/2)
    sum_gaussian_xzz_xrr /= (np.einsum("ijk,ijk->",sum_gaussian_xzz_xrr,sum_gaussian_xzz_xrr)*dV)**(1/2)
    sum_gaussian_zxx_zyy /= (np.einsum("ijk,ijk->",sum_gaussian_zxx_zyy,sum_gaussian_zxx_zyy)*dV)**(1/2)
    sum_gaussian_xxx_xyy /= (np.einsum("ijk,ijk->",sum_gaussian_xxx_xyy,sum_gaussian_xxx_xyy)*dV)**(1/2)

    return sum_gaussian_zzz_zrr, sum_gaussian_xzz_xrr, sum_gaussian_yzz_yrr, sum_gaussian_zxx_zyy, sum_gaussian_xyz, sum_gaussian_xxx_xyy, sum_gaussian_yyy_xxy




def gaussian_g(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix):
    """
    gaussian_g(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)

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
        voxel_matrix : 2D ndarray
            the voxel matrix used to calculated the volume

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
    n,nx,ny,nz = np.shape(r)
    sum_gaussian_zzzz = np.zeros((nx,ny,nz))
    sum_gaussian_zzzy = np.zeros((nx,ny,nz))
    sum_gaussian_zzzx = np.zeros((nx,ny,nz))
    sum_gaussian_zzxy = np.zeros((nx,ny,nz))
    sum_gaussian_zz_xx_yy = np.zeros((nx,ny,nz))
    sum_gaussian_zyyy = np.zeros((nx,ny,nz))
    sum_gaussian_zxxx = np.zeros((nx,ny,nz))
    sum_gaussian_xy_xx_yy = np.zeros((nx,ny,nz))
    sum_gaussian_xxxx_yyyy = np.zeros((nx,ny,nz))
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0
    r0_reshaped = np.array(r0).reshape(3,1,1,1)

    x_x = x*x
    y_y = y*y
    z_z = z*z
    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian_zzzz += (35*z_z*z_z - 30*z_z*r_r + 3*r_r*r_r)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zzzy += y*z*(7*z_z-3*r_r)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zzzx += x*z*(7*z_z-3*r_r)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zzxy += 2*x*y*(7*z_z-r_r)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zz_xx_yy += (x_x-y_y)*(7*z_z-r_r)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zyyy += y*z*(y_y-3*x_x)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zxxx += x*z*(3*y_y-x_x)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xy_xx_yy += 4*x*y*(y_y-x_x)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xxxx_yyyy += (6*x_x*y_y-x_x*x_x-y_y*y_y)*coefs[0] * np.exp(-coefs[1]*r_r)


    sum_gaussian_zzzz /= (np.einsum("ijk,ijk->",sum_gaussian_zzzz,sum_gaussian_zzzz)*dV)**(1/2)
    sum_gaussian_zzzy /= (np.einsum("ijk,ijk->",sum_gaussian_zzzy,sum_gaussian_zzzy)*dV)**(1/2)
    sum_gaussian_zzzx /= (np.einsum("ijk,ijk->",sum_gaussian_zzzx,sum_gaussian_zzzx)*dV)**(1/2)
    sum_gaussian_zzxy /= (np.einsum("ijk,ijk->",sum_gaussian_zzxy,sum_gaussian_zzxy)*dV)**(1/2)
    sum_gaussian_zz_xx_yy /= (np.einsum("ijk,ijk->",sum_gaussian_zz_xx_yy,sum_gaussian_zz_xx_yy)*dV)**(1/2)
    sum_gaussian_zyyy /= (np.einsum("ijk,ijk->",sum_gaussian_zyyy,sum_gaussian_zyyy)*dV)**(1/2)
    sum_gaussian_zxxx /= (np.einsum("ijk,ijk->",sum_gaussian_zxxx,sum_gaussian_zxxx)*dV)**(1/2)
    sum_gaussian_xy_xx_yy /= (np.einsum("ijk,ijk->",sum_gaussian_xy_xx_yy,sum_gaussian_xy_xx_yy)*dV)**(1/2)
    sum_gaussian_xxxx_yyyy /= (np.einsum("ijk,ijk->",sum_gaussian_xxxx_yyyy,sum_gaussian_xxxx_yyyy)*dV)**(1/2)

    return sum_gaussian_zzzz, sum_gaussian_zzzx, sum_gaussian_zzzy, sum_gaussian_zz_xx_yy, sum_gaussian_zzxy, sum_gaussian_zxxx, sum_gaussian_zyyy, sum_gaussian_xxxx_yyyy, sum_gaussian_xy_xx_yy



def create_voxel_from_molecule(mol,grid_points,delta=5,delta_at=.4):
    """
    create_voxel_from_molecule(mol,grid_points,delta=5)

    Creates a voxel from the position of atoms in a molecule

    Parameters
    ----------
        mol : molecule_object
        grid_points : list of 3
            the number of points for each coordinates
        delta : float, optional
            the length added on all directiosn to the box containing all atomic centers

    Returns
    -------
        voxel_origin : list of 3
            the origin of the voxel
        voxel_matrix : 2d list
            the matrix of the voxel
    """

    print("##############################################")
    print("#    No voxel was found, creating a voxel    #")
    print("# with a delta = {:3.3f} around the molecule #".format(delta))
    print("#   on a grid of {} points   #".format(grid_points))
    print("#   and divides subspaces around each atoms  #")
    print("#       with the same number of points       #")
    print("#            and a delta = {:3.3f}           #".format(delta_at))
    print("##############################################")

    list_pos = mol.list_property("pos")
    nx,ny,nz = grid_points
    xmin,ymin,zmin = np.min(list_pos,axis=(0))
    xmax,ymax,zmax = np.max(list_pos,axis=(0))

    voxel_origin = [xmin-delta,ymin-delta,zmin-delta]

    x_step = (xmax - xmin + 2*delta)/nx
    y_step = (ymax - ymin + 2*delta)/ny
    z_step = (zmax - zmin + 2*delta)/nz

    voxel_matrix = [[x_step,0,0],[0,y_step,0],[0,0,z_step]]
    r = create_coordinates_from_voxel(grid_points,voxel_origin,voxel_matrix)

    subspaces_voxel_matrix = []
    subspaces_voxel_origin = []

    for atom in mol.atoms:
        pos = atom.pos
        pos_min = (pos - delta_at).reshape((3,1,1,1))
        pos_max = (pos + delta_at).reshape((3,1,1,1))

        pos_min_norm = np.linalg.norm(r-pos_min,axis=0)
        pos_max_norm = np.linalg.norm(r-pos_max,axis=0)

        min_index_x,min_index_y,min_index_z = np.unravel_index(np.argmin(pos_min_norm),(nx,ny,nz))
        max_index_x,max_index_y,max_index_z = np.unravel_index(np.argmin(pos_max_norm),(nx,ny,nz))

        subspaces_voxel_origin.append([min_index_x,min_index_y,min_index_z])
        delta_pos = r[:,max_index_x,max_index_y,max_index_z] - r[:,min_index_x,min_index_y,min_index_z]

        x_step_sub = delta_pos[0]*2/grid_points[0]
        y_step_sub = delta_pos[1]*2/grid_points[1]
        z_step_sub = delta_pos[2]*2/grid_points[2]
        subspaces_voxel_matrix.append([[x_step_sub,0,0],[0,y_step_sub,0],[0,0,z_step_sub]])


    return r,voxel_origin,voxel_matrix,subspaces_voxel_matrix,subspaces_voxel_origin


def create_coordinates_from_voxel(grid_points,voxel_origin,voxel_matrix):
    """
    create_coordinates_from_voxel(grid_points,voxel_origin,voxel_matrix)

    Creates coordinates from a voxel

    Parameters
    ----------
        grid_points : list of 3
        voxel_origin : list of 3
        voxel_matrix : 2d list

    Return
    ------
        r : ndarray corresponding to the coordinates
    """

    nx,ny,nz = grid_points

    voxel_end = [0,0,0]

    voxel_end[0] = voxel_origin[0] + voxel_matrix[0][0]*nx
    voxel_end[1] = voxel_origin[1] + voxel_matrix[1][1]*ny
    voxel_end[2] = voxel_origin[2] + voxel_matrix[2][2]*nz


    x = np.linspace(voxel_origin[0],voxel_end[0],nx,endpoint=False)
    y = np.linspace(voxel_origin[1],voxel_end[1],ny,endpoint=False)
    z = np.linspace(voxel_origin[2],voxel_end[2],nz,endpoint=False)

    r = x.reshape((1,nx,1,1))*np.array([1.,0,0]).reshape((3,1,1,1)) + \
        y.reshape((1,1,ny,1))*np.array([0,1.,0]).reshape((3,1,1,1)) + \
        z.reshape((1,1,1,nz))*np.array([0,0,1.]).reshape((3,1,1,1))

    return r




def calculate_AO(self,grid_points,delta=3):
    """
    calculate_AO(grid_points,delta=3)

    Calculate all atomic orbitals for a molecule and puts it in molecule.properties["AO_calculated"]
    If no voxel were associated with the molecule, it will automatically create a voxel

    Parameters
    ----------
        grid_points : list of 3
        delta : float, optional
            the length added on all directiosn to the box containing all atomic centers

    Returns
    -------
        AO_calculated : list of ndarray
    """

    if not "voxel_origin" in self.properties:
        coordinates,voxel_origin,voxel_matrix,subspaces_voxel_matrix,subspaces_voxel_origin = create_voxel_from_molecule(self,grid_points,delta=delta)
        self.properties["voxel_origin"] = voxel_origin
        self.properties["voxel_matrix"] = voxel_matrix
        self.properties["grid_points"] = grid_points
        self.properties["subspaces_voxel_matrix"] = subspaces_voxel_matrix
        self.properties["subspaces_voxel_origin"] = subspaces_voxel_origin
        self.properties["coordinates"] = coordinates

    voxel_origin = self.properties["voxel_origin"]
    voxel_matrix = self.properties["voxel_matrix"]
    r = self.properties["coordinates"]


    if not "AO_list" in self.properties:
        raise TypeError("The Atomic Orbital functions were not found.")

    AO_list = self.properties["AO_list"]
    AO_type_list = self.properties["AO_type_list"]

    AO_calculated = []
    print("###############################")
    print("# Calculating Atomic Orbitals #")
    print("###############################")
    time_before_calc = time.time()

    for AO_atom in zip(self.atoms,AO_list,AO_type_list):
        r0 = AO_atom[0].pos
        for AO in zip(AO_atom[1],AO_atom[2]):
            exponent_primitives,contraction_coefficients=np.array(AO[0]).transpose()
            if AO[1] == "s":
                AO_calc = gaussian_s(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)
            elif AO[1] == "p":
                AO_calc = gaussian_p(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)
            elif AO[1] == "d":
                AO_calc = gaussian_d(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)
            elif AO[1] == "f":
                AO_calc = gaussian_f(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)
            elif AO[1] == "g":
                AO_calc = gaussian_g(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)
            else: raise ValueError("This type of orbital has not been yet implemented")

            for ao in AO_calc:
                AO_calculated.append(ao)

    self.properties["voxel_coordinates"] = r
    self.properties["AO_calculated"] = np.array(AO_calculated)


    time_taken = time.time() - time_before_calc
    text_finished = "# Finished Calculating Atomic Orbitals #"
    text_time = "# in {:3.3f} s #".format(time_taken)
    numb_di = max(len(text_finished),len(text_time))
    print("#"*numb_di)
    print(text_finished)
    print(text_time)
    print("#"*numb_di)
    return np.array(AO_calculated)



def calculate_MO(self,grid_points,delta=3):
    """
    calculate_MO(grid_points,delta=3)

    Calculate all molecular orbitals for a molecule and puts it in molecule.properties["MO_calculated"]

    If no voxel were associated with the molecule, it will automatically create a voxel
    If the AO were not calculated it will also calculate them

    Parameters
    ----------
        grid_points : list of 3
        delta : float, optional
            the length added on all directiosn to the box containing all atomic centers

    Returns
    -------
        MO_calculated : list of ndarray
    """

    if not "AO_calculated" in self.properties:
        calculate_AO(self,grid_points=grid_points,delta=delta)

    AO_calculated = self.properties["AO_calculated"]
    N_AO = len(AO_calculated)
    MO_calculated = []
    MO_list = np.array(self.properties["MO_list"])
    MO_occupancy = self.properties["MO_occupancy"][0]


    voxel_matrix = self.properties["voxel_matrix"]
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]

    print('##################################')
    print("# Calculating Molecular Orbitals #")
    print('##################################')

    time_before_calc = time.time()


    for MO in MO_list:
        AO_contribution_reshaped = np.array(MO).reshape((N_AO,1,1,1))
        MO_not_normalized = np.einsum("ijkl,ijkl->jkl",AO_calculated,AO_contribution_reshaped)
        MO_calculated.append(MO_not_normalized / (np.einsum("ijk,ijk->",MO_not_normalized,MO_not_normalized)*dV/MO_occupancy)**(1/2))

    self.properties["MO_calculated"] = np.array(MO_calculated)

    time_taken = time.time() - time_before_calc

    print("###########################################")
    print("# Finished Calculating Molecular Orbitals #")
    print("# in {:3.3f} s #".format(time_taken))
    print("###########################################")

    return np.array(MO_calculated)



def calculate_occupied_MO(self,grid_points,delta=3):
    """
    calculate_MO(grid_points,delta=3)

    Calculate all molecular orbitals for a molecule and puts it in molecule.properties["MO_calculated"]

    If no voxel were associated with the molecule, it will automatically create a voxel
    If the AO were not calculated it will also calculate them

    Parameters
    ----------
        grid_points : list of 3
        delta : float, optional
            the length added on all directiosn to the box containing all atomic centers

    Returns
    -------
        MO_calculated : list of ndarray
    """

    if "MO_calculated" in self.properties:
        return self.properties["MO_calculated"]

    if not "AO_calculated" in self.properties:
        calculate_AO(self,grid_points=grid_points,delta=delta)

    AO_calculated = self.properties["AO_calculated"]
    N_AO = len(AO_calculated)
    MO_calculated = []
    MO_list = np.array(self.properties["MO_list"])

    voxel_matrix = self.properties["voxel_matrix"]
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]

    MO_occupancy = self.properties["MO_occupancy"]


    for MO,MO_occ in zip(MO_list,MO_occupancy):
        if MO_occ==0:
            break
        AO_contribution_reshaped = np.array(MO).reshape((N_AO,1,1,1))
        MO_not_normalized = np.einsum("ijkl,ijkl->jkl",AO_calculated,AO_contribution_reshaped)
        MO_calculated.append(MO_not_normalized / (np.einsum("ijk,ijk->",MO_not_normalized,MO_not_normalized)*dV/MO_occ)**(1/2))

    self.properties["MO_calculated"] = np.array(MO_calculated)
    return np.array(MO_calculated)


def calculate_MO_chosen(self,MO_chosen,grid_points=(80,80,80),delta=3):
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
        MO_calculated : ndarray
    """
    if not "AO_calculated" in self.properties:
        calculate_AO(self,grid_points=grid_points,delta=delta)

    if not "MO_calculated" in self.properties:
        self.properties["MO_calculated"] = [[] for k in range(len(self.properties["MO_list"]))]

    AO_calculated = self.properties["AO_calculated"]
    N_AO = len(AO_calculated)
    MO_chosen_calculated = self.properties["MO_calculated"][MO_chosen]
    if type(MO_chosen_calculated) is not list:
        return MO_chosen_calculated

    voxel_matrix = self.properties["voxel_matrix"]
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]

    MO = np.array(self.properties["MO_list"][MO_chosen])
    MO_occupancy = self.properties["MO_occupancy"][0]


    AO_contribution_reshaped = np.array(MO).reshape((N_AO,1,1,1))

    MO_chosen_calculated = np.einsum("ijkl,ijkl->jkl",AO_calculated,AO_contribution_reshaped)

    self.properties["MO_calculated"][MO_chosen] = MO_chosen_calculated / (np.einsum("ijk,ijk->",MO_chosen_calculated,MO_chosen_calculated)*dV/MO_occupancy)**(1/2)

    return MO_chosen_calculated / (np.sum(MO_chosen_calculated**2*dV/MO_occupancy)**(1/2))



def calculate_electron_density(self,grid_points,delta=5):
    """
    calculate_electron_density(grid_points,delta=5)

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

    if not "MO_calculated" in self.properties:
        self.calculate_occupied_MO(grid_points,delta=delta)

    MO = np.array(self.properties["MO_calculated"])

    MO_occ = np.array(self.properties["MO_occupancy"])
    MO_occ_arr = MO_occ>0
    MO_occ_index = np.argmin(MO_occ)

    MO_occ = MO_occ[MO_occ_arr]
    MO_occ = MO_occ.reshape(len(MO_occ),1,1,1)

    MO_occupied = MO[:MO_occ_index]

    electron_density = np.einsum("ijkl,ijkl->jkl",MO_occupied,MO_occupied)

    self.properties["electron_density"] = electron_density
    return electron_density


molecule.calculate_AO = calculate_AO
molecule.calculate_MO = calculate_MO
molecule.calculate_occupied_MO = calculate_occupied_MO
molecule.calculate_MO_chosen = calculate_MO_chosen
molecule.calculate_electron_density = calculate_electron_density

####################
## EXCITED STATES ##
####################


def calculate_transition_density(self,grid_points,delta=3):
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

    if not "MO_calculated" in self.properties:
        self.calculate_MO(grid_points=grid_points,delta=delta)

    nprocs = rgmol.nprocs
    nx,ny,nz = grid_points


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

    # print(num_transition,len(np.einsum("ijk,ijk->i",transition_density_coefficients,transition_density_coefficients)))
    # print(np.einsum("ijk,ijk->i",transition_density_coefficients,transition_density_coefficients))

    if nprocs > 1 and 0:
    # if nprocs > 1:
        transition_density_list = calculate_transition_density_multithread(self,transitions,transition_density_coefficients,grid_points,nprocs)
    else:
        transition_density_list = np.zeros((len(transition_list),nx,ny,nz))
        for occ in range(transi_occ_max):
            for virt in range(transi_virt_max):
                if transitions[occ,virt]:

                    MO_OCC = calculate_MO_chosen(self,occ,grid_points,delta=delta)
                    MO_VIRT = calculate_MO_chosen(self,virt,grid_points,delta=delta)
                    MO_product = MO_OCC * MO_VIRT

                    transition_coeffs = transition_density_coefficients[:,occ,virt]
                    for transition in range(num_transition):
                        coeff = transition_coeffs[transition]
                        if coeff!=0:
                            transition_density_list[transition] = transition_density_list[transition] + coeff * MO_product


    self.properties["transition_density_list"] = transition_density_list


    time_taken = time.time()-time_before_calc
    print("###########################################")
    print("# Finished calculating Transition Density #")
    print('# in {:3.3f} s #'.format(time_taken))
    print("###########################################")
    return transition_density_list


def calculate_chosen_transition_density(self,chosen_transition_density,grid_points,delta=3):
    """
    calculate_chosen_transition_density(chosen_transition_density,grid_points,delta=3)

    Calculates a chosen transition density for a molecule

    Parameters
    ----------
        chosen_transition_density : int
        grid_points : list of 3
        delta : float, optional
            the length added on all directions of the box containing all atomic centers

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

    if not "transition_list" in self.properties:
        raise ValueError("No transitions were found, one should use rgmol.extract_excited_states.extract_transition to extract transition")


    #Initialize transition density list
    if not "transition_density_list" in self.properties:
        self.properties["transition_density_list"] = [[] for k in range(len(self.properties["transition_list"]))]

    if type(self.properties["transition_density_list"][chosen_transition_density]) is not list:
        return self.properties["transition_density_list"][chosen_transition_density]


    transition_list = self.properties["transition_list"][chosen_transition_density]
    transition_factor_list = self.properties["transition_factor_list"][chosen_transition_density]


    transition_factor_list = np.array(transition_factor_list)
    transition_factor_list /= np.sum(transition_factor_list**2)**(1/2)

    nx,ny,nz = grid_points
    transition_density = np.zeros((nx,ny,nz))


    for transition_MO in zip(transition_list, transition_factor_list):

        MO_OCC = calculate_MO_chosen(self,transition_MO[0][0],grid_points,delta=delta)
        MO_VIRT = calculate_MO_chosen(self,transition_MO[0][1],grid_points,delta=delta)

        transition_density += transition_MO[1][0] * MO_OCC * MO_VIRT

    self.properties["transition_density_list"][chosen_transition_density] = transition_density
    return transition_density


molecule.calculate_transition_density = calculate_transition_density
molecule.calculate_chosen_transition_density = calculate_chosen_transition_density




