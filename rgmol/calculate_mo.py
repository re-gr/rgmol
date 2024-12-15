#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import objects

def gaussian_s(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix):
    n,nx,ny,nz = np.shape(r)
    sum_gaussian = np.zeros((nx,ny,nz))
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0
    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian += coefs[0] * np.exp(-coefs[1]*r_r)

    sum_gaussian /= np.sum(sum_gaussian**2)/dV
    return sum_gaussian

def gaussian_p(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix):
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

    sum_gaussian_x /= np.sum(sum_gaussian_x**2)/dV
    sum_gaussian_y /= np.sum(sum_gaussian_y**2)/dV
    sum_gaussian_z /= np.sum(sum_gaussian_z**2)/dV

    return sum_gaussian_x,sum_gaussian_y,sum_gaussian_z


def gaussian_d(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix):
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
        sum_gaussian_xx_yy += (y_y-x_x)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zz += (2*z*z-x_x-y_y)*coefs[0] * np.exp(-coefs[1]*r_r)

    sum_gaussian_xy /= np.sum(sum_gaussian_xy**2)/dV
    sum_gaussian_xz /= np.sum(sum_gaussian_xz**2)/dV
    sum_gaussian_yz /= np.sum(sum_gaussian_yz**2)/dV
    sum_gaussian_xx_yy /= np.sum(sum_gaussian_xx_yy**2)/dV
    sum_gaussian_zz /= np.sum(sum_gaussian_zz**2)/dV

    o = np.zeros((nx,ny,nz))

    return sum_gaussian_zz, sum_gaussian_xz, sum_gaussian_yz, sum_gaussian_xx_yy, sum_gaussian_xy


def create_voxel_from_molecule(mol,grid_points,delta=5):
    """Create a voxel matrix from a molecule"""

    list_pos = mol.list_property("pos")
    nx,ny,nz = grid_points
    xmin,ymin,zmin = np.min(list_pos,axis=(0))
    xmax,ymax,zmax = np.max(list_pos,axis=(0))

    voxel_origin = [xmin-delta,ymin-delta,zmin-delta]

    x_step = (xmax - xmin + 2*delta)/nx
    y_step = (ymax - ymin + 2*delta)/ny
    z_step = (zmax - zmin + 2*delta)/nz

    voxel_matrix = [[x_step,0,0],[0,y_step,0],[0,0,z_step]]

    return voxel_origin,voxel_matrix


def create_coordinates_from_voxel(grid_points,voxel_origin,voxel_matrix):
    """Create coordinates from voxel"""

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
    """Calculate atomic orbitals for all atoms of a molecule"""
    voxel_origin,voxel_matrix = create_voxel_from_molecule(self,grid_points,delta=delta)

    r = create_coordinates_from_voxel(grid_points,voxel_origin,voxel_matrix)
    AO_list = self.properties["AO_list"]
    AO_type_list = self.properties["AO_type_list"]

    AO_calculated = []

    print("Calculating Atomic Orbitals")

    for AO_atom in zip(self.atoms,AO_list,AO_type_list):
        r0 = AO_atom[0].pos
        for AO in zip(AO_atom[1],AO_atom[2]):
            exponent_primitives,contraction_coefficients=np.array(AO[0]).transpose()
            if AO[1] == "s":
                AO_calculated.append(gaussian_s(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix))
            elif AO[1] == "p":
                AO_calc = gaussian_p(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)
                for ao in AO_calc:
                    AO_calculated.append(ao)

            elif AO[1] == "d":
                AO_calc = gaussian_d(r,contraction_coefficients,exponent_primitives,r0,voxel_matrix)
                for ao in AO_calc:
                    AO_calculated.append(ao)

            else: raise ValueError("This type of orbital has not been yet implemented")

    self.properties["voxel_origin"] = voxel_origin
    self.properties["voxel_matrix"] = voxel_matrix
    self.properties["voxel_coordinates"] = r
    self.properties["AO_calculated"] = np.array(AO_calculated)


    print("Finished Calculating Atomic Orbitals")



def calculate_MO(self,grid_points,delta=3):
    """Calculate the molecular orbitals of a molecule"""

    if not "AO_calculated" in self.properties:
        calculate_AO(self,grid_points=grid_points,delta=delta)

    AO_calculated = self.properties["AO_calculated"]
    N_AO = len(AO_calculated)
    MO_calculated = []
    MO_list = np.array(self.properties["MO_list"])

    for MO in MO_list:
        AO_contribution_reshaped = np.array(MO).reshape((N_AO,1,1,1))
        MO_calculated.append(np.sum(AO_calculated*AO_contribution_reshaped,axis=0))
    self.properties["MO_calculated"] = np.array(MO_calculated)



def calculate_MO_chosen(self,MO_chosen,grid_points,delta=3):
    """Calculate one MO"""
    if not "AO_calculated" in self.properties:
        calculate_AO(self,grid_points=grid_points,delta=delta)

    if not "MO_calculated" in self.properties:
        self.properties["MO_calculated"] = [[] for k in range(len(self.properties["MO_list"]))]


    AO_calculated = self.properties["AO_calculated"]
    N_AO = len(AO_calculated)
    MO_chosen_calculated = self.properties["MO_calculated"][MO_chosen]
    if MO_chosen_calculated!=[]:
        return MO_calculated

    MO = np.array(self.properties["MO_list"])[MO_chosen]
    AO_contribution_reshaped = np.array(MO).reshape((N_AO,1,1,1))

    MO_chosen_calculated = np.sum(AO_calculated*AO_contribution_reshaped,axis=0)

    self.properties["MO_calculated"][MO_chosen] = MO_chosen_calculated

    return MO_chosen_calculated


objects.molecule.calculate_AO = calculate_AO
objects.molecule.calculate_MO = calculate_MO
objects.molecule.calculate_MO_chosen = calculate_MO_chosen






