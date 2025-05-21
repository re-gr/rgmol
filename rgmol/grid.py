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

class list_grid:
    def __init__(self,grids):
        self.grids = grids



class grid:
    def __init__(self,r,grid_type):
        self.grid_type = grid_type
        self.center = center


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






def create_grid_atom(N_r,N_ang,R=1):
    Theta = (np.arange(N_ang)*np.pi/N_ang).reshape((1,1,N_ang))
    Phi = (np.arange(N_ang)*2*np.pi/N_ang).reshape((1,N_ang,1))

    cos_theta = np.cos(Theta)
    sin_theta = np.sin(Theta)
    cos_phi = np.cos(Phi)
    sin_phi = np.sin(Phi)

    r_0 = np.cos((np.arange(N_r)+1)/(N_r+1) *np.pi) #Gauss Chebyshev Type 2
    r_0 = r_0 - (r_0==1)*0.001 #Convert x=1 to x=0.999
    r = R*(1+r_0)/(1-r_0) #Becke Transform
    r = r.reshape((N_r,1,1))


    Coords = r.reshape((1,N_r,1,1))*np.array([1.,0,0]).reshape((3,1,1,1)) + \
             Theta.reshape((1,1,N_ang,1))*np.array([0,1.,0]).reshape((3,1,1,1)) + \
             Phi.reshape((1,1,1,N_ang))*np.array([0,0,1.]).reshape((3,1,1,1))

    return r_0,Coords




def voronoi_becke(mol,grids,N_rad,N_ang):
    """
    """
    Mu = []
    for grid in grids.grids:
        center = grid.center
        xyz_coords = grid.xyz_coords
        mu_i = np.zeros((len(xyz_coords)))
        for x,y,z in xyz_coords:
            pass



        mu_ij = (ri-rj) /rij



    s3 = 1/2 * (1-function_cutoff(function_cutoff(function_cutoff(mu_ij))))

    Pi = np.prod(s3,axis=1)
    wn = Pi / np.sum(Pi)



def integrate_on_atom_grid(mol,grid,arr,R):

    r_0,Coords = create_grid_atom(n_r,n_ang,R=R)

    r = Coords[0]
    w_gc = (np.pi/(n_r+1) * np.sin(np.arange(n_r)/(n_r+1)*np.pi)).reshape((n_r,1,1)) #Weight Gauss Chebyshev Type 2

    Theta = (np.arange(n_ang)*np.pi/n_ang).reshape((1,n_ang,1))

    r_0 = r_0.reshape((n_r,1,1))
    dV = 2/(1-r_0)**2 * R * w_gc * \#Radial Part
        (r)**2 *np.sin(Theta)*2*np.pi/n_ang/n_ang*np.pi #Angular part

    Int = np.einsum("ijk,ijk->",arr,dV)
    return Int