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

class list_grids:
    def __init__(self,grids):
        self.grids = grids
        grids_centers = []
        for grid_i in grids:
            grids_centers.append(grid_i.center)
        self.grids_centers = np.array(grids_centers)



class grid:
    def __init__(self,coords,center,grid_type,coords_gc=None,R=1):
        self.grid_type = grid_type
        self.center = center
        self.coords = coords
        self.number_points = np.prod(np.shape(coords)[1:])
        self.R = R
        self.coords_gc = coords_gc

        if grid_type.lower() == "cubic":
            self.xyz_coords = coords.reshape((3,self.number_points))

        elif grid_type.lower() == "atomic":
            r,Theta,Phi = coords
            cos_theta = np.cos(Theta)
            sin_theta = np.sin(Theta)
            cos_phi = np.cos(Phi)
            sin_phi = np.sin(Phi)

            x = r * sin_theta * cos_phi + self.center[0]
            y = r * sin_theta * sin_phi + self.center[1]
            z = r * cos_theta + self.center[2]

            self.xyz_coords = np.array([x,y,z]).reshape((3,self.number_points))


        else:
            raise TypeError("This type of grid {} has not been implemented. Please use cubic or atomic.".format(grid_type))




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
    xmin,ymin,zmin = np.min(list_pos,axis=0)
    xmax,ymax,zmax = np.max(list_pos,axis=0)
    print(zmin,zmax)

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


    x = np.linspace(voxel_origin[0],voxel_end[0],nx,endpoint=True)
    y = np.linspace(voxel_origin[1],voxel_end[1],ny,endpoint=True)
    z = np.linspace(voxel_origin[2],voxel_end[2],nz,endpoint=True)

    r = x.reshape((1,nx,1,1))*np.array([1.,0,0]).reshape((3,1,1,1)) + \
        y.reshape((1,1,ny,1))*np.array([0,1.,0]).reshape((3,1,1,1)) + \
        z.reshape((1,1,1,nz))*np.array([0,0,1.]).reshape((3,1,1,1))

    return r


def create_grid_atom(atom,N_r,N_ang,R=1):
    Theta = (np.arange(N_ang)*np.pi/N_ang).reshape((1,1,N_ang))
    Phi = (np.arange(N_ang)*2*np.pi/N_ang).reshape((1,N_ang,1))

    r_0 = np.cos((np.arange(N_r)+1)/(N_r+1) *np.pi) #Gauss Chebyshev Type 2
    r_0 = r_0 - (r_0==1)*0.001 #Convert x=1 to x=0.999
    r = R*(1+r_0)/(1-r_0) #Becke Transform
    r = r.reshape((N_r,1,1))


    coords = r.reshape((1,N_r,1,1))*np.array([1.,0,0]).reshape((3,1,1,1)) + \
             Theta.reshape((1,1,N_ang,1))*np.array([0,1.,0]).reshape((3,1,1,1)) + \
             Phi.reshape((1,1,1,N_ang))*np.array([0,0,1.]).reshape((3,1,1,1))


    grid_i = grid(coords,atom.pos,"atomic",R=R,coords_gc=r_0)

    return grid_i


def function_cutoff(mu):
    return 3/2*mu - 1/2*mu**3

def voronoi_becke(grids,N_rad,N_ang):
    """
    Becke, A. D.The Journal of Chemical Physics 1988,88, 2547–2553
    """
    num_grids = len(grids.grids)
    grids_centers = grids.grids_centers

    for grid_i,grid_i_index in zip(grids.grids,range(num_grids)):
        #Dimensions : grid_i,grid_j,points,coordinates

        number_points = grid_i.number_points

        xyz_coords = grid_i.xyz_coords.transpose().reshape((1,1,number_points,3))

        grids_centers_i = grids_centers.reshape((num_grids,1,1,3))
        grids_centers_j = grids_centers.reshape((1,num_grids,1,3))
        R_ij = np.linalg.norm(grids_centers_i-grids_centers_j,axis=3)
        R_ij = R_ij + (R_ij==0)*1e9 #Remove diagonal part

        r_i = np.linalg.norm(xyz_coords - grids_centers_i,axis=3)
        r_j = np.linalg.norm(xyz_coords - grids_centers_j,axis=3)

        Mu_ij = (r_i - r_j) / R_ij

        s3 = 1/2 * (1-function_cutoff(function_cutoff(function_cutoff(Mu_ij))))

        Pi = np.prod(s3,axis=1)
        wn = Pi[grid_i_index] / np.sum(Pi,axis=0)

        grid_i.wn = wn




def integrate_on_atom_grid(mol,grid,arr):
    coords_gc = grid.coords_gc
    Coords = grid.coords
    R = grid.R

    r = Coords[0]
    n_r,n_ang = np.shape(Coords)[1:3]
    wn = grid.wn
    w_gc = (np.pi/(n_r+1) * np.sin(np.arange(n_r)/(n_r+1)*np.pi)).reshape((n_r,1,1)) #Weight Gauss Chebyshev Type 2

    Theta = (np.arange(n_ang)*np.pi/n_ang).reshape((1,n_ang,1))

    coords_gc = coords_gc.reshape((n_r,1,1))

    dV = 2/(1-coords_gc)**2 * R * w_gc * \
        (r)**2 *np.sin(Theta)*2*np.pi/n_ang/n_ang*np.pi #Angular part

    Int = np.einsum("ijk,ijk->",arr,dV)
    return Int

