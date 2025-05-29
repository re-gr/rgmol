#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script adds functions, and methods to compute potential from a molecule.
"""

import os
import numpy as np
import scipy as sp
from rgmol.objects import *
# from rgmol.molecular_calculations import create_coordinates_from_voxel
# from rgmol.molecular_calculations import create_voxel_from_molecule

def set_external_potential(self,ext_mol,grid_points=(100,100,100),delta=5,recalculate=False):
    """

    """

    if not recalculate:
        #tries to find the cube file of the
        pass

    print("Computing the external potential. This may take a while")

    if not "voxel_origin" in self.properties:
        voxel_origin,voxel_matrix = create_voxel_from_molecule(self,grid_points,delta=delta)
        self.properties["voxel_origin"] = voxel_origin
        self.properties["voxel_matrix"] = voxel_matrix
        self.properties["grid_points"] = grid_points

    voxel_origin = self.properties["voxel_origin"]
    voxel_matrix = self.properties["voxel_matrix"]
    dV = voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]


    r = create_coordinates_from_voxel(grid_points,voxel_origin,voxel_matrix)

    total_external_potential = np.zeros(grid_points)

    for ext_atom in ext_mol.atoms:
        pos_pot = np.reshape(ext_atom.pos,(3,1,1,1))
        charge = ext_atom.atomic_number

        dist = np.sum((r-pos_pot)**2,axis=0)**(1/2)

        total_external_potential += charge / dist

    ext_electron_density = ext_mol.calculate_electron_density(grid_points=grid_points,delta=delta)

    ext_voxel_origin = ext_mol.properties["voxel_origin"]
    ext_voxel_matrix = ext_mol.properties["voxel_matrix"]

    rp = create_coordinates_from_voxel(grid_points,ext_voxel_origin,ext_voxel_matrix)

    for x in range(grid_points[0]):
        for y in range(grid_points[1]):
            for z in range(grid_points[2]):
                dist = np.sum((r[:,x,y,z].reshape((3,1,1,1)) - rp)**2,axis=0)**(1/2)
                total_external_potential[x,y,z] = total_external_potential[x,y,z] + np.sum(ext_electron_density/dist)

    print("Finished computing the external potential")


  #
  # 		for file in D[k*mult:(k+1)*mult]:
  # 			a=th.Thread(target=deconvo,args=(peaks,file[2:-4],chem+file,dim,perfect,name_sample+"_"+file[2:-4],A[1],height))
  # 			a.start()
  # 		a.join()
  #
  #






molecule.set_external_potential = set_external_potential