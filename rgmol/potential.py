    #   !/usr/bin/env python3
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
import rgmol.grid
import time

def set_external_potential(self,ext_mol,N_r=None,d_leb=None,N_r_ext=None,d_leb_ext=None,recalculate=False):
    """


    """

    if not recalculate:
        #tries to find the cube file of the
        pass

    print("Computing the external potential. This may take a while")

    if not self.mol_grids:
        rgmol.grid.create_grid_from_mol(self,N_r=N_r,d_leb=d_leb)
        mol_grids = self.mol_grids

    coords = mol_grids.coords
    N_grids, dim, N_r, N_ang = np.shape(coords)
    total_external_potential = np.zeros((N_grids,N_r,N_ang))


    for ext_atom in ext_mol.atoms:
        pos_pot = np.reshape(ext_atom.pos,(1,3,1,1))

        charge = ext_atom.atomic_number
        dist = np.linalg.norm(coords - pos_pot,axis=1)
        # dist = np.sum((coords-pos_pot)**2,axis=0)**(1/2)

        total_external_potential -= charge / dist

    rgmol.grid.create_grid_from_mol(ext_mol,N_r=N_r_ext,d_leb=d_leb_ext)
    ext_electron_density = ext_mol.calculate_electron_density()

    mol_grids_ext = ext_mol.mol_grids
    ext_coords = mol_grids_ext.coords
    N_grids_ext,dim_ext,N_r_ext,N_ang_ext = np.shape(ext_coords)


    N_t = 100
    x = np.cos((np.arange(N_t)+1)/(N_t+1) *np.pi)
    x = x - (x==1)*0.0001 #Convert x=1 to x=0.999
    t = (1+x)/(1-x) #Becke Transform
    dt = 2/(1-x)**2 / (1-x**2)**(1/2) #Becke transform dr
    w_gc = np.pi/(N_t+1) * np.sin((np.arange(N_t)+1)/(N_t+1)*np.pi)**2  #Weight Gauss Chebyshev Type 2

    dV_ext = mol_grids_ext.get_dV()


    R2 = ext_coords.reshape((1,N_grids_ext,dim_ext,N_r_ext,N_ang_ext))
    T2 = (t*t).reshape((N_t,1,1,1))

    def calc_on_grid(total_external_potential,grid):
        for r_ind in range(N_r):
            for ang_ind in range(N_ang):
                r1 = coords[grid,:,r_ind,ang_ind].reshape((1,1,3,1,1))
                # i:t, j:grids, k:r, l:ang
                total_external_potential[grid,r_ind,ang_ind] += 2/np.pi**(1/2) * np.einsum("ijkl,jkl,i,jkl->",np.exp(-T2*(np.linalg.norm(R2-r1,axis=2)**2)),ext_electron_density,w_gc*dt,dV_ext)
    import threading as th
    T = []
    for grid in range(N_grids):
        thread = th.Thread(target=calc_on_grid,args=(total_external_potential,grid))
        T.append(thread)
        thread.start()

    for thread in T:
        thread.join()


    # for grid in range(N_grids):
    #     for r_ind in range(N_r):
    #         for ang_ind in range(N_ang):
    #             r1 = coords[grid,:,r_ind,ang_ind].reshape((1,1,3,1,1))
    #             # i:t, j:grids, k:r, l:ang
    #             total_external_potential[grid,r_ind,ang_ind] -= 1/np.pi**(1/2) * np.einsum("ijkl,jkl,i,jkl->",np.exp(-T2*(np.linalg.norm(R2-r1,axis=2)**2)),ext_electron_density,w_gc*dt,dV_ext)

    np.save("pot",total_external_potential)
    self.properties["external_potential"] = total_external_potential
    print("Finished computing the external potential")
    return total_external_potential




def set_external_potential_from_file(self,file):
    """
    """

    self.properties["external_potential"] = np.load(file)






molecule.set_external_potential = set_external_potential
molecule.set_external_potential_from_file = set_external_potential_from_file


def comp_ext_pot_xyz(mol,ext_mol,grid_points=(80,80,80),delta=5):

    if not "Reconstructed_coords" in mol.properties:
        r,c = rgmol.grid.create_cubic_grid_from_molecule(mol,grid_points=grid_points,delta=delta)
        mol.properties["Reconstructed_coords"] = c
        mol.properties["Reconstructed_coords_xyz"] = r
    c = mol.properties["Reconstructed_coords"]
    r = mol.properties["Reconstructed_coords_xyz"]


    if not "Reconstructed_coords" in ext_mol.properties:
        r_ext,c_ext = rgmol.grid.create_cubic_grid_from_molecule(ext_mol,grid_points,delta=delta)
        ext_mol.properties["Reconstructed_coords"] = c_ext
        ext_mol.properties["Reconstructed_coords_xyz"] = r_ext
    c_ext = ext_mol.properties["Reconstructed_coords"]
    r_ext = ext_mol.properties["Reconstructed_coords_xyz"]


    nx,ny,nz = np.shape(r)[1:]
    nx_ext,ny_ext,nz_ext = np.shape(r_ext)[1:]
    x,y,z = c
    x_ext,y_ext,z_ext = c_ext
    dx = x_ext[1]-x_ext[0]
    dy = y_ext[1]-y_ext[0]
    dz = z_ext[1]-z_ext[0]

    x = x.reshape((1,1,nx))
    y = y.reshape((1,1,ny))
    z = z.reshape((1,1,nz))
    x_ext = x_ext.reshape((1,nx_ext,1))
    y_ext = y_ext.reshape((1,ny_ext,1))
    z_ext = z_ext.reshape((1,nz_ext,1))

    total_external_potential = np.zeros((nx,ny,nz))


    for ext_atom in ext_mol.atoms:
        pos_pot = np.reshape(ext_atom.pos,(3,1,1,1))

        charge = ext_atom.atomic_number
        dist = np.linalg.norm(r - pos_pot,axis=0)
        # dist = np.sum((coords-pos_pot)**2,axis=0)**(1/2)

        total_external_potential -= charge / dist



    ext_electron_density = rgmol.rectilinear_grid_reconstruction.reconstruct_electron_density(ext_mol)[1]
    print(np.sum(ext_electron_density))
    import tensorly as tl
    coefs,(d_x,d_y,d_z) = tl.decomposition.tucker(ext_electron_density,None)


    N_t = 100
    d = np.cos((np.arange(N_t)+1)/(N_t+1) *np.pi)
    d = d - (d==1)*0.0001 #Convert x=1 to x=0.999
    t = (1+d)/(1-d) #Becke Transform
    dt = 2/(1-d)**2 / (1-d**2)**(1/2) #Becke transform dr
    w_gc = np.pi/(N_t+1) * np.sin((np.arange(N_t)+1)/(N_t+1)*np.pi)**2  #Weight Gauss Chebyshev Type 2

    T2 = (t*t).reshape((N_t,1,1))
    gx = np.einsum("tjx,ja->txa",np.exp(-T2*((x_ext-x)**2)),d_x)*dx
    print("x done")
    gy = np.einsum("tky,kb->tyb",np.exp(-T2*((y_ext-y)**2)),d_y)*dy
    print("y done")
    gz = np.einsum("tlz,lc->tzc",np.exp(-T2*((z_ext-z)**2)),d_z)*dz
    print("z done")

    # np.einsum("tjx,tky,tlz,xa,yb,zc,abc,t->xyz",np.exp(-T2*((x_ext-x)**2)),np.exp(-T2*((y_ext-y)**2)),np.exp(-T2*((z_ext-z)**2)),d_x,d_y,d_z,coefs,w_gc*dt)


    txbc = np.einsum("txa,abc->txbc",gx,coefs)
    txyc = np.einsum("txbc,tyb->txyc",txbc,gy)
    txyz = np.einsum("txyc,tzc->txyz",txyc,gz)
    total_external_potential += 2/np.pi**(1/2) * np.einsum("txyz,t->xyz",txyz,w_gc*dt)

    # print(2/np.pi**(1/2) * np.einsum("txyz,t->xyz",txyz,w_gc*dt))
    np.save("pot_xyz",total_external_potential)
    return total_external_potential



    # print(np.shape(D),np.shape(E))
    # print("A")
    # A2 = np.einsum("jkl,xj,yk,zl->xyz",D,E[0],E[1],E[2])
    # print("B")
    # print(np.max(abs(A)-abs(A2)))
    # u2,s2,v2 = np.linalg.svd(v)
