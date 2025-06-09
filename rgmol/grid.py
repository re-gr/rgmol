#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script adds functions, and methods to the molecule objects.
These methods allow the calculation of various chemical properties such as AO, MO, Transition densities or linear response function.
"""

import numpy as np
import scipy as sp
from rgmol.objects import *
from rgmol.threading_functions import *
import scipy as sp
from rgmol.dicts import *


class mol_grids:
    """
    mol_grids(grids)

    Constructs a list of multiple atomic grids for a molecule

    Parameters
    ----------
        grids : list, optional
            List containing all the grids

    Returns
    -------
        mol_grids
            mol_grids object

    Attributes
    ----------
        grids : list
            The list of all the grids
        centers : list
            The list of the centers
    """
    def __init__(self,grids):
        """
        mol_grids(grids)

        Constructs a list of multiple atomic grids for a molecule

        Parameters
        ----------
            grids : list, optional
                List containing all the grids

        Returns
        -------
            mol_grids
                mol_grids object

        Attributes
        ----------
            grids : list
                The list of all the grids
            centers : list
                The list of the centers
            dV : array
                The dV of all grids
        """
        self.grids = grids
        grids_centers = []

        for grid_i in grids:
            grids_centers.append(grid_i.center)

        self.grids_centers = np.array(grids_centers)


    def integrate(self,arr):
        """
        integrate(arr)

        Integrates an array on all the atomic grid

        Parameters
        ----------
            arr : ndarray
                The array to be integrated

        Returns
        -------
            Int
                the integral
        """

        Int = 0
        for grid,grid_index in zip(self.grids,range(len(self.grids))):
            Int += grid.integrate(arr[grid_index])
        return Int


    def integrate_product(self,arr,arr_2):
        """
        integrate_product(arr,arr_2)

        Integrates the product of two arrays on all the atomic grid.
        This method should be used as it is around 3 times faster as using integrate with the product as an argument

        Parameters
        ----------
            arr : ndarray
                The array to be multiplied and integrated
            arr_2 : ndarray
                The array to be multiplied and integrated

        Returns
        -------
            Int
                the integral
        """

        Int = 0
        for grid,grid_index in zip(self.grids,range(len(self.grids))):
            Int += grid.integrate_product(arr[grid_index],arr_2[grid_index])
        return Int

    def get_dV(self):
        """
        get_dV

        Gets the integration elements for all grids in the format : N_grids,N_r,N_ang

        Parameters
        ----------
            None

        Returns
        -------
            dV
                The integration elements
        """
        N_r,N_ang = np.shape(self.grids[0].dV)
        dV = np.zeros((len(self.grids),N_r,N_ang))
        for grid_i,index_grid in zip(self.grids,range(len(self.grids))):
            dV[index_grid] = grid_i.dV * grid_i.wn

        return dV

class grid:
    """
    grid(coords,center,grid_type,dV,coords_gc=None)

    Constructs an atomic grid around a center.
    For contructing the grid properly around an atom, one should use create_atomic_grid

    Parameters
    ----------
        coords : ndarray
            Array containing the spherical coordinates of each point of the grid
        center : ndarray
            The center of the grid
        grid_type : str
            The type of grid, should be atomic as cubic are deprecated
        dV : array
            The dV of each point used for integration
        coords_gc : array, optional
            The coordinates of the gauss-chebychev integration.
            This represents the coordinates between -1 and 1
    Returns
    -------
        grid
            grid object

    Attributes
    ----------
        coords : ndarray
        center : ndarray
        grid_type : str
        dV : array
        coords_gc : array
        number_points : int
        xyz_coords : array
    """
    def __init__(self,coords,center,grid_type,dV,coords_gc=None):
        """
        grid(coords,center,grid_type,dV,coords_gc=None)

        Constructs an atomic grid around a center.
        For contructing the grid properly around an atom, one should use create_atomic_grid

        Parameters
        ----------
            coords : ndarray
                Array containing the spherical coordinates of each point of the grid
            center : ndarray
                The center of the grid
            grid_type : str
                The type of grid, should be atomic as cubic are deprecated
            dV : array
                The dV of each point used for integration
            coords_gc : array, optional
                The coordinates of the gauss-chebychev integration.
                This represents the coordinates between -1 and 1
        Returns
        -------
            grid
                grid object

        Attributes
        ----------
            coords : ndarray
            center : ndarray
            grid_type : str
            dV : array
            coords_gc : array
            number_points : int
            xyz_coords : array
        """

        self.grid_type = grid_type
        self.center = center
        self.coords = coords
        self.number_points = np.prod(np.shape(coords)[1:])
        self.coords_gc = coords_gc
        self.dV = dV

        if grid_type.lower() == "cubic":
            self.xyz_coords = coords

        elif grid_type.lower() == "atomic":
            r,Theta,Phi = coords
            cos_theta = np.cos(Theta)
            sin_theta = np.sin(Theta)
            cos_phi = np.cos(Phi)
            sin_phi = np.sin(Phi)

            x = r * sin_theta * cos_phi + self.center[0]
            y = r * sin_theta * sin_phi + self.center[1]
            z = r * cos_theta + self.center[2]

            self.xyz_coords = np.array([x,y,z])

        else:
            raise TypeError("This type of grid {} has not been implemented. Please use cubic or atomic.".format(grid_type))



    def integrate(self,arr):
        """
        integrate(arr)

        Integrates an array on the atomic grid

        Parameters
        ----------
            arr : ndarray
                The array to be integrated

        Returns
        -------
            Int
                the integral
        """

        Coords = self.coords
        wn = self.wn
        dV = self.dV

        dV = wn * dV

        Int = np.einsum("ij,ij->",arr,dV)

        return Int


    def integrate_product(self,arr,arr_2):
        """
        integrate_product(arr,arr_2)

        Integrates the product of two arrays on the atomic grid.
        This method should be used as it is around 3 times faster as using integrate with the product as an argument

        Parameters
        ----------
            arr : ndarray
                The array to be multiplied and integrated
            arr_2 : ndarray
                The array to be multiplied and integrated

        Returns
        -------
            Int
                the integral
        """

        Coords = self.coords
        wn = self.wn
        dV = self.dV

        dV = wn * dV

        Int = np.einsum("ij,ij,ij->",arr,arr_2,dV)

        return Int



def create_cubic_grid_from_molecule(mol,grid_points,delta=5):
    """
    create_cubic_grid_from_molecule(mol,grid_points,delta=5)

    Creates a cubic voxel from the position of atoms in a molecule

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

    mol.properties["grid_points"] = grid_points
    mol.properties["delta"] = delta

    list_pos = mol.list_property("pos")
    nx,ny,nz = grid_points
    xmin,ymin,zmin = np.min(list_pos,axis=0)
    xmax,ymax,zmax = np.max(list_pos,axis=0)

    voxel_origin = [xmin-delta,ymin-delta,zmin-delta]

    x_step = (xmax - xmin + 2*delta)/nx
    y_step = (ymax - ymin + 2*delta)/ny
    z_step = (zmax - zmin + 2*delta)/nz

    voxel_matrix = [[x_step,0,0],[0,y_step,0],[0,0,z_step]]
    r,c = create_coordinates_from_cubic_grid(grid_points,voxel_origin,voxel_matrix)

    return r,c

def create_coordinates_from_cubic_grid(grid_points,voxel_origin,voxel_matrix):
    """
    create_coordinates_from_cubic_grid(grid_points,voxel_origin,voxel_matrix)

    Creates coordinates from a cubic grid

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

    return r,[x,y,z]


def create_rectilinear_grid_from_molecule(mol,grid_points=(80,80,80),delta=7):
    """
    create_cubic_grid_from_molecule(mol,grid_points=(80,80,80),delta=7)

    Creates a cubic voxel from the position of atoms in a molecule

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

    mol.properties["grid_points"] = grid_points
    mol.properties["delta"] = delta

    list_pos = mol.list_property("pos")
    nx,ny,nz = grid_points
    xmin,ymin,zmin = np.min(list_pos,axis=0)
    xmax,ymax,zmax = np.max(list_pos,axis=0)

    N_atoms = len(mol.atoms)

    nx,ny,nz = grid_points
    zeta=1
    alpha=0.6

    list_x = []
    list_y = []
    list_z = []

    nx,ny,nz = nx//2,ny//2,nz//2
    #Gauss Chebyshev Type 2
    mx = np.cos((np.arange(nx)+1)/(nx+1) *np.pi)
    mx = mx - (mx==1)*0.0001 #Convert x=1 to x=0.999

    x = zeta / np.log(2) * np.log( (2)/ (1-mx) ) * (1+mx)**(alpha) #M4
    x_tot = np.append(-x,x[::-1])

    my = np.cos((np.arange(ny)+1)/(ny+1) *np.pi)
    my = my - (my==1)*0.0001 #Convert x=1 to x=0.999
    y = zeta / np.log(2) * np.log( (2)/ (1-my) ) * (1+my)**(alpha) #M4
    y_tot = np.append(-y,y[::-1])

    mz = np.cos((np.arange(nz)+1)/(nz+1) *np.pi)
    mz = mz - (mz==1)*0.0001 #Convert x=1 to x=0.999
    z = zeta / np.log(2) * np.log( (2)/ (1-mz) ) * (1+mz)**(alpha) #M4
    z_tot = np.append(-z,z[::-1])

    r = x_tot.reshape((1,nx*2,1,1))*np.array([1.,0,0]).reshape((3,1,1,1)) + \
        y_tot.reshape((1,1,ny*2,1))*np.array([0,1.,0]).reshape((3,1,1,1)) + \
        z_tot.reshape((1,1,1,nz*2))*np.array([0,0,1.]).reshape((3,1,1,1))

    # d = dx_tot.reshape((1,nx*2,1,1))*np.array([1.,0,0]).reshape((3,1,1,1)) + \
    #     dy_tot.reshape((1,1,ny*2,1))*np.array([0,1.,0]).reshape((3,1,1,1)) + \
    #     dz_tot.reshape((1,1,1,nz*2))*np.array([0,0,1.]).reshape((3,1,1,1))

    return r,[x_tot,y_tot,z_tot]




def create_grid_from_mol(mol,N_r=None,d_leb=None,zeta_list=None,alpha_list=None):
    """
    create_grid_from_mol(mol,N_r=None,d_leb=None,zeta_list=None,alpha_list=None)

    Creates atomic grids for all the atoms in the molecule.
    For more details on the atomic grids, see :doc:`create_atomic_grid<create_atomic_grid>`

    Parameters
    ----------
        mol : molecule
            The molecule for which the grids will be computed
        N_r : list, optional
            The number of radial points for each atomic grid
            If not provided, an already optimized value will be used
        d_leb : list, optional
            The order of Lebedev quadrature used for each atomic grid
            If not provided, an already optimized value will be used
        zeta_list : list, optional
            The zeta parameter, more info on :doc:`create_atomic_grid<create_atomic_grid>`
            If not provided, an already optimized value will be used
        alpha_list : list, optional
            The alpha parameter, more info on :doc:`create_atomic_grid<create_atomic_grid>`
            If not provided, an already optimized value will be used

    Returns
    -------
        None
            The molecule attribute mol_grids will contain the result
    """

    if mol.mol_grids:
        return

    list_grids = []

    N_r_list = []
    d_leb_list = []
    for atom in mol.atoms:
        if not N_r:
            N_r = dict_N_r[atom.name]
        if not d_leb:
            d_leb = dict_d_leb[atom.name]
        N_r_list.append(N_r)
        d_leb_list.append(d_leb)
    d_leb = np.max(d_leb_list)
    N_r_list = np.max(N_r_list)

    for atom,index_atom in zip(mol.atoms,range(len(mol.atoms))):
        zeta, alpha = None, None
        if alpha_list: alpha = alpha_list[index_atom]
        if zeta_list: zeta = zeta_list[index_atom]

        list_grids.append(create_atomic_grid(atom,N_r=N_r,d_leb=d_leb,zeta=zeta,alpha=alpha))



    m_grids = mol_grids(list_grids)
    voronoi_becke(m_grids)
    mol.mol_grids = m_grids





def create_atomic_grid(atom,N_r=None,d_leb=None,zeta=None,alpha=None):
    """
    create_atomic_grid(atom,N_r=None,d_leb=None,zeta=None,alpha=None)

    Creates an atomic grid
    It follows the Treutler and Ahlrichs M4 formula [1]_ for the radial part.
    And the Lebedev grid [2]_ for the angular part.
    If the parameters of the grid are not provided, already optimized ones will be used.


    Parameters
    ----------
        atom : atom
            the atom at the center of the grid
        N_r : int, optional
            the number of points for the radial part
        d_leb : int, optional
            the degree of the Lebedev quadrature
        zeta : float, optional
            the zeta parameter of the radial function M4
        alpha : float, optional
            the alpha parameter of the radial function M4

    Return
    ------
        grid

    Notes
    -----

        The formula M4 is the following :
        :math:`r = \\frac{\\zeta}{ln(2)}(1+x)^{\\alpha}ln\\left(\\frac{2}{1-x}`

    References
    ----------

        .. [1] Treutler, O.; Ahlrichs, R.The Journal of Chemical Physics1995,102,346–354.
        .. [2] Lebedev, V. I.; Laikov, D. N. InDoklady Mathematics, 1999; Vol. 59,pp 477–481.
    """

    if not zeta:
        zeta = dict_zeta[atom.name]
    if not alpha:
        alpha = dict_alpha[atom.name]

    coords_lebedev_xyz,weight_lebedev = sp.integrate.lebedev_rule(d_leb)
    X,Y,Z = coords_lebedev_xyz
    N_ang = len(X)

    X = X + (X==0)*1e-13 #remove dividing by 0
    R = 1 #Radius of the sphere

    #Convert to spheric coordinates
    Theta = np.arccos(Z/R)
    Phi = np.sign(Y) * np.arccos(X/(X**2+Y**2)**(1/2))
    Phi = np.arctan2(Y,X)


    #Gauss Chebyshev Type 2
    x = np.cos((np.arange(N_r)+1)/(N_r+1) *np.pi)
    x = x - (x==1)*0.0001 #Convert x=1 to x=0.999

    # r = R*(1+x)/(1-x) #Becke Transform
    # r = zeta / np.log(2) * np.log( (2)/ (1-x) ) #M3
    r = zeta / np.log(2) * np.log( (2)/ (1-x) ) * (1+x)**(alpha) #M4

    r = r.reshape((N_r,1))
    x = x.reshape((N_r,1))

    weight_lebedev = weight_lebedev.reshape((1,N_ang))

    coords = r.reshape((1,N_r,1))*np.array([1.,0,0]).reshape((3,1,1)) + \
             Theta.reshape((1,1,N_ang))*np.array([0,1.,0]).reshape((3,1,1)) + \
             Phi.reshape((1,1,N_ang))*np.array([0,0,1.]).reshape((3,1,1))

    # dr = 2/(1-x)**2 / (1-x**2)**(1/2) #Becke transform dr
    # dr = zeta/np.log(2) * (1/(1-x)) / (1-x**2)**(1/2) #M3 dr
    dr = zeta/np.log(2) * (1+x)**(alpha) * (alpha / (1+x) * np.log((2)/(1-x)) + 1/(1-x) ) / (1-x**2)**(1/2) #M4 dr
    w_gc = (np.pi/(N_r+1) * np.sin((np.arange(N_r)+1)/(N_r+1)*np.pi)**2).reshape((N_r,1)) #Weight Gauss Chebyshev Type 2

    dV = dr * w_gc * weight_lebedev * r * r
    grid_i = grid(coords,atom.pos,"atomic",coords_gc=x,dV=dV)

    return grid_i

# def create_grid_atom_lin(atom,N_r,N_ang,R=1):
#
#
#     Theta = (np.arange(N_ang)*np.pi/N_ang).reshape((1,1,N_ang))
#     Phi = (np.arange(N_ang)*2*np.pi/N_ang).reshape((1,N_ang,1))
#
#     r_0 = np.cos((np.arange(N_r)+1)/(N_r+1) *np.pi) #Gauss Chebyshev Type 2
#     r_0 = r_0 - (r_0==1)*0.001 #Convert x=1 to x=0.999
#     r = R*(1+r_0)/(1-r_0) #Becke Transform
#
#     r = r.reshape((N_r,1,1))
#
#     coords = r.reshape((1,N_r,1,1))*np.array([1.,0,0]).reshape((3,1,1,1)) + \
#              Theta.reshape((1,1,N_ang,1))*np.array([0,1.,0]).reshape((3,1,1,1)) + \
#              Phi.reshape((1,1,1,N_ang))*np.array([0,0,1.]).reshape((3,1,1,1))
#
#
#     grid_i = grid(coords,atom.pos,"atomic",R=R,coords_gc=r_0)
#
#     return grid_i



def voronoi_becke(mol_grids):
    """
    voronoi_becke(mol_grids)

    This functions uses the Voronoi-Becke scheme for the partition of space [1]_
    This allows to divide the whole space in multiple atomic grids

    Parameters
    ----------
        mol_grids : mol_grids
            All the grids

    Returns
    -------
        None
            The weights are automatically attributed to each grid inside mol_grids


    References
    ----------
        .. [1] Becke, A. D.The Journal of Chemical Physics1988,88, 2547–2553.
    """

    def function_cutoff(mu):
        return 3/2*mu - 1/2*mu**3

    num_grids = len(mol_grids.grids)
    grids_centers = mol_grids.grids_centers


    for grid_i,grid_i_index in zip(mol_grids.grids,range(num_grids)):
        #Dimensions : grid_i,grid_j,rad,ang,coordinates
        number_points = grid_i.number_points
        n_r,n_l = np.shape(grid_i.xyz_coords)[1:]
        xyz_coords = grid_i.xyz_coords.transpose((1,2,0)).reshape((1,1,n_r,n_l,3))

        grids_centers_i = grids_centers.reshape((num_grids,1,1,1,3))
        grids_centers_j = grids_centers.reshape((1,num_grids,1,1,3))
        R_ij = np.linalg.norm(grids_centers_i-grids_centers_j,axis=4)
        R_ij = R_ij + (R_ij==0)*1 #Remove diagonal part

        r_i = np.linalg.norm(xyz_coords - grids_centers_i,axis=4)
        r_j = np.linalg.norm(xyz_coords - grids_centers_j,axis=4)
        Mu_ij = (r_i - r_j) / R_ij

        s3 = 1/2 * (1-function_cutoff(function_cutoff(function_cutoff(Mu_ij))))
        # s3 = Mu_ij<0 #voronoi scheme

        Pi = np.prod(s3,axis=1)
        wn = Pi[grid_i_index] / np.sum(Pi,axis=0)
        grid_i.wn = wn




