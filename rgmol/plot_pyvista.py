#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script adds functions, and methods to the molecule objects.
These methods allow the plotting of chemical properties using pyvista.
"""

import numpy as np
import pyvista
from rgmol.objects import *
import rgmol.molecular_calculations


########################################
## Adding Plotting Methods for Atoms  ##
########################################



def plot(self,plotter,plotted_property="radius",opacity=1,factor=1):
    """
    plot(plotter,plotted_property="radius",opacity=1,factor=1)

    Plot a property of the atom on the plotter using pyvista

    Parameters :
    ------------
        plotter : pyvista.plotter
            The plotter object from pyvita on which the atom will be plotted. It can be easily defined using plotter = pyvista.Plotter()
        plotted_property : string, optional
            The property to be plotted. By default the radius is plotted.
        opacity : float, optional
            The opacity of the plot. By default equals to 1
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default equals to 1

    Returns :
    ---------
        None
            The atom is plotted on the plotter object
    """
    _plot_atom(plotter,self,plotted_property=plotted_property,opacity=opacity,factor=factor)



def plot_vector(self,plotter,vector,opacity=1,factor=1):
    """
    plot_vector(plotter,plotted_property="radius",opacity=1,factor=1)

    Plot a value of a vector on the position of the atom using pyvista

    Parameters :
    ------------
        plotter : pyvista.plotter
            The plotter object from pyvita on which the atom will be plotted. It can be easily defined using plotter = pyvista.Plotter()
        vector : float
            The value to be plotted
        opacity : float, optional
            The opacity of the plot. By default equals to 1
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default equals to 1

    Returns :
    ---------
        None
            The atom is plotted on the plotter object
    """
    _plot_vector_atom(plotter,self,vector,opacity=opacity,factor=factor)


atom.plot = plot
atom.plot_vector = plot_vector


############################################
## Adding Plotting Methods for Molecules  ##
############################################



def plot(self,plotter,plotted_property="radius",opacity=1,factor=1,show_bonds=True):
    """
    plot(plotter,plotted_property="radius",opacity=1,factor=1,show_bonds=True)

    Plot a property for the entire molecule

    Parameters
    ----------
        plotter : pyvista.plotter
            The plotter object from pyvita on which the atom will be plotted. It can be easily defined using plotter = pyvista.Plotter()
        plotted_property : float, optional
            The property to be plotted. By default the radius
        opacity : float, optional
            The opacity of the plot. By default equals to 1
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default equals to 1
        show_bonds : bool, optional
            Chose to show the bonds between the atoms or not. By default True

    Returns
    -------
        None
            The atom is plotted on the plotter object
    """
    for atom_x in self.atoms:
        atom_x.plot(plotter,plotted_property=plotted_property,opacity=opacity,factor=factor)
    if show_bonds:
        _bonds_plotting(plotter,self.bonds,self.list_property("pos"),self.list_property(plotted_property),factor=factor)
    return



def plot_vector(self,plotter,vector,opacity=1,factor=1):
    """
    plot_vector(plotter,vector,opacity=1,factor=1,show_bonds=True)

    Plot a vector on each atom for the entire molecule

    Parameters
    ----------
        plotter : pyvista.plotter
            The plotter object from pyvita on which the atom will be plotted. It can be easily defined using plotter = pyvista.Plotter()
        vector : list or ndarray
            The vector to be plotted. The length should be equal to the number of atoms.
        opacity : float, optional
            The opacity of the plot. By default equals to 1
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default equals to 1
        show_bonds : bool, optional
            Choose to show the bonds between the atoms or not. By default True

    Returns
    -------
        None
            The atom is plotted on the plotter object
    """
    for atom_x in range(len(self.atoms)):
        self.atoms[atom_x].plot_vector(plotter,vector[atom_x],opacity=opacity,factor=factor)
    return


def plot_radius(self,opacity=1,factor=.4,show_bonds=True):
    """
    plot_radius(opacity=1,factor=.4,show_bonds=True)

    Plot the radius of the entire molecule

    Parameters
    ----------
    opacity : float, optional
        The opacity of the plot. By default equals to 1
    factor : float, optional
        The factor by which the plotted_property will be multiplied. By default equals to 1
    show_bonds : bool, optional
        Choose to show the bonds between the atoms or not. By default True

    Returns
    -------
    None
        The plotter should display when using this function
    """
    plotter = pyvista.Plotter()
    for atom_x in self.atoms:
        atom_x.plot(plotter,opacity=opacity,factor=factor)
    if show_bonds:
        _bonds_plotting(plotter,self.bonds,self.list_property("pos"),self.list_property("radius"),factor=factor)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.show(full_screen=False)
    return


def plot_property(self,plotted_property,opacity=.4,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3):
    """
    plot_property(plotted_property,opacity=.4,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3)

    Plot a property for the entire molecule

    Parameters
    ----------
        plotted_property : float, optional
            The property to be plotted. By default the radius
        opacity : float, optional
            The opacity of the plot. By default equals to 1
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default equals to 1
        with_radius : bool, optional
            Chose to show the radius and the bonds between the atoms or not. By default True
        opacity_radius : float, optional
            The opacity of the radius plot. By default .8
        factor_radius : float, optional
            The factor by which the radius will be multiplied. By default .3

    Returns
    -------
        None
            The plotter should display when using this function
    """
    X = self.properties[plotted_property]
    plotter = pyvista.Plotter()

    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)
    self.plot_vector(plotter,X,opacity=opacity,factor=factor)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.show(full_screen=False)
    return


def plot_diagonalized_condensed_kernel(self,kernel,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3):
    """
    plot_diagonalized_condensed_kernel(kernel,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3)

    Diagonalize and plot a condensed kernel for a molecule. One can navigate through the eigenmodes using a slider.

    Parameters
    ----------
        kernel : str
            The kernel to be diagonalized and plotted
        opacity : float, optional
            The opacity of the plot. By default equals to 1
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default equals to 1
        with_radius : bool, optional
            Chose to show the radius and the bonds between the atoms or not. By default True
        opacity_radius : float, optional
            The opacity of the radius plot. By default .8
        factor_radius : float, optional
            The factor by which the radius will be multiplied. By default .3

    Returns
    -------
        None
            The plotter should display when using this function
    """
    X = self.properties[kernel]
    Xvp,XV = np.linalg.eigh(X)
    ncols = len(X)

    plotter = pyvista.Plotter()
    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)
    def create_mesh_diagonalized_kernel(value):
        vector_number = int(round(value))
        self.plot_vector(plotter,XV[:,vector_number-1],opacity=opacity,factor=factor)
        plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(Xvp[vector_number-1]),name="eigenvalue")


    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_diagonalized_kernel, [1, len(XV)],value=1,title="Eigenvector", fmt="%1.0f")
    plotter.show(full_screen=False)
    return


def plot_isodensity(self,plotted_isodensity="cube",cutoff=.2,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3):
    """
    plot_isodensity(plotted_isodensity="cube",cutoff=.2,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3)

    Plot an isodensity

    Parameters
    ----------
        plotted_isodensity : str, optional
            The isodensity to be plotted. By default "cube"
        cutoff : float, optional
            The cutoff of the isodensity plot. By default .2
        opacity : float, optional
            The opacity of the plot. By default equals to 1
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default equals to 1
        with_radius : bool, optional
            Chose to show the radius and the bonds between the atoms or not. By default True
        opacity_radius : float, optional
            The opacity of the radius plot. By default .8
        factor_radius : float, optional
            The factor by which the radius will be multiplied. By default .3

    Returns
    -------
        None
            The plotter should display when using this function
    """
    plotter = pyvista.Plotter()
    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius,show_bonds=True)
    _plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],self.properties["cube"],opacity=opacity,factor=factor,cutoff=cutoff)

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.show(full_screen=False)
    return




def plot_AO(self,grid_points=(40,40,40),delta=3,cutoff=.2,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3):
    """
    plot_AO(grid_points=(40,40,40),delta=3,cutoff=.2,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3)

    Plot the Atomic Orbitals of a molecule
    The Atomic Orbitals will be calculated on the grid that will be defined by the number of grid points and around the molecule. The delta defines the length to be added to the extremities of the position of the atoms.
    The order of the Atomic Orbitals is defined in the molden file

    Parameters
    ----------
        grid_points : list of 3, optional
            The number of points for the grid in each dimension. By default (40,40,40)
        delta : float, optional
            The length added on all directions of the box containing all atomic centers. By default 3
        cutoff : float, optional
            The cutoff of the isodensity plot. By default .2
        opacity : float, optional
            The opacity of the plot. By default .5
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default 1
        with_radius : bool, optional
            Chose to show the radius and the bonds between the atoms or not. By default True
        opacity_radius : float, optional
            The opacity of the radius plot. By default .8
        factor_radius : float, optional
            The factor by which the radius will be multiplied. By default .3

    Returns
    -------
        None
            The plotter should display when using this function
    """

    plotter = pyvista.Plotter()

    if not "AO_calculated" in self.properties:
        rgmol.molecular_calculations.calculate_AO(self,grid_points,delta=delta)

    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)
    def create_mesh_AO(value):
        _plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],self.properties["AO_calculated"][int(round(value))-1],opacity=opacity,factor=factor,cutoff=cutoff)
        AO_number = int(round(value))
        # plotter.add_text(text=r"AO = "+'{:3.3f} (a.u.)'.format(),name="ao number")

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_AO, [1, len(self.properties["AO_calculated"])],value=1,title="Number", fmt="%1.0f")
    plotter.show(full_screen=False)
    return





def plot_MO(self,grid_points=(40,40,40),cutoff=.2,delta=3,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3):
    """
    plot_MO(grid_points=(40,40,40),cutoff=.2,delta=3,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3)

    Plot the Molecular Orbitals of a molecule
    The Molecular Orbitals will be calculated on the grid that will be defined by the number of grid points and around the molecule. The delta defines the length to be added to the extremities of the position of the atoms.

    Parameters
    ----------
        grid_points : list of 3, optional
            The number of points for the grid in each dimension. By default (40,40,40)
        delta : float, optional
            The length added on all directions of the box containing all atomic centers. By default 3
        cutoff : float, optional
            The cutoff of the isodensity plot. By default .2
        opacity : float, optional
            The opacity of the plot. By default .5
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default 1
        with_radius : bool, optional
            Chose to show the radius and the bonds between the atoms or not. By default True
        opacity_radius : float, optional
            The opacity of the radius plot. By default .8
        factor_radius : float, optional
            The factor by which the radius will be multiplied. By default .3

    Returns
    -------
        None
            The plotter should display when using this function
    """

    plotter = pyvista.Plotter()

    if not "MO_calculated" in self.properties:
        self.properties["MO_calculated"] = [[] for k in range(len(self.properties["MO_list"]))]

    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)
    def create_mesh_MO(value):
        MO_number = int(round(value))
        MO_calculated = self.calculate_MO_chosen(MO_number-1,grid_points,delta=delta)

        _plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],MO_calculated,opacity=opacity,factor=factor,cutoff=cutoff)

        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["MO_energy"][MO_number-1]),name="mo energy")


    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_MO, [1, len(self.properties["MO_calculated"])],value=1,title="Number", fmt="%1.0f")
    plotter.show(full_screen=False)
    return



def plot_transition_density(self,grid_points=(40,40,40),delta=3,cutoff=.2,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3):
    """
    plot_transition_density(grid_points=(40,40,40),delta=3,cutoff=.2,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3)


    Plot the Transition Densities of a molecule.
    All the AO and the MO will be calculated on the grid if they were not calculated.
    The grid is defined by the number of grid points and around the molecule. The delta defines the length to be added to the extremities of the position of the atoms.

    Parameters
    ----------
        grid_points : list of 3, optional
            The number of points for the grid in each dimension. By default (40,40,40)
        delta : float, optional
            The length added on all directions of the box containing all atomic centers. By default 3
        cutoff : float, optional
            The cutoff of the isodensity plot. By default .2
        opacity : float, optional
            The opacity of the plot. By default .5
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default 1
        with_radius : bool, optional
            Chose to show the radius and the bonds between the atoms or not. By default True
        opacity_radius : float, optional
            The opacity of the radius plot. By default .8
        factor_radius : float, optional
            The factor by which the radius will be multiplied. By default .3

    Returns
    -------
        None
            The plotter should display when using this function

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

    plotter = pyvista.Plotter()

    #Initialize MO list
    if not "MO_calculated" in self.properties:
        self.properties["MO_calculated"] = [[] for k in range(len(self.properties["MO_list"]))]

    #Initialize transition density list
    if not "transition_density_list" in self.properties:
        self.properties["transition_density_list"] = [[] for k in range(len(self.properties["transition_list"]))]

    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)


    def create_mesh_transition_density(value):
        transition_number = int(round(value))
        transition_density_calculated = self.calculate_chosen_transition_density(transition_number-1,grid_points,delta=delta)

        _plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],transition_density_calculated,opacity=opacity,factor=factor,cutoff=cutoff)

        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["transition_energy"][transition_number-1]),name="transition energy")


    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_transition_density, [1, len(self.properties["transition_density_list"])],value=1,title="Number", fmt="%1.0f")
    plotter.show(full_screen=False)





def plot_diagonalized_kernel(self,kernel,method="partial",number_eigenvectors=20,grid_points=(20,20,20),delta=3,cutoff=.2 ,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3):
    """
    plot_diagonalized_kernel(kernel,method="partial",number_eigenvectors=20,grid_points=(20,20,20),delta=3,cutoff=.2 ,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3)

    Calculate and diagonalize a kernel. Only the linear response function is implemented for now.
    Only the first number_eigenvectors will be computed to limit the calculation time.
    The grid is defined by the number of grid points and around the molecule. The delta defines the length to be added to the extremities of the position of the atoms.

    Parameters
    ----------
        kernel : str
            The kernel to be diagonalized and plotted
        method : str, optional
            The method of calculation of the kernel, partial by default. More information can be found in the notes.
        number_eigenvectors : int, optional
            The number of eigenvectors to be computed
        grid_points : list of 3, optional
            The number of points for the grid in each dimension. By default (40,40,40)
        delta : float, optional
            The length added on all directions of the box containing all atomic centers. By default 3
        cutoff : float, optional
            The cutoff of the isodensity plot. By default .2
        opacity : float, optional
            The opacity of the plot. By default .5
        factor : float, optional
            The factor by which the plotted_property will be multiplied. By default 1
        with_radius : bool, optional
            Chose to show the radius and the bonds between the atoms or not. By default True
        opacity_radius : float, optional
            The opacity of the radius plot. By default .8
        factor_radius : float, optional
            The factor by which the radius will be multiplied. By default .3

    Returns
    -------
        None
            The plotter should display when using this function


    Notes
    -----
        Because the kernels are 6-dimensional, they scale up drastically in terms of memory used. That is why a partial method has been implemented which allows to remove the part of the space where the transition densities are almost zero. For each transition density, the space is sorted and the lower dense part that makes up to less than 1% is removed. In practice this removes as much as 90% of the space. More details on this method can be found :doc:`here<../orbitals/calculate_linear_response>`.

    """

    if kernel != "linear_response_function":
        raise ValueError("Only linear response function implemented for now")

    self.diagonalize_kernel(kernel,number_eigenvectors,grid_points,method=method,delta=delta)

    eigenvectors = self.properties["linear_response_eigenvectors"]
    eigenvalues = self.properties["linear_response_eigenvalues"]

    plotter = pyvista.Plotter()
    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)

    def create_mesh_diagonalized_kernel(value):
        vector_number = int(round(value))
        _plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],eigenvectors[vector_number-1],opacity=opacity,factor=factor,cutoff=cutoff)
        plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(eigenvalues[vector_number-1]),name="eigenvalue")


    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_diagonalized_kernel, [1, len(eigenvectors)],value=1,title="Eigenvector", fmt="%1.0f")
    plotter.show(full_screen=False)




molecule.plot = plot
molecule.plot_vector = plot_vector
molecule.plot_radius = plot_radius
molecule.plot_property = plot_property
molecule.plot_diagonalized_condensed_kernel = plot_diagonalized_condensed_kernel
molecule.plot_isodensity = plot_isodensity
molecule.plot_AO = plot_AO
molecule.plot_MO = plot_MO
molecule.plot_transition_density = plot_transition_density
molecule.plot_diagonalized_kernel = plot_diagonalized_kernel


##############################
## Representation functions ##
##############################


def _Rz(alpha):
    """3D Rotation matrix around the z axis"""
    return np.array([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
def _Ry(beta):
    """3D Rotation matrix around the y axis"""
    return np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
def _Rx(gamma):
    """3D Rotation matrix around the x axis"""
    return np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])

def _corr_angle(angle,x,y):
    """Corrects the angle given by arctan to be the proper angle."""
    if x>0 and y<0:
        return -angle
    if x<0 and y>0:
        return np.pi-angle
    if x<0 and y<0:
        return np.pi+angle
    return angle



def _rota_bonds(Vec,x,y,z):
    """
    Roates the bonds
    """
    #Gets the two angles alpha and beta of the vector
    Vec=Vec/np.linalg.norm(Vec)
    if Vec[0]==0:
        alpha = -np.pi/2
    else:
        alpha=np.arctan(np.abs(Vec[1]/Vec[0]))
    alpha=_corr_angle(alpha,Vec[0],Vec[1])
    Vec2=_Rz(-alpha).dot(Vec)
    if Vec2[2] == 0:
        beta = -np.pi/2
    else:
        beta=np.arctan(abs(Vec2[0]/Vec2[2]))
    beta=_corr_angle(beta,Vec2[2],Vec2[0])
    Rota=_Rz(alpha).dot(_Ry(beta))#The rotation matrix to convert to a vector of the x axis to a colinear vector of Vec the starting vector
    #Rotates the bonds
    x2,y2,z2=[],[],[]
    for k in range(len(x)):
        xt,yt,zt=[],[],[]
        for j in range(len(x[0])):
            pos3=Rota.dot(np.array([x[k][j],y[k][j],z[k][j]]))
            xt.append(pos3[0])
            yt.append(pos3[1])
            zt.append(pos3[2])
        x2.append(xt)
        y2.append(yt)
        z2.append(zt)
    return np.array(x2),np.array(y2),np.array(z2)


def _find_angle(Vec1,Vec2):
    """
    Finds the angle between two vectors
    """
    cross=np.cross(Vec1,Vec2) #Calculate the sign of the angle
    cross= cross*(abs(cross)>1e-5)#Remove almost zero components
    scross=np.sign(np.prod(cross[np.where(cross!=0)]))
    cosangle=(Vec1.dot(Vec2) / np.linalg.norm(Vec1)/np.linalg.norm(Vec2))#cos angle
    if cosangle>1: cosangle=1 #If the two vectors are colinear, in some cases the scalar product gives 1 + 1e-15
    if cosangle<-1: cosangle=-1
    return ( np.arccos(cosangle))*scross

def _rota_mol(Pos,Vec1,Vec2):
    """
    Roates the molecule
    """

    #Find the angles of the vector we want the other vector to be colinear to
    if Vec1[0]==0:
        alpha1=np.pi/2
    else: alpha1=np.arctan(np.abs(Vec1[1]/Vec1[0]))
    alpha1=_corr_angle(alpha1,Vec1[0],Vec1[1])
    Vec12=_Rz(-alpha1).dot(Vec1)
    if Vec12[2]==0:
        beta1=np.pi/2
    else: beta1=np.arctan(abs(Vec12[0]/Vec12[2]))
    beta1=_corr_angle(beta1,Vec12[2],Vec12[0])
    Vec13=_Ry(-beta1).dot(Vec12)

    if Vec2[0]==0:
        alpha2=np.pi/2
    else: alpha2=np.arctan(np.abs(Vec2[1]/Vec2[0]))
    alpha2=_corr_angle(alpha2,Vec2[0],Vec2[1])
    Vec22=_Rz(-alpha2).dot(Vec2)
    if Vec22[2]==0:
        beta2=np.pi/2
    else: beta2=np.arctan(abs(Vec22[0]/Vec22[2]))
    beta2=_corr_angle(beta2,Vec22[2],Vec22[0])
    Vec23=_Ry(-beta2).dot(Vec22)

    #V2= _Rz._Ry.v2 => v2= _RyT._RzT.V2
    #=> v = _Rz1._Ry1._RyT._RzT.V2
    Rota=_Rz(alpha1).dot(_Ry(beta1).dot(_Ry(-beta2).dot(_Rz(-alpha2))))
    Rota_ax=_Rz(alpha1).dot(_Ry(beta1))
    x2,y2,z2=[],[],[]#Rotates the bonds
    # for k in range(len(Pos)):
    #     Pos[k]=Rota.dot(Pos[k])
    Pos=Rota.dot(Pos.transpose()).transpose()
    return Pos,Rota_ax



def _orthonormal_basis(Pos,B,k):
    """
    Creates an orthonormal basis centered on an atom with the "z" axis perpendicular to the surface created by the 3 atoms and the "y" axis perpendicular to the bond but on the same surface created by the 3 atoms

    Inputs :    Pos (ndarray)
                B (list)
                k (int)

    Output :
                z (ndarray dim(3)) z axis
                y (ndarray dim(3)) y axis
    """


    ind=int(B[k][1])-1
    ind2=int(B[k][0])-1


    Vec=(Pos[ind2]-Pos[ind])/np.linalg.norm((Pos[ind]-Pos[ind2]))

    #In order to create an orthonormal basis, the strategy is : taking two vectors linked to two adjacent atoms that are not colinear, one can claculate the cross product wihch gives a perpendicular vector. And by taking the cross product with the last vector and one of the two first, the result is an orthonormal basis that is centered on the atom of interest
    for j in B:

        if int(j[0])-1==ind and int(j[1])-1!=ind2:

            num=int(j[1])-1
            Dist=(Pos[ind]-Pos[num])/np.linalg.norm((Pos[ind]-Pos[num]))
            angl=_find_angle(Dist,Vec)

            if angl<0: Dist=-Dist
            if abs(Dist.dot(Vec))<0.95:
                per=np.cross(Vec,Dist)
                return per,np.cross(Vec,per)

        if int(j[1])-1==ind and int(j[0])-1!=ind2:

            num=int(j[0])-1
            Dist=(Pos[ind]-Pos[num])/np.linalg.norm((Pos[ind]-Pos[num]))
            angl=_find_angle(Dist,Vec)

            if angl<0: Dist=-Dist
            if abs(Dist.dot(Vec))<0.95:
                per=np.cross(Vec,Dist)
                return per,np.cross(Vec,per)


        if int(j[0])-1==ind2 and int(j[1])-1!=ind:

            num=int(j[1])-1
            Dist=(Pos[ind2]-Pos[num])/np.linalg.norm((Pos[ind2]-Pos[num]))
            angl=_find_angle(Dist,Vec)

            if angl<0: Dist=-Dist
            if abs(Dist.dot(Vec))<0.95:
                per=np.cross(Vec,Dist)
                return per,np.cross(Vec,per)

        if int(j[1])-1==ind2 and int(j[0])-1!=ind:

            num=int(j[0])-1
            Dist=(Pos[ind2]-Pos[num])/np.linalg.norm((Pos[ind2]-Pos[num]))
            angl=_find_angle(Dist,Vec)

            if angl<0: Dist=-Dist
            if abs(Dist.dot(Vec))<0.95:
                per=np.cross(Vec,Dist)
                return per,np.cross(Vec,per)

    #Linear => we take a random vector that is not colinear
    if Vec[1]!=0 or Vec[2]!=0:
        aVec=np.copy(Vec+np.array([1,0,0]))
    else: aVec=np.copy(Vec+np.array([0,0,1]))
    aVec/=np.linalg.norm(aVec)
    per=np.cross(Vec,aVec)
    return per,np.array([1,-1,-1.3])#linear



def _bonds_plotting(plotter,bonds,Pos,Vec,factor=1):
    """
    Plot the bonds
    """
    #initial values for the parameters of the bonds
    Radbond=0.05
    u=np.linspace(0,2*np.pi,30)#base of the cylinder
    v=np.linspace(0,np.pi,2) #height of the cylinder

    for k in range(len(bonds)):
        one,two=int(bonds[k][0])-1,int(bonds[k][1])-1
        order = bonds[k][2]
        Vect=Pos[one]-Pos[two]

        dist=np.linalg.norm(Pos[one]-Pos[two])

        x=Radbond*(np.outer(np.cos(u),np.ones(np.size(v))))
        y=Radbond*(np.outer(np.sin(u),np.ones(np.size(v))))
        z=(np.outer(np.ones(np.size(u)),np.linspace((abs(Vec[two]*factor)-1/20),(dist-abs(Vec[one]*factor)+1/20),np.size(v))))
        x,y,z=_rota_bonds(Vect,x,y,z)
        x,y,z=x+Pos[two][0],y+Pos[two][1],z+Pos[two][2]

        if order==1:
            grid = pyvista.StructuredGrid(x,y,z)
            plotter.add_mesh(grid,name="bond_{}_{}".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)


        elif order==1.5:
            zpe,pe=_orthonormal_basis(Pos,bonds,k)#Get a orthonormal vector in order to distance the two cylinders
            pe=pe/np.linalg.norm(pe)/15
            grid = pyvista.StructuredGrid(x-pe[0],y-pe[1],z-pe[2])
            grid2 = pyvista.StructuredGrid(x+pe[0],y+pe[1],z+pe[2])
            plotter.add_mesh(grid,name="bond_{}_{}_1".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
            plotter.add_mesh(grid2,name="bond_{}_{}_2".format(one,two),color="white",pbr=True,roughness=.2,metallic=.7,opacity=.5)



        elif order==2:
            zpe,pe=_orthonormal_basis(Pos,bonds,k)
            pe=pe/np.linalg.norm(pe)/15
            grid = pyvista.StructuredGrid(x-pe[0],y-pe[1],z-pe[2])
            grid2 = pyvista.StructuredGrid(x+pe[0],y+pe[1],z+pe[2])
            plotter.add_mesh(grid,name="bond_{}_{}_1".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
            plotter.add_mesh(grid2,name="bond_{}_{}_2".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
        else:
            zpe,pe=_orthonormal_basis(Pos,bonds,k)
            pe=pe/np.linalg.norm(pe)/12
            grid = pyvista.StructuredGrid(x,y,z)
            grid2 = pyvista.StructuredGrid(x-pe[0],y-pe[1],z-pe[2])
            grid3 = pyvista.StructuredGrid(x+pe[0],y+pe[1],z+pe[2])
            plotter.add_mesh(grid,name="bond_{}_{}_1".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
            plotter.add_mesh(grid2,name="bond_{}_{}_2".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
            plotter.add_mesh(grid3,name="bond_{}_{}_3".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
    return





def _plot_atom(plotter,atom,plotted_property="radius",opacity=1,factor=1):
    """plot atom as a sphere"""
    Norm = atom.properties[plotted_property]
    atom_sphere = pyvista.Sphere(radius=Norm*factor, phi_resolution=100, theta_resolution=100,center=atom.pos)
    plotter.add_mesh(atom_sphere,name=atom.nickname,color=atom.color,pbr=False,roughness=0.0,metallic=0.0,diffuse=1,opacity=opacity)
    return


def _plot_vector_atom(plotter,atom,vector,opacity=1,factor=1):
    """plot atom as a sphere"""

    colors=[[255,0,0],[255,255,255]]
    atom_sphere = pyvista.Sphere(radius=abs(vector)*factor, phi_resolution=100, theta_resolution=100,center=atom.pos)
    plotter.add_mesh(atom_sphere,name=atom.nickname+"vector",color=colors[(vector>0)*1],pbr=True,roughness=.4,metallic=.4,diffuse=1,opacity=opacity)

    return


def _plot_cube(plotter,voxel_origin,voxel_matrix,cube,cutoff=0.1,opacity=1,factor=1):
    """plot atom as a sphere"""

    nx,ny,nz = np.shape(cube)
    cube_transposed = np.transpose(cube,(2,1,0))

    grid = pyvista.ImageData(dimensions=(nx,ny,nz),spacing=(voxel_matrix[0][0], voxel_matrix[1][1], voxel_matrix[2][2]),origin=voxel_origin)

    #Calculate cube density
    cube_density = cube_transposed**2 * voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]
    #Calculate renormalization as for some reason some cube files are not normalized
    cube_density = cube_density / np.sum(cube_density)

    array_sort = np.argsort(cube_density,axis=None)[::-1]
    cube_sorted = cube_density.flatten()[array_sort]
    cube_values_sorted = np.cumsum(cube_sorted)

    #Find how to unsort the array. There should be a more efficient way to do this
    indexes = np.arange(len(array_sort),dtype=int)
    array_unsort = np.zeros(len(array_sort),dtype=int)

    for k in range(len(array_sort)):
        array_unsort[array_sort[k]] = indexes[k]
    cube_values = cube_values_sorted[array_unsort]

    cube_values_positive = cube_values + (cube_transposed<0).flatten() * (1-cube_values)
    cube_values_negative = cube_values + (cube_transposed>0).flatten() * (1-cube_values)

    contour_positive = grid.contour(isosurfaces=2,scalars=cube_values_positive,rng=[0,1-cutoff])
    contour_negative = grid.contour(isosurfaces=2,scalars=cube_values_negative,rng=[0,1-cutoff])

    if len(contour_positive.point_data["Contour Data"]):
        plotter.add_mesh(contour_positive,name="isosurface_cube_positive",opacity=opacity,pbr=True,roughness=.5,metallic=.2,color="red")
    else:
        plotter.remove_actor("isosurface_cube_positive")
    if len(contour_negative.point_data["Contour Data"]):
        plotter.add_mesh(contour_negative,name="isosurface_cube_negative",opacity=opacity,pbr=True,roughness=.5,metallic=.2,color="blue")
    else:
        plotter.remove_actor("isosurface_cube_negative")

