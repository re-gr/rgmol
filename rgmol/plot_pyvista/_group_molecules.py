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
from ._functions import *

###########################
## function for subplots ##
###########################


def _subplotting(number_molecules):
    """
    finds the proper number of subplots for group of molecules plotting
    """

    if number_molecules < 4:
        return (1,number_molecules)
    elif number_molecules < 7:
        return (2,int(round(number_molecules/2)))
    elif number_molecules < 10:
        print("Lots of molecules")
        return (3,int(round(number_molecules/3)))
    else:
        print("Too much molecules, the graph will still be plotted, but it will not be good")
        return (3,int(round(number_molecules)))

def get_subplot(number_molecules,index):
    """
    Gets the 2d subplot index from the index and the number of molecules
    """

    if number_molecules < 4:
        return (0,index)
    elif number_molecules < 7:
        frac = int(round(number_molecules/2))
        return (index//frac,index%frac)
    else:
        frac = int(round(number_molecules/3))
        return (index//frac,index%frac)


class _subplot:
    def __init__(self,indexes,plotter,func,molecule):
        self.indexes = indexes
        self.plotter = plotter
        self.func = func
        self.molecule = molecule

    def __call__(self, value):
        self.func(value,self.indexes,self.molecule)



#####################################################
## Adding Plotting Methods for Groud of Molecules  ##
#####################################################




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
    number_molecules = len(self.molecules)
    shape = _subplotting(number_molecules)
    plotter = pyvista.Plotter(shape=shape,border=True)
    for index_molecule in range(number_molecules):
        index = get_subplot(number_molecules,index_molecule)
        plotter.subplot(index[0],index[1])

        molecule = self.molecules[index_molecule]
        for atom_x in molecule.atoms:
            atom_x.plot(plotter,opacity=opacity,factor=factor)
        if show_bonds:
            bonds_plotting(plotter,molecule.bonds,molecule.list_property("pos"),molecule.list_property("radius"),factor=factor)

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

    number_molecules = len(self.molecules)
    shape = _subplotting(number_molecules)
    plotter = pyvista.Plotter(shape=shape,border=True)
    for index_molecule in range(number_molecules):
        index = get_subplot(number_molecules,index_molecule)
        plotter.subplot(index[0],index[1])

        molecule = self.molecules[index_molecule]

        X = molecule.properties[plotted_property]

        if with_radius:
            molecule.plot(plotter,factor=factor_radius,opacity=opacity_radius)
        molecule.plot_vector(plotter,X,opacity=opacity,factor=factor)
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


    number_molecules = len(self.molecules)
    shape = _subplotting(number_molecules)
    plotter = pyvista.Plotter(shape=shape,border=True)

    def create_mesh_diagonalized_kernel(value,indexes,molecule):
        plotter.subplot(indexes[0],indexes[1])
        vector_number = int(round(value))
        molecule.plot_vector(plotter,XV[:,vector_number-1],opacity=opacity,factor=factor)
        plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(Xvp[vector_number-1]),name="eigenvalue")

    list_subpl = [_subplot(get_subplot(number_molecules,index_molecule),plotter,create_mesh_diagonalized_kernel,self.molecules[index_molecule]) for index_molecule in range(number_molecules)]

    for index_molecule in range(number_molecules):
        index = get_subplot(number_molecules,index_molecule)
        plotter.subplot(index[0],index[1])

        molecule = self.molecules[index_molecule]

        X = molecule.properties[kernel]
        Xvp,XV = np.linalg.eigh(X)

        if with_radius:
            molecule.plot(plotter,factor=factor_radius,opacity=opacity_radius)

        light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
        plotter.add_light(light)
        plotter.add_slider_widget(list_subpl[index_molecule], [1, len(XV)],value=1,title="Eigenvector", fmt="%1.0f")
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

    number_molecules = len(self.molecules)
    shape = _subplotting(number_molecules)
    plotter = pyvista.Plotter(shape=shape,border=True)
    for index_molecule in range(number_molecules):
        index = get_subplot(number_molecules,index_molecule)
        plotter.subplot(index[0],index[1])

        molecule = self.molecules[index_molecule]
        if with_radius:
            molecule.plot(plotter,factor=factor_radius,opacity=opacity_radius,show_bonds=True)
        plot_cube(plotter,molecule.properties["voxel_origin"],molecule.properties["voxel_matrix"],molecule.properties["cube"],opacity=opacity,factor=factor,cutoff=cutoff)

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

    number_molecules = len(self.molecules)
    shape = _subplotting(number_molecules)
    plotter = pyvista.Plotter(shape=shape,border=True)

    def create_mesh_AO(value,indexes,molecule):
        plotter.subplot(indexes[0],indexes[1])
        AO_number = int(round(value))
        plot_cube(plotter,molecule.properties["voxel_origin"],molecule.properties["voxel_matrix"],molecule.properties["AO_calculated"][int(round(value))-1],opacity=opacity,factor=factor,cutoff=cutoff,add_name=str(shape[1]*indexes[0]+indexes[1]))


    list_subpl = [_subplot(get_subplot(number_molecules,index_molecule),plotter,create_mesh_AO,self.molecules[index_molecule]) for index_molecule in range(number_molecules)]

    for index_molecule in range(number_molecules):
        index = get_subplot(number_molecules,index_molecule)
        plotter.subplot(index[0],index[1])

        molecule = self.molecules[index_molecule]

        if not "AO_calculated" in molecule.properties:
            rgmol.molecular_calculations.calculate_AO(molecule,grid_points,delta=delta)

        if with_radius:
            molecule.plot(plotter,factor=factor_radius,opacity=opacity_radius)

        light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
        plotter.add_light(light)
        plotter.add_slider_widget(list_subpl[index_molecule], [1, len(molecule.properties["AO_calculated"])],value=1,title="Number", fmt="%1.0f")
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

    number_molecules = len(self.molecules)
    shape = _subplotting(number_molecules)
    plotter = pyvista.Plotter(shape=shape,border=True)

    def create_mesh_MO(value,indexes,molecule):
        plotter.subplot(indexes[0],indexes[1])
        MO_number = int(round(value))
        MO_calculated = molecule.calculate_MO_chosen(MO_number-1,grid_points,delta=delta)

        plot_cube(plotter,molecule.properties["voxel_origin"],molecule.properties["voxel_matrix"],MO_calculated,opacity=opacity,factor=factor,cutoff=cutoff,add_name=str(shape[1]*indexes[0]+indexes[1]))
        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(molecule.properties["MO_energy"][MO_number-1]),name="mo energy")
        print_occupancy(plotter,molecule.properties["MO_occupancy"],MO_number,divy=shape[1])


    list_subpl = [_subplot(get_subplot(number_molecules,index_molecule),plotter,create_mesh_MO,self.molecules[index_molecule]) for index_molecule in range(number_molecules)]

    for index_molecule in range(number_molecules):
        index = get_subplot(number_molecules,index_molecule)
        plotter.subplot(index[0],index[1])

        molecule = self.molecules[index_molecule]

        if not "MO_calculated" in molecule.properties:
            molecule.properties["MO_calculated"] = [[] for k in range(len(molecule.properties["MO_list"]))]

        if with_radius:
            molecule.plot(plotter,factor=factor_radius,opacity=opacity_radius)


        light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
        plotter.add_light(light)
        plotter.add_slider_widget(list_subpl[index_molecule], [1, len(molecule.properties["MO_calculated"])],value=1,title="Number", fmt="%1.0f")
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


    number_molecules = len(self.molecules)
    shape = _subplotting(number_molecules)
    plotter = pyvista.Plotter(shape=shape,border=True)

    def create_mesh_transition_density(value,indexes,molecule):
        plotter.subplot(indexes[0],indexes[1])
        transition_number = int(round(value))
        transition_density_calculated = molecule.calculate_chosen_transition_density(transition_number-1,grid_points,delta=delta)

        plot_cube(plotter,molecule.properties["voxel_origin"],molecule.properties["voxel_matrix"],transition_density_calculated,opacity=opacity,factor=factor,cutoff=cutoff,add_name=str(shape[1]*indexes[0]+indexes[1]))

        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(molecule.properties["transition_energy"][transition_number-1]),name="transition energy")

    list_subpl = [_subplot(get_subplot(number_molecules,index_molecule),plotter,create_mesh_transition_density,self.molecules[index_molecule]) for index_molecule in range(number_molecules)]

    for index_molecule in range(number_molecules):
        index = get_subplot(number_molecules,index_molecule)
        plotter.subplot(index[0],index[1])

        molecule = self.molecules[index_molecule]
        #Initialize MO list
        if not "MO_calculated" in molecule.properties:
            molecule.properties["MO_calculated"] = [[] for k in range(len(molecule.properties["MO_list"]))]

        #Initialize transition density list
        if not "transition_density_list" in molecule.properties:
            molecule.properties["transition_density_list"] = [[] for k in range(len(molecule.properties["transition_list"]))]

        if with_radius:
            molecule.plot(plotter,factor=factor_radius,opacity=opacity_radius)

        light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
        plotter.add_light(light)
        plotter.add_slider_widget(list_subpl[index_molecule], [1, len(molecule.properties["transition_density_list"])],value=1,title="Number", fmt="%1.0f")


    plotter.show(full_screen=False)





def plot_diagonalized_kernel(self,kernel,method="only eigenmodes",plotting_method="isodensity",number_eigenvectors=20,grid_points=(20,20,20),delta=3,cutoff=.2,number_isodensities=10 ,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3):
    """
    plot_diagonalized_kernel(kernel,method="only eigenmodes",plotting_method="isodensity",number_eigenvectors=20,grid_points=(20,20,20),delta=3,cutoff=.2,number_isodensities=10 ,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3)

    Calculate and diagonalize a kernel. Only the linear response function is implemented for now.
    Only the first number_eigenvectors will be computed to limit the calculation time.
    The grid is defined by the number of grid points and around the molecule. The delta defines the length to be added to the extremities of the position of the atoms.

    Parameters
    ----------
        kernel : str
            The kernel to be diagonalized and plotted
        method : str, optional
            The method of calculation of the kernel, only eigenmodes by default. More information can be found in the notes.
        plotting_method : str, optional
            The method used for plotting. By default isodensity. The other methods are : "multiple isodensities", "volume"
        number_eigenvectors : int, optional
            The number of eigenvectors to be computed
        grid_points : list of 3, optional
            The number of points for the grid in each dimension. By default (40,40,40)
        delta : float, optional
            The length added on all directions of the box containing all atomic centers. By default 3
        cutoff : float, optional
            The cutoff of the isodensity plot for the isodensity plotting method. By default .2
        number_isodensities : int, optional
            The number of isodensities to be plotted if the method used is multiple isodensities
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
        Because the kernels are 6-dimensional, they scale up drastically in terms of memory used.
        If one only wants to look at the eigenmodes, the "only eigenmodes" method is just that. It computes the eigenmodes without computing the total kernel. More info can be found :doc:`here<../orbitals/calculate_eigenmodes_linear_response_function>`.
        Otherwise, the partial method has been implemented which allows to remove the part of the space where the transition densities are almost zero. For each transition density, the space is sorted and the lower dense part that makes up to less than 1% is removed. In practice this removes as much as 90% of the space. More details on this method can be found :doc:`here<../orbitals/diagonalize_kernel>`.

    """

    number_molecules = len(self.molecules)
    shape = _subplotting(number_molecules)
    plotter = pyvista.Plotter(shape=shape,border=True)


    if plotting_method == "isodensity":

        def create_mesh_diagonalized_kernel(value,indexes,molecule):
            plotter.subplot(indexes[0],indexes[1])
            vector_number = int(round(value))
            eigenvectors = molecule.properties["linear_response_eigenvectors"]
            eigenvalues = molecule.properties["linear_response_eigenvalues"]
            plot_cube(plotter,molecule.properties["voxel_origin"],molecule.properties["voxel_matrix"],eigenvectors[vector_number-1],opacity=opacity,factor=factor,cutoff=cutoff,add_name=str(shape[1]*indexes[0]+indexes[1]))
            plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(eigenvalues[vector_number-1]),name="eigenvalue")

            if method == "only eigenmodes":
                print_contribution_transition_density(plotter,vector_number,contrib_eigenvectors,divy=shape[1])

    elif plotting_method == "multiple isodensities":
        def create_mesh_diagonalized_kernel(value,indexes,molecule):
            plotter.subplot(indexes[0],indexes[1])
            vector_number = int(round(value))
            eigenvectors = molecule.properties["linear_response_eigenvectors"]
            eigenvalues = molecule.properties["linear_response_eigenvalues"]
            plot_cube_multiple_isodensities(plotter,molecule.properties["voxel_origin"],molecule.properties["voxel_matrix"],eigenvectors[vector_number-1],factor=factor,add_name=str(shape[1]*indexes[0]+indexes[1]))
            plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(eigenvalues[vector_number-1]),name="eigenvalue")

            if method == "only eigenmodes":
                print_contribution_transition_density(plotter,vector_number,contrib_eigenvectors,divy=shape[1])

    elif plotting_method == "volume":
        def create_mesh_diagonalized_kernel(value,indexes,molecule):
            plotter.subplot(indexes[0],indexes[1])
            vector_number = int(round(value))
            eigenvectors = molecule.properties["linear_response_eigenvectors"]
            eigenvalues = molecule.properties["linear_response_eigenvalues"]
            plot_cube_volume(plotter,molecule.properties["voxel_origin"],molecule.properties["voxel_matrix"],eigenvectors[vector_number-1],factor=factor,add_name=str(shape[1]*indexes[0]+indexes[1]))
            plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(eigenvalues[vector_number-1]),name="eigenvalue")

            if method == "only eigenmodes":
                print_contribution_transition_density(plotter,vector_number,contrib_eigenvectors,divy=shape[1])

    list_subpl = [_subplot(get_subplot(number_molecules,index_molecule),plotter,create_mesh_diagonalized_kernel,self.molecules[index_molecule]) for index_molecule in range(number_molecules)]


    for index_molecule in range(number_molecules):
        index = get_subplot(number_molecules,index_molecule)
        plotter.subplot(index[0],index[1])

        molecule = self.molecules[index_molecule]
        if kernel != "linear_response_function":
            raise ValueError("Only linear response function implemented for now")

        if method == "only eigenmodes":
            molecule.calculate_eigenmodes_linear_response_function(grid_points,delta=delta)
            contrib_eigenvectors = molecule.properties["contibution_linear_response_eigenvectors"]
        else:
            molecule.diagonalize_kernel(kernel,number_eigenvectors,grid_points,method=method,delta=delta)

        eigenvalues = molecule.properties["linear_response_eigenvalues"]

        if with_radius:
            molecule.plot(plotter,factor=factor_radius,opacity=opacity_radius)

        light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
        plotter.add_light(light)
        plotter.add_slider_widget(list_subpl[index_molecule], [1, len(eigenvalues)],value=1,title="Eigenvector", fmt="%1.0f")
    plotter.show(full_screen=False)




group_molecules.plot_radius = plot_radius
group_molecules.plot_property = plot_property
group_molecules.plot_diagonalized_condensed_kernel = plot_diagonalized_condensed_kernel
group_molecules.plot_isodensity = plot_isodensity
group_molecules.plot_AO = plot_AO
group_molecules.plot_MO = plot_MO
group_molecules.plot_transition_density = plot_transition_density
group_molecules.plot_diagonalized_kernel = plot_diagonalized_kernel

