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

###############################
## Defines class for sliders ##
###############################

class _slider:
    def __init__(self,func,value,cutoff):
        self.func = func
        self.value = value
        self.cutoff = cutoff

    def __call__(self, slider, value):
        if slider == "cutoff":
            self.cutoff = value
        else:
            self.value = value
        self.func(self.value,self.cutoff)

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
    plot_atom(plotter,self,plotted_property=plotted_property,opacity=opacity,factor=factor)



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
    plot_vector_atom(plotter,self,vector,opacity=opacity,factor=factor)


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
        bonds_plotting(plotter,self.bonds,self.list_property("pos"),self.list_property(plotted_property),factor=factor)
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


def plot_radius(self,opacity=1,factor=.4,show_bonds=True,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_radius(opacity=1,factor=.4,show_bonds=True,screenshot_button=True,window_size_screenshot=(1000,1000))

    Plot the radius of the entire molecule

    Parameters
    ----------
    opacity : float, optional
        The opacity of the plot. By default equals to 1
    factor : float, optional
        The factor by which the plotted_property will be multiplied. By default equals to 1
    show_bonds : bool, optional
        Choose to show the bonds between the atoms or not. By default True
    screenshot_button : bool, optional
        Adds a screenshot button. True by default
    window_size_screenshot : tuple, optional
        The size of the screenshots. By default (1000,1000)

    Returns
    -------
    None
        The plotter should display when using this function
    """
    plotter = pyvista.Plotter()
    for atom_x in self.atoms:
        atom_x.plot(plotter,opacity=opacity,factor=factor)
    if show_bonds:
        bonds_plotting(plotter,self.bonds,self.list_property("pos"),self.list_property("radius"),factor=factor)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)
    plotter.show(full_screen=False)
    return


def plot_property(self,plotted_property,opacity=.4,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_property(plotted_property,opacity=.4,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,screenshot_button=True,window_size_screenshot=(1000,1000))

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
        screenshot_button : bool, optional
            Adds a screenshot button. True by default
        window_size_screenshot : tuple, optional
            The size of the screenshots. By default (1000,1000)

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
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)
    plotter.show(full_screen=False)
    return


def plot_diagonalized_condensed_kernel(self,kernel,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_diagonalized_condensed_kernel(kernel,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,screenshot_button=True,window_size_screenshot=(1000,1000))

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
        screenshot_button : bool, optional
            Adds a screenshot button. True by default
        window_size_screenshot : tuple, optional
            The size of the screenshots. By default (1000,1000)

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
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)
    plotter.show(full_screen=False)
    return


def plot_isodensity(self,plotted_isodensity="cube",opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_isodensity(plotted_isodensity="cube",opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))

    Plot an isodensity

    Parameters
    ----------
        plotted_isodensity : str, optional
            The isodensity to be plotted. By default "cube"
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
        cutoff : float, optional
            The initial cutoff of the isodensity plot. By default .2
        screenshot_button : bool, optional
            Adds a screenshot button. True by default
        window_size_screenshot : tuple, optional
            The size of the screenshots. By default (1000,1000)

    Returns
    -------
        None
            The plotter should display when using this function
    """
    plotter = pyvista.Plotter()
    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius,show_bonds=True)
    def create_mesh_cube(value):
        plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],self.properties["cube"],opacity=opacity,factor=factor,cutoff=value)

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_cube, [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)
    plotter.show(full_screen=False)

    return




def plot_multiple_isodensities(base_name_file,list_files,plotted_isodensity="cube",delimiter="&",opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_multiple_isodensities(range_files,plotted_isodensity="cube",delimiter=" ",opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))

    Plot multiple isodensities, each one can be selected using a slider. The delimiter is replaced in the name file by number in the range_file to load multiple files
    Only one delimiter should be used in the base_name_file

    Parameters
    ----------
        base_name_file : str
            The base name of the files. If one wants to view files called H2CO.mo0a.cube, H2CO.mo1a.cube ...
            One should set the base_name_file as H2CO.mo&a.cube
        list_files : tuple
            The list of strings to replace the delimiter with
        plotted_isodensity : str, optional
            The isodensity to be plotted. By default "cube"
        delimiter : str, optional
            The
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
        cutoff : float, optional
            The initial cutoff of the isodensity plot. By default .2
        screenshot_button : bool, optional
            Adds a screenshot button. True by default
        window_size_screenshot : tuple, optional
            The size of the screenshots. By default (1000,1000)

    Returns
    -------
        None
            The plotter should display when using this function
    """

    plotter = pyvista.Plotter()

    def open_isodensity(value,cutoff):
        value = int(round(value))
        splitted_name = base_name_file.split(delimiter)
        file = splitted_name[0] + str(value) + splitted_name[1]
        mol = rgmol.extract_cube.extract(file,do_find_bonds=1)

        if with_radius:
            mol.plot(plotter,factor=factor_radius,opacity=opacity_radius,show_bonds=True)
        plot_cube(plotter,mol.properties["voxel_origin"],mol.properties["voxel_matrix"],mol.properties["cube"],opacity=opacity,factor=factor,cutoff=cutoff)

    slider_function = _slider(open_isodensity,1,cutoff)

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(lambda value: slider_function("AO",value), [int(list_files[0]),len(list_files)],value=1,title="Number", fmt="%1.0f")
    plotter.add_slider_widget(lambda value: slider_function("cutoff",value), [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)
    plotter.show(full_screen=False)
    return




def plot_AO(self,grid_points=(40,40,40),delta=3,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_AO(grid_points=(40,40,40),delta=3,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))

    Plot the Atomic Orbitals of a molecule
    The Atomic Orbitals will be calculated on the grid that will be defined by the number of grid points and around the molecule. The delta defines the length to be added to the extremities of the position of the atoms.
    The order of the Atomic Orbitals is defined in the molden file

    Parameters
    ----------
        grid_points : list of 3, optional
            The number of points for the grid in each dimension. By default (40,40,40)
        delta : float, optional
            The length added on all directions of the box containing all atomic centers. By default 3
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
        cutoff : float, optional
            The initial cutoff of the isodensity plot. By default .2
        screenshot_button : bool, optional
            Adds a screenshot button. True by default
        window_size_screenshot : tuple, optional
            The size of the screenshots. By default (1000,1000)

    Returns
    -------
        None
            The plotter should display when using this function
    """

    plotter = pyvista.Plotter()

    def create_mesh_AO(value,cutoff):
        plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],self.properties["AO_calculated"][int(round(value))-1],opacity=opacity,factor=factor,cutoff=cutoff)
        AO_number = int(round(value))

    if not "AO_calculated" in self.properties:
        rgmol.molecular_calculations.calculate_AO(self,grid_points,delta=delta)



    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)

    slider_function = _slider(create_mesh_AO,1,cutoff)

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(lambda value: slider_function("AO",value), [1, len(self.properties["AO_calculated"])],value=1,title="Number", fmt="%1.0f")
    plotter.add_slider_widget(lambda value: slider_function("cutoff",value), [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.show(full_screen=False)
    return





def plot_MO(self,grid_points=(40,40,40),delta=3,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_MO(grid_points=(40,40,40),delta=3,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))

    Plot the Molecular Orbitals of a molecule
    The Molecular Orbitals will be calculated on the grid that will be defined by the number of grid points and around the molecule. The delta defines the length to be added to the extremities of the position of the atoms.

    Parameters
    ----------
        grid_points : list of 3, optional
            The number of points for the grid in each dimension. By default (40,40,40)
        delta : float, optional
            The length added on all directions of the box containing all atomic centers. By default 3
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
        cutoff : float, optional
            The initial cutoff of the isodensity plot. By default .2
        screenshot_button : bool, optional
            Adds a screenshot button. True by default
        window_size_screenshot : tuple, optional
            The size of the screenshots. By default (1000,1000)

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

    def create_mesh_MO(value,cutoff):
        MO_number = int(round(value))
        MO_calculated = self.calculate_MO_chosen(MO_number-1,grid_points,delta=delta)

        plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],MO_calculated,opacity=opacity,factor=factor,cutoff=cutoff)
        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["MO_energy"][MO_number-1]),name="mo energy")
        print_occupancy(plotter,self.properties["MO_occupancy"],MO_number)

    slider_function = _slider(create_mesh_MO,1,cutoff)


    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(lambda value: slider_function("AO",value), [1, len(self.properties["MO_calculated"])],value=1,title="Number", fmt="%1.0f")
    plotter.add_slider_widget(lambda value: slider_function("cutoff",value), [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))

    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.show(full_screen=False)
    return



def plot_product_MO(self,grid_points=(40,40,40),delta=3,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_product_MO(grid_points=(40,40,40),delta=3,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))

    Plot the Molecular Orbitals of a molecule on the left and the right.
    The Molecular Orbitals will be calculated on the grid that will be defined by the number of grid points and around the molecule.
    The delta defines the length to be added to the extremities of the position of the atoms.
    The center part represents the product of both sides which is calculated when pushing the button.

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
        cutoff : float, optional
            The initial cutoff of the isodensity plot. By default .2
        screenshot_button : bool, optional
            Adds a screenshot button. True by default
        window_size_screenshot : tuple, optional
            The size of the screenshots. By default (1000,1000)

    Returns
    -------
        None
            The plotter should display when using this function
    """

    plotter = pyvista.Plotter(shape=(1,3),border=True)

    if not "MO_calculated" in self.properties:
        self.properties["MO_calculated"] = [[] for k in range(len(self.properties["MO_list"]))]

    if with_radius:
        plotter.subplot(0,0)
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)
        plotter.subplot(0,1)
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)
        plotter.subplot(0,2)
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)

    self.properties["MO_number_1"] = 1
    self.properties["MO_number_2"] = 1


    def create_mesh_MO_1(value):
        plotter.subplot(0,0)
        MO_number_1 = int(round(value))
        self.properties["MO_number_1"] = MO_number_1
        MO_calculated = self.calculate_MO_chosen(MO_number_1-1,grid_points,delta=delta)
        plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],MO_calculated,opacity=opacity,factor=factor,cutoff=cutoff,add_name="1")
        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["MO_energy"][MO_number_1-1]),name="mo energy")
        print_occupancy(plotter,self.properties["MO_occupancy"],MO_number_1)


    def create_mesh_MO_2(value):
        plotter.subplot(0,2)
        MO_number_2 = int(round(value))
        self.properties["MO_number_2"] = MO_number_2
        MO_calculated = self.calculate_MO_chosen(MO_number_2-1,grid_points,delta=delta)
        plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],MO_calculated,opacity=opacity,factor=factor,cutoff=cutoff,add_name="2")
        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["MO_energy"][MO_number_2-1]),name="mo energy")
        print_occupancy(plotter,self.properties["MO_occupancy"],MO_number_2)

    def calculate_product_MO(MO_ind_1,MO_ind_2):
        MO_1 = self.calculate_MO_chosen(MO_ind_1,grid_points,delta=delta)
        MO_2 = self.calculate_MO_chosen(MO_ind_2,grid_points,delta=delta)
        MO_prod = MO_1 * MO_2
        plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],MO_prod,opacity=opacity,factor=factor,cutoff=cutoff,add_name="prod")
        plotter.add_text(text=r"Product of {} and {}".format(MO_ind_1,MO_ind_2),name="mo prod")

    def button_product_MO(value):
        plotter.subplot(0,1)
        calculate_product_MO(self.properties["MO_number_1"],self.properties["MO_number_2"])




    plotter.subplot(0,0)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_MO_1, [1, len(self.properties["MO_calculated"])],value=1,title="Number", fmt="%1.0f")

    plotter.subplot(0,1)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_checkbox_button_widget(button_product_MO,size=80,color_off="blue")
    plotter.add_text(text="Calculate the product",name="calculate",position=(90.,20.))

    plotter.subplot(0,2)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_MO_2, [1, len(self.properties["MO_calculated"])],value=1,title="Number", fmt="%1.0f")

    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.show(full_screen=False)
    return



def plot_transition_density(self,grid_points=(40,40,40),delta=3,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_transition_density(grid_points=(40,40,40),delta=3,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))


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
        cutoff : float, optional
            The initial cutoff of the isodensity plot. By default .2
        screenshot_button : bool, optional
            Adds a screenshot button. True by default
        window_size_screenshot : tuple, optional
            The size of the screenshots. By default (1000,1000)

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


    def create_mesh_transition_density(value,cutoff):
        transition_number = int(round(value))
        transition_density_calculated = self.calculate_chosen_transition_density(transition_number-1,grid_points,delta=delta)

        plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],transition_density_calculated,opacity=opacity,factor=factor,cutoff=cutoff)

        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["transition_energy"][transition_number-1]),name="transition energy")

    slider_function = _slider(create_mesh_transition_density,1,cutoff)


    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)

    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.add_slider_widget(lambda value: slider_function("AO",value), [1, len(self.properties["transition_density_list"])],value=1,title="Number", fmt="%1.0f")
    plotter.add_slider_widget(lambda value: slider_function("cutoff",value), [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))
    plotter.show(full_screen=False)





def plot_diagonalized_kernel(self,kernel,method="only eigenmodes",plotting_method="isodensity",number_eigenvectors=20,grid_points=(20,20,20),delta=3,number_isodensities=10 ,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_diagonalized_kernel(kernel,method="only eigenmodes",plotting_method="isodensity",number_eigenvectors=20,grid_points=(20,20,20),delta=3,number_isodensities=10 ,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))


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
            Only used in the methods "total" and "partial". The number of eigenvectors to be computed
        grid_points : list of 3, optional
            The number of points for the grid in each dimension. By default (40,40,40)
        delta : float, optional
            The length added on all directions of the box containing all atomic centers. By default 3
        cutoff : float, optional
            The initial cutoff of the isodensity plot for the isodensity plotting method. By default .2
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
        screenshot_button : bool, optional
            Adds a screenshot button. True by default
        window_size_screenshot : tuple, optional
            The size of the screenshots. By default (1000,1000)

    Returns
    -------
        None
            The plotter should display when using this function


    Notes
    -----
        Because the kernels are 6-dimensional, they scale up drastically in terms of memory used.
        If one only wants to look at the eigenmodes, the "only eigenmodes" method is just that. It computes the eigenmodes without computing the total kernel. More info can be found :doc:`here<../orbitals/calculate_eigenmodes_linear_response_function>`.
        Otherwise, the partial method has been implemented which allows to remove the part of the space where the transition densities are almost zero. For each transition density, the space is sorted and the lower dense part that makes up to less than 1% is removed. In practice this removes as much as 90% of the space. More details on this method can be found :doc:`here<../orbitals/diagonalize_kernel>`.
        The last method is "total" which is just the calculation of the linear response on the whole space
    """

    if kernel != "linear_response_function":
        raise ValueError("Only linear response function implemented for now")

    if method == "only eigenmodes":
        self.calculate_eigenmodes_linear_response_function(grid_points,delta=delta)
        contrib_eigenvectors = self.properties["contibution_linear_response_eigenvectors"]


    else:
        self.diagonalize_kernel(kernel,number_eigenvectors,grid_points,method=method,delta=delta)

    eigenvectors = self.properties["linear_response_eigenvectors"]
    eigenvalues = self.properties["linear_response_eigenvalues"]
    transition_list = self.properties["transition_list"]
    transition_factor_list = self.properties["transition_factor_list"]
    plotter = pyvista.Plotter()
    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)

    if plotting_method == "isodensity":

        def create_mesh_diagonalized_kernel(value,cutoff):
            vector_number = int(round(value))
            plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],eigenvectors[vector_number-1],opacity=opacity,factor=factor,cutoff=cutoff)
            plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(eigenvalues[vector_number-1]),name="eigenvalue")

            if method == "only eigenmodes":
                print_contribution_transition_density(plotter,vector_number,contrib_eigenvectors,transition_list,transition_factor_list)


        slider_function = _slider(create_mesh_diagonalized_kernel,1,cutoff)
        plotter.add_slider_widget(lambda value: slider_function("cutoff",value), [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))

    elif plotting_method == "multiple isodensities":
        def create_mesh_diagonalized_kernel(value,cutoff):
            vector_number = int(round(value))
            plot_cube_multiple_isodensities(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],eigenvectors[vector_number-1],factor=factor)
            plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(eigenvalues[vector_number-1]),name="eigenvalue")

            if method == "only eigenmodes":
                print_contribution_transition_density(plotter,vector_number,contrib_eigenvectors,transition_list,transition_factor_list)

    elif plotting_method == "volume":
        def create_mesh_diagonalized_kernel(value,cutoff):
            vector_number = int(round(value))
            plot_cube_volume(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],eigenvectors[vector_number-1],factor=factor)
            plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(eigenvalues[vector_number-1]),name="eigenvalue")

            if method == "only eigenmodes":
                print_contribution_transition_density(plotter,vector_number,contrib_eigenvectors,transition_list,transition_factor_list)
    slider_function = _slider(create_mesh_diagonalized_kernel,1,cutoff)

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)

    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.add_slider_widget(lambda value: slider_function("AO",value), [1, len(eigenvectors)],value=1,title="Eigenvector", fmt="%1.0f")
    plotter.show(full_screen=False)



molecule.plot = plot
molecule.plot_vector = plot_vector
molecule.plot_radius = plot_radius
molecule.plot_property = plot_property
molecule.plot_diagonalized_condensed_kernel = plot_diagonalized_condensed_kernel
molecule.plot_isodensity = plot_isodensity
molecule.plot_multiple_isodensities = plot_multiple_isodensities
molecule.plot_AO = plot_AO
molecule.plot_MO = plot_MO
molecule.plot_product_MO = plot_product_MO
molecule.plot_transition_density = plot_transition_density
molecule.plot_diagonalized_kernel = plot_diagonalized_kernel

