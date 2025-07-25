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
import matplotlib.pyplot as plt
from rgmol.objects import *
import rgmol.molecular_calculations
import rgmol.grid
import rgmol.rectilinear_grid_reconstruction
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

class _three_buttons:
    def __init__(self,func,value,mode_D,mode_P,mode_M):
        self.func = func
        self.value = value
        self.mode_D = mode_D
        self.mode_P = mode_P
        self.mode_M = mode_M

    def __call__(self,mode,value):
        if mode=="value":
            self.value = value
        elif mode=="d":
            self.mode_D = value
        elif mode=="p":
            self.mode_P = value
        elif mode=="m":
            self.mode_M = value
        self.func(self.value,self.mode_D,self.mode_P,self.mode_M)

##############################################
## Adding Plotting Methods for Atomic Grids ##
##############################################


def plot_atomic_grid(self,N_r_list=None,d_leb_list=None,zeta_list=None,alpha_list=None,opacity=0.5):
    """
    plot_atomic_grid(N_r_list=None,d_leb_list=None,zeta_list=None,alpha_list=None,opacity=0.5)

    Do a plot of the grid of the molecule, each point of the grid is represented as a point.
    The color of each point is the same as the atom that is at the center of the atomic grid
    For more information on the grids, please check :doc:`here<../grid/create_atomic_grid>`

    Parameters
    ----------
        N_r_list : list
            The number of radial points that will be taken for each atom repectively
        d_leb_list : list
            The degree of the Lebedev quadrature that will be taken for each atom respectively
        zeta_list : list
            The zeta parameter that will be taken for each atom repectively
        alpha_list : list
            The alpha parameter that will be taken for each atom repectively

    Returns
    -------
        None
            The plotter will show
    """
    if not self.mol_grids:
        rgmol.grid.create_grid_from_mol(self)


    plotter = pyvista.Plotter()


    for grid,atom in zip(self.mol_grids.grids,self.atoms):
        coords = grid.xyz_coords
        num_points = grid.number_points
        coords = coords.reshape((3,num_points)).transpose()
        points = pyvista.PolyData(coords)
        plotter.add_points(points,render_points_as_spheres=True,point_size=40,opacity=opacity,color=atom.color)

    for atom_x in self.atoms:
        atom_x.plot(plotter,opacity=1.0,factor=1)



    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.show(full_screen=False)
    return


def plot_on_atomic_grid(self,arr,N_r_list=None,d_leb_list=None,zeta_list=None,alpha_list=None):
    """
    plot_on_atomic_grid(self,arr,N_r_list=None,d_leb_list=None,zeta_list=None,alpha_list=None)

    Do a plot of an array on the atomic grids.
    The array should be of the shape : (N_grids, N_r, N_ang)
    For more information on the grids, please check :doc:`here<../grid/create_atomic_grid>`

    Parameters
    ----------
        arr : ndarray
            The array to be plotted, the shape must be (N_grids, N_r, N_ang)
        N_r_list : list, optional
            The number of radial points that will be taken for each atom repectively
        d_leb_list : list, optional
            The degree of the Lebedev quadrature that will be taken for each atom respectively
        zeta_list : list, optional
            The zeta parameter that will be taken for each atom repectively
        alpha_list : list, optional
            The alpha parameter that will be taken for each atom repectively

    Returns
    -------
        None
            The plotter will show
    """
    if not self.mol_grids:
        rgmol.grid.create_grid_from_mol(self,N_r_list=N_r_list,d_leb_list=d_leb_list,zeta_list=zeta_list,alpha_list=alpha_list)


    plotter = pyvista.Plotter()


    maxx = 0
    for grid,array in zip(self.mol_grids.grids,arr):
        wn = grid.wn
        dV = grid.dV
        val = np.max(abs(array*wn*dV))
        if val > maxx:
            maxx = val

    for grid,atom,array_to_plot in zip(self.mol_grids.grids,self.atoms,arr):
        coords = grid.xyz_coords
        num_points = grid.number_points
        coords = coords.reshape((3,num_points)).transpose()
        points = pyvista.PolyData(coords)
        wn = grid.wn
        dV = grid.dV

        points = plotter.add_points(points,render_points_as_spheres=True,point_size=40,opacity=[1.0,0.4,0.0,0.4,1.0],scalars=array_to_plot*wn*dV,clim=(-maxx,maxx),cmap="rainbow4")


    for atom_x in self.atoms:
        atom_x.plot(plotter,opacity=0.3,factor=0.5)
    bonds_plotting(plotter,self.bonds,self.list_property("pos"),self.list_property("radius"),factor=.5)



    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.show(full_screen=False)
    return

molecule.plot_atomic_grid = plot_atomic_grid
molecule.plot_on_atomic_grid = plot_on_atomic_grid


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


def plot_condensed_kernel(self,kernel,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_condensed_kernel(kernel,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,screenshot_button=True,window_size_screenshot=(1000,1000))

    Plots a condensed kernel for a molecule. One can navigate through the vectors using a slider.

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
    ncols = len(X)

    plotter = pyvista.Plotter()
    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)
    def create_mesh_diagonalized_kernel(value):
        vector_number = int(round(value))
        plot_single_atom(plotter,self.atoms[vector_number-1],factor=factor_radius*1.5)
        self.plot_vector(plotter,X[:,vector_number-1],opacity=opacity,factor=factor)


    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_diagonalized_kernel, [1, len(X)],value=1,title="Vector", fmt="%1.0f")
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)
    plotter.show(full_screen=False)
    return

def plot_diagonalized_condensed_kernel(self,kernel,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_diagonalized_condensed_kernel(kernel,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,screenshot_button=True,window_size_screenshot=(1000,1000))

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

#########################
## General 3D plotting ##
#########################

def plot_isodensity(self,plotted_isodensity="cube",opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_isodensity(plotted_isodensity="cube",opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))

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
    r,coords = rgmol.grid.create_cubic_grid_from_molecule(self,grid_points=np.shape(self.properties[plotted_isodensity]))
    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius,show_bonds=True)
    def create_mesh_cube(value):
        plot_cube(plotter,coords,self.properties[plotted_isodensity],opacity=opacity,factor=factor,cutoff=value)

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_cube, [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)
    plotter.show(full_screen=False)

    return




def plot_multiple_isodensities(base_name_file,list_files,plotted_isodensity="cube",delimiter="&",opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_multiple_isodensities(range_files,plotted_isodensity="cube",delimiter=" ",opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))

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




############################################
## Plotting Atomic / Molecular Properties ##
############################################

def plot_AO(self,grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_AO(grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))

    Plot the Atomic Orbitals of a molecule.
    Because no calculations are done, the Atomic Orbitals will be calculated on the representative grid.
    The representative grid is a cubic grid around the molecule.
    The delta defines the length to be added to the extremities of the position of the atoms.
    The order of the Atomic Orbitals is defined in the molden file

    Parameters
    ----------
        grid_points : list of 3, optional
            The number of points for the representative grid in each dimension. By default (80,80,80)
        delta : float, optional
            The length added in all directions for the construction of the representative grid. By default 5
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
        AO_number = int(round(value)) - 1
        AO = AO_calculated[AO_number]
        AO = np.sign(AO)*AO**2
        plot_cube(plotter,coords,AO,opacity=opacity,factor=factor,cutoff=cutoff)

    coords,AO_calculated = rgmol.rectilinear_grid_reconstruction.reconstruct_AO(self,grid_points=grid_points,delta=delta)

    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)

    slider_function = _slider(create_mesh_AO,1,cutoff)

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(lambda value: slider_function("AO",value), [1, len(AO_calculated)],value=1,title="Number", fmt="%1.0f")
    plotter.add_slider_widget(lambda value: slider_function("cutoff",value), [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.show(full_screen=False)
    return





def plot_MO(self,grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_MO(grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))

    Plot the Molecular Orbitals of a molecule.
    Because no calculations are done, the Molecular Orbitals will be calculated on the representative grid.
    The representative grid is a cubic grid around the molecule.
    The delta defines the length to be added to the extremities of the position of the atoms.

    Parameters
    ----------
        grid_points : list of 3, optional
            The number of points for the representative grid in each dimension. By default (80,80,80)
        delta : float, optional
            The length added in all directions for the construction of the representative grid. By default 5
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

    coords,MO_calculated = rgmol.rectilinear_grid_reconstruction.reconstruct_MO(self,grid_points=grid_points,delta=delta)


    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)

    def create_mesh_MO(value,cutoff):
        MO_number = int(round(value))
        MO = MO_calculated[MO_number-1]
        MO = np.sign(MO)*MO**2

        plot_cube(plotter,coords,MO,opacity=opacity,factor=factor,cutoff=cutoff)
        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["MO_energy"][MO_number-1]),name="mo energy")
        print_occupancy(plotter,self.properties["MO_occupancy"],MO_number)

    slider_function = _slider(create_mesh_MO,1,cutoff)


    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(lambda value: slider_function("AO",value), [1, len(MO_calculated)],value=1,title="Number", fmt="%1.0f")
    plotter.add_slider_widget(lambda value: slider_function("cutoff",value), [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))

    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.show(full_screen=False)
    return



def plot_product_MO(self,grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_product_MO(grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))

    Plot the Molecular Orbitals of a molecule on the left and the right.
    The Molecular Orbitals will be calculated on the grid that will be defined by the number of grid points and around the molecule.
    The delta defines the length to be added to the extremities of the position of the atoms.
    The center part represents the product of both sides which is calculated when pushing the button.

    Parameters
    ----------
        grid_points : list of 3, optional
            The number of points for the representative grid in each dimension. By default (80,80,80)
        delta : float, optional
            The length added in all directions for the construction of the representative grid. By default 3
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
    coords,AO_calculated = rgmol.rectilinear_grid_reconstruction.reconstruct_AO(self,grid_points=grid_points,delta=delta)
    MO_list = self.properties["MO_list"]

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
        coords,MO = rgmol.rectilinear_grid_reconstruction.reconstruct_chosen_MO(self,MO_number_1-1,grid_points=grid_points,delta=delta)
        plot_cube(plotter,coords,MO,opacity=opacity,factor=factor,cutoff=cutoff,add_name="1")
        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["MO_energy"][MO_number_1-1]),name="mo energy")
        print_occupancy(plotter,self.properties["MO_occupancy"],MO_number_1)


    def create_mesh_MO_2(value):
        plotter.subplot(0,2)
        MO_number_2 = int(round(value))
        self.properties["MO_number_2"] = MO_number_2
        coords,MO = rgmol.rectilinear_grid_reconstruction.reconstruct_chosen_MO(self,MO_number_2-1,grid_points=grid_points,delta=delta)
        plot_cube(plotter,coords,MO,opacity=opacity,factor=factor,cutoff=cutoff,add_name="2")
        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["MO_energy"][MO_number_2-1]),name="mo energy")
        print_occupancy(plotter,self.properties["MO_occupancy"],MO_number_2)


    def calculate_product_MO(MO_ind_1,MO_ind_2):
        coords,MO_1 = rgmol.rectilinear_grid_reconstruction.reconstruct_chosen_MO(self,MO_ind_1-1,grid_points=grid_points,delta=delta)
        coords,MO_2 = rgmol.rectilinear_grid_reconstruction.reconstruct_chosen_MO(self,MO_ind_2-1,grid_points=grid_points,delta=delta)
        MO_prod = MO_1 * MO_2
        plot_cube(plotter,coords,MO_prod,opacity=opacity,factor=factor,cutoff=cutoff,add_name="prod")
        plotter.add_text(text=r"Product of {} and {}".format(MO_ind_1,MO_ind_2),name="mo prod")

    def button_product_MO(value):
        plotter.subplot(0,1)
        calculate_product_MO(self.properties["MO_number_1"],self.properties["MO_number_2"])




    plotter.subplot(0,0)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_MO_1, [1, len(MO_list)],value=1,title="Number", fmt="%1.0f")

    plotter.subplot(0,1)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_checkbox_button_widget(button_product_MO,size=80,color_off="blue")
    plotter.add_text(text="Calculate the product",name="calculate",position=(90.,20.))

    plotter.subplot(0,2)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_MO_2, [1, len(MO_list)],value=1,title="Number", fmt="%1.0f")

    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.show(full_screen=False)
    return



def plot_electron_density(self,grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_electron_density(grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))


    Plot the Electron Density of a molecule.
    Because no calculations are truly done, the electron density will be calculated on the representative grid.
    The representative grid is a cubic grid around the molecule.
    The delta defines the length to be added to the extremities of the position of the atoms.
    The order of the Atomic Orbitals is defined in the molden file

    Parameters
    ----------
        grid_points : list of 3, optional
            The number of points for the representative grid in each dimension. By default (80,80,80)
        delta : float, optional
            The length added in all directions for the construction of the representative grid. By default 5
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

    plotter = pyvista.Plotter()
    coords,electron_density = rgmol.rectilinear_grid_reconstruction.reconstruct_electron_density(self,grid_points)

    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)

    def create_mesh_cube(value):
        plot_cube(plotter,coords,electron_density,opacity=opacity,factor=factor,cutoff=value)

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_cube, [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)
    plotter.show(full_screen=False)



def plot_fukui_function(self,mol_p=None,mol_m=None,fukui_type="0",grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_fukui_function(mol_p=None,mol_m=None,fukui_type="0",grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))


    Plots the fukui function desired.
    The fukui functions are calculated using finite differences of electron density.
    Thus, the molecule with one electron added (mol_p) or removed (mol_m) must be computed and added as argument to this function.
    Because no calculations are truly done, the fukui function will be calculated on the representative grid.
    The representative grid is a cubic grid around the molecule.
    The delta defines the length to be added to the extremities of the position of the atoms.
    The order of the Atomic Orbitals is defined in the molden file

    Parameters
    ----------
        mol_p : molecule, optional
            The molecule with an electron added. Needed for calculating the softness kernel with a fukui_type of "0" or "+"
        mol_n : molecule, optional
            The molecule with an electron removed. Needed for calculating the softness kernel with a fukui_type of "0" or "-"
        fukui_type : molecule, optional
            The type of fukui function used to calculate the softness kernel. The available types are "0", "+" or "-"
        grid_points : list of 3, optional
            The number of points for the representative grid in each dimension. By default (80,80,80)
        delta : float, optional
            The length added in all directions for the construction of the representative grid. By default 5
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

    plotter = pyvista.Plotter()

    coords,f0,fp,fm = rgmol.rectilinear_grid_reconstruction.reconstruct_fukui_function(self,mol_p=mol_p,mol_m=mol_m,grid_points=grid_points,delta=delta)

    if "0" in fukui_type:
        fukui = f0
    elif "+" in fukui_type or "p" in fukui_type:
        fukui = fp
    elif "-" in fukui_type or "m" in fukui_type:
        fukui = fm
    else:
        raise ValueError('This kind of fukui function does not exist. Please chose between 0, + or -')

    if type(fukui) is None:
        raise TypeError("This fukui function could not be computed, did you give mol_m or mol_p ?")


    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)

    def create_mesh_cube(value):
        plot_cube(plotter,coords,fukui,opacity=opacity,factor=factor,cutoff=value)

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.add_slider_widget(create_mesh_cube, [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))
    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)
    plotter.show(full_screen=False)




def plot_dipole_moment(self,opacity=0.8,factor=1,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_dipole_moment(opacity=0.8,factor=1,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))


    Plots the electric and magnetic dipole moments of a molecule.
    Data has to be extracted from an orca TDDFT calculation using :doc:`rgmol.extract_orca.extract_transition<../extract_orca>`.

    Parameters
    ----------
        opacity : float, optional
            The opacity of the arrow. By default .5
        factor : float, optional
            The factor by which the norm of the arrow is multiplied. By default 1
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

    center_of_mass = self.properties["center_of_mass"]
    D = self.properties["D"]
    P = self.properties["P"]
    M = self.properties["M"]
    state_transition = self.properties["state_transition"]

    self.plot(plotter,factor=factor_radius,opacity=opacity_radius)

    def create_mesh_dipole_moment(value,mode_D,mode_P,mode_M):
        transition_number = int(round(value))
        directions = [[],[],[]]
        if mode_D: directions[0] = np.array(D[transition_number-1])
        if mode_P: directions[1] = np.array(P[transition_number-1])
        if mode_M: directions[2] = np.array(M[transition_number-1])
        compt = 1
        for direc in range(3):
            direction = directions[direc]
            if len(direction)>0:
                norm = np.sum(direction**2)**(1/2)
                dipole_moment = pyvista.Arrow(start=center_of_mass,direction=direction,scale=factor*norm)
                if direc==0:
                    plotter.add_mesh(dipole_moment,opacity=opacity,pbr=True,roughness=.5,metallic=.2,color="blue",name="dipole moment{}".format(direc))
                    plotter.add_text(text="Electric dipole\nNorm = "+'{:3.3f} (a.u.)'.format(norm),name="norm{}".format(direc),position=(0.05,0.8-.07*(2-direc)),viewport=True,font_size=int(15*plotter.window_size[1]/1000),color="blue")
                elif direc==1:
                    plotter.add_mesh(dipole_moment,opacity=opacity,pbr=True,roughness=.5,metallic=.2,color="green",name="dipole moment{}".format(direc))
                    plotter.add_text(text="Electric dipole velocity\nNorm = "+'{:3.3f} (a.u.)'.format(norm),name="norm{}".format(direc),position=(0.05,0.8-.07*(2-direc)),viewport=True,font_size=int(15*plotter.window_size[1]/1000),color="green")
                else:
                    plotter.add_mesh(dipole_moment,opacity=opacity,pbr=True,roughness=.5,metallic=.2,color="red",name="dipole moment{}".format(direc))
                    plotter.add_text(text="Magnetic dipole\nNorm = "+'{:3.3f} (a.u.)'.format(norm),name="norm{}".format(direc),position=(0.05,0.8-.07*(2-direc)),viewport=True,font_size=int(15*plotter.window_size[1]/1000),color="red")
                compt+=1
            else:
                plotter.remove_actor("norm{}".format(direc))
                plotter.remove_actor("dipole moment{}".format(direc))

            plotter.add_text(text=r"Transition :"+ state_transition[transition_number-1] ,name="transi num",position=(0.05,.9),viewport=True)
        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["energy_transition_spectra"][transition_number-1]),name="transition energy")


    three_buttons = _three_buttons(create_mesh_dipole_moment,1,0,0,0)


    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)

    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.add_slider_widget(lambda value: three_buttons("value",value), [1, len(D)],value=1,title="Transition number", fmt="%1.0f")
    plotter.add_checkbox_button_widget(lambda value_d: three_buttons("d",value_d) ,size=80,position=(20.,150))
    plotter.add_checkbox_button_widget(lambda value_p: three_buttons("p",value_p) ,size=80,position=(20.,250),color_on="green")
    plotter.add_checkbox_button_widget(lambda value_m: three_buttons("m",value_m) ,size=80,position=(20.,350),color_on='red')



    plotter.show(full_screen=False)


def plot_transition_density(self,grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plot_transition_density(grid_points=(80,80,80),delta=5,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000))


    Plot the Transition Densities of a molecule.
    All the AOs and the MOs will be calculated on the grid if they were not calculated.
    Because no calculations are truly done, the electron density will be calculated on the representative grid.
    The representative grid is a cubic grid around the molecule.
    The delta defines the length to be added to the extremities of the position of the atoms.
    The order of the Atomic Orbitals is defined in the molden file

    Parameters
    ----------
        grid_points : list of 3, optional
            The number of points for the representative grid in each dimension. By default (80,80,80)
        delta : float, optional
            The length added in all directions for the construction of the representative grid. By default 5
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
    # coords,transition_density_list = rgmol.rectilinear_grid_reconstruction.reconstruct_transition_density(self,grid_points=grid_points,delta=delta)
    coords,MO_calculated = rgmol.rectilinear_grid_reconstruction.reconstruct_MO(self,grid_points=grid_points,delta=delta)


    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)


    def create_mesh_transition_density(value,cutoff):
        transition_number = int(round(value))
        coords,transition_density_calculated = rgmol.rectilinear_grid_reconstruction.reconstruct_chosen_transition_density(self,transition_number-1,grid_points=grid_points,delta=delta)
        plot_cube(plotter,coords,transition_density_calculated,opacity=opacity,factor=factor,cutoff=cutoff)


        plotter.add_text(text=r"Energy = "+'{:3.3f} (a.u.)'.format(self.properties["transition_energy"][transition_number-1]),name="transition energy")

    slider_function = _slider(create_mesh_transition_density,1,cutoff)


    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)

    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.add_slider_widget(lambda value: slider_function("AO",value), [1, len(self.properties["transition_list"])],value=1,title="Number", fmt="%1.0f")
    plotter.add_slider_widget(lambda value: slider_function("cutoff",value), [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))
    plotter.show(full_screen=False)







def plot_diagonalized_kernel(self,kernel,mol_p=None,mol_m=None,fukui_type="0",number_plotted_eigenvectors=100,try_reading=True,save=True ,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000),grid_points=(80,80,80),delta=5):
    """
    plot_diagonalized_kernel(kernel,mol_p=None,mol_m=None,fukui_type="0",number_plotted_eigenvectors=100,try_reading=True,save=True,plotting_method="isodensity",number_isodensities=10 ,opacity=0.8,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000),grid_points=(80,80,80),delta=5)

    Calculate and diagonalize a kernel.
    The available kernels are : "linear_response_function" and "softness_kernel"
    Only the eigenmodes are computed using a method that does not calculate the whole kernel.
    All the computations will be done on the atomic grids.

    The eigenmodes will also be computed on the atomic grids.
    One might consider using :doc:`plot_on_atomic_grid<plot_on_atomic_grid>` for visualization

    By default the eigenmodes will also be plotted on the representative grid using the calculations on the atomic grids.
    If one wants not to do the plot, putting number_plotted_eigenvectors to 0 will not produce any plot.

    For the softness kernel as it is calculated using the Parr-Berkowitz relation, the fukui functions need to be computed. For that, a calculation adding (mol_p) or removing an electron (mol_m) needs to be done with the same geometry.

    Parameters
    ----------
        kernel : str
            The kernel to be diagonalized and plotted
        mol_p : molecule, optional
            The molecule with an electron added. Needed for calculating the softness kernel with a fukui_type of "0" or "+"
        mol_m : molecule, optional
            The molecule with an electron removed. Needed for calculating the softness kernel with a fukui_type of "0" or "-"
        fukui_type : molecule, optional
            The type of fukui function used to calculate the softness kernel. The available types are "0", "+" or "-"
        number_plotted_eigenvectors : int, optional
            The Number of eigenvectors to be plotted. If 0, no plot will be produced. By default 100
        try_reading : bool, optional
            If the eigenmodes were calculated and saved before, try reading them from the rgmol folder. If False, the computation will be remade. True by default
        save : bool, optional
            The eigenmodes, eigenvalues and contributions of transition densities will be save in the folder rgmol that will be created where the molecule file is located
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
            The initial cutoff of the isodensity plot for the isodensity plotting method. By default .2
        screenshot_button : bool, optional
            Adds a screenshot button. True by default
        window_size_screenshot : tuple, optional
            The size of the screenshots. By default (1000,1000)
        grid_points : list of 3, optional
            The number of points for the representative grid in each dimension. By default (80,80,80)
        delta : float, optional
            The length added in all directions for the construction of the representative grid. By default 5

    Returns
    -------
        None
            The plotter should display when using this function and all the properties will be sorted inside the molecule.properties
    """
    did_read = 1
    if kernel == "linear_response_function":
        if "Reconstructed_linear_response_eigenvectors" in self.properties:
            print("############################################")
            print("# Eigenmodes already extracted or computed #")
            print("############################################")


        elif try_reading:
            try: self.read()
            except: pass

        if try_reading and not "Reconstructed_linear_response_eigenvectors" in self.properties:
            did_read = 0
            if not "contribution_linear_response_eigenvectors" in self.properties:
                print("########################################")
                print("# No previous calculations were found. #")
                print("#   The eigenmodes will be computed.   #")
                print("########################################")
                self.calculate_eigenmodes_linear_response_function()
            else:
                print("#########################################")
                print("#    Previous calculation found but     #")
                print("# the eigenmodes were not reconstructed #")
                print("#########################################")
        elif not try_reading:
            did_read = 0
            self.calculate_eigenmodes_linear_response_function()

        # eigenvectors = self.properties["linear_response_eigenvectors"]
        eigenvalues = self.properties["linear_response_eigenvalues"]
        contrib_eigenvectors = self.properties["contribution_linear_response_eigenvectors"]
        transition_list = self.properties["transition_list"]
        transition_factor_list = self.properties["transition_factor_list"]

        coords,eigenvectors = rgmol.rectilinear_grid_reconstruction.reconstruct_eigenvectors(self,kernel)

    elif kernel == "softness_kernel":
        self.calculate_hardness()
        if try_reading:
            try: self.read()
            except: pass

        if try_reading and not "Reconstructed_softness_kernel_eigenvectors" in self.properties:
            did_read = 0
            if not "contribution_softness_kernel_eigenvectors" in self.properties:
                print("########################################")
                print("# No previous calculations were found. #")
                print("#   The eigenmodes will be computed.   #")
                print("########################################")
                self.calculate_softness_kernel_eigenmodes(mol_p=mol_p,mol_m=mol_m,fukui_type=fukui_type)
            else:
                print("#########################################")
                print("#    Previous calculation found but     #")
                print("# the eigenmodes were not reconstructed #")
                print("#########################################")
        elif not try_reading:
            did_read = 0
            self.calculate_softness_kernel_eigenmodes(mol_p=mol_p,mol_m=mol_m,fukui_type=fukui_type)

        # eigenvectors = self.properties["softness_kernel_eigenvectors"]
        eigenvalues = self.properties["softness_kernel_eigenvalues"]
        contrib_eigenvectors = self.properties["contribution_softness_kernel_eigenvectors"]
        transition_list = self.properties["transition_list"] + [[[-1,-1]]]
        transition_factor_list = self.properties["transition_factor_list"] + [[[1]]]

        coords,eigenvectors = rgmol.rectilinear_grid_reconstruction.reconstruct_eigenvectors(self,kernel,mol_m=mol_m,mol_p=mol_p,grid_points=grid_points,delta=delta,fukui_type=fukui_type)


    else:
        raise TypeError("The kernel {} has not been implemented. The available kernels are : linear_response_function and softness_kernel".format(kernel))

    if save and not did_read:
        self.save(kernel)

    #Stop here if no eigenvectors are to be plotted
    if number_plotted_eigenvectors < 1:
        return


    eigenvectors = eigenvectors [:number_plotted_eigenvectors]
    eigenvalues = eigenvalues [:number_plotted_eigenvectors]


    plotter = pyvista.Plotter()
    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)


    # if plotting_method == "isodensity":
    def create_mesh_diagonalized_kernel(value,cutoff):
        vector_number = int(round(value))
        plot_cube(plotter,coords,eigenvectors[vector_number-1],opacity=opacity,factor=factor,cutoff=cutoff)
        plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(eigenvalues[vector_number-1]),name="eigenvalue")

        print_contribution_transition_density(plotter,self,kernel,vector_number,contrib_eigenvectors,transition_list,transition_factor_list)

    slider_function = _slider(create_mesh_diagonalized_kernel,1,cutoff)
    plotter.add_slider_widget(lambda value: slider_function("cutoff",value), [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))

    # elif plotting_method == "multiple isodensities":
    #     def create_mesh_diagonalized_kernel(value,cutoff):
    #         vector_number = int(round(value))
    #         plot_cube_multiple_isodensities(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],eigenvectors[vector_number-1],factor=factor)
    #         plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(eigenvalues[vector_number-1]),name="eigenvalue")
    #
    #         print_contribution_transition_density(plotter,self,kernel,vector_number,contrib_eigenvectors,transition_list,transition_factor_list)


    slider_function = _slider(create_mesh_diagonalized_kernel,1,cutoff)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)

    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.add_slider_widget(lambda value: slider_function("AO",value), [1, len(eigenvectors)],value=1,title="Eigenvector", fmt="%1.0f")
    plotter.show(full_screen=False)




def plot_analysis_kernel(self,kernel,mol_p=None,mol_m=None,fukui_type="0",list_vectors=[1,2,3,4,5],try_reading_diagonalized=True,save_diagonalized=True,save=True,file_name="analysis"):
    """
    plot_analysis_kernel(kernel,mol_p=None,mol_m=None,fukui_type="0",list_vectors=[1,2,3,4,5],try_reading_diagonalized=True,save_diagonalized=True,save=1,file_name="analysis")

    Computes and plot some analysis of the eigenmodes of the linear_response_function or the softness_kernel.
    The analysis are : Percentage of occupied MO in the mode, percentage of virtual MO in the mode.
    For the softness kernel, the proportion of electron transferred in each mode.
    By default this will produce a file called analysis.png

    Parameters
    ----------
        kernel : str
            The kernel on which the analysis will be made. Either "linear_reponse_function" or "softness_kernel"
        mol_p : molecule, optional
            The molecule with an electron added. Needed for calculating the softness kernel with a fukui_type of "0" or "+"
        mol_m : molecule, optional
            The molecule with an electron removed. Needed for calculating the softness kernel with a fukui_type of "0" or "-"
        fukui_type : molecule, optional
            The type of fukui function used to calculate the softness kernel. The available types are "0", "+" or "-"
        list_vectors : list, optional
            The list of the vectors that will be decomposed on Occupied Virtual orbitals. Starting to 1 instead of 0. By default [1,2,3,4,5]
        try_reading_diagonalized : bool, optional
            If the eigenmodes will be read or not. By default True
        save_diagonalized : bool, optional
            If the eigenmodes will be saved or not. By default True
        save : bool, optional
            If the image is saved or showed. By default True
        file_name : str, optional
            The name of the file, the extension is not necessary as the default one is png. But if one wants to save in another format, putting the format will save the file in that format. By default "analysis"

    Returns
    -------
        None
            The image is either save or plotted
    """

    if kernel == "linear_response_function":
        if not "linear_response_eigenvectors" in self.list_properties():
            self.plot_diagonalized_kernel(kernel,number_plotted_eigenvectors=0,try_reading=try_reading_diagonalized,save=save_diagonalized)

    elif kernel == "softness_kernel":
        if not "softness_kernel_eigenvectors" in self.list_properties():
            self.plot_diagonalized_kernel(kernel,number_plotted_eigenvectors=0,mol_p=mol_p,mol_m=mol_m,fukui_type=fukui_type,try_reading=try_reading_diagonalized,save=save_diagonalized)

    else:
        raise ValueError("The kernel {} has not been implemented. The available kernels are : linear_response_function and softness_kernel".format(kernel))

    self.analysis_eigenmodes(kernel=kernel,list_vectors=list_vectors)

    from_occ = np.array(self.properties["from_occ"])
    to_virt = np.array(self.properties["to_virt"])

    colors = ["red","blue","green","darkred","cyan","lime","magenta","teal","purple","darkorange"]

    fig=plt.figure(figsize=(15,7),dpi=200)
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['svg.fonttype'] = 'none'

    if kernel == "linear_response_function":
        plt.subplot(1,2,1)
        for index in range(len(from_occ)):
            plt.plot(from_occ[index],"o-",color=colors[index],label=list_vectors[index])

    elif kernel == "softness_kernel":
        plt.subplot(2,2,1)
        for index in range(len(from_occ)):
            plt.plot(from_occ[index,:-1],"o-",color=colors[index],label=list_vectors[index])
            plt.plot(len(from_occ[0])-1,from_occ[index,-1],"o",color=colors[index])
        plt.xticks(ticks=np.arange(len(from_occ[0])),label=[str(k) for k in range(len(from_occ[0])-1)] + ["f"])

    plt.xlabel("Occupied MO index",size=17)
    plt.ylabel("Coefficient",size=17)
    plt.legend()

    n_occ = len(from_occ[0])
    n_virt = len(to_virt[0])

    if kernel == "linear_response_function":
        plt.subplot(1,2,2)
    elif kernel == "softness_kernel":
        plt.subplot(2,2,2)
        n_occ-=1
    for index in range(len(to_virt)):
        plt.plot(np.arange(n_occ,n_virt),to_virt[index][n_occ:],"o-",color=colors[index],label=list_vectors[index])
    plt.xlabel("Virtual MO index",size=17)
    plt.ylabel("Coefficient",size=17)

    if kernel == "softness_kernel":
        plt.subplot(2,1,2)
        fukui_decomposition = self.properties["fukui_decomposition"]
        plt.plot(fukui_decomposition,"o-r")
        plt.xlabel("Eigenmode Index",size=17)
        plt.ylabel("Electron Transfer Coefficient",size=17)
        # plt.ylim(0,1)


    plt.tight_layout()
    if save:
        plt.savefig(file_name)
    else: plt.show()






molecule.plot = plot
molecule.plot_vector = plot_vector
molecule.plot_radius = plot_radius
molecule.plot_property = plot_property
molecule.plot_condensed_kernel = plot_condensed_kernel
molecule.plot_diagonalized_condensed_kernel = plot_diagonalized_condensed_kernel
molecule.plot_isodensity = plot_isodensity
molecule.plot_multiple_isodensities = plot_multiple_isodensities
molecule.plot_AO = plot_AO
molecule.plot_MO = plot_MO
molecule.plot_product_MO = plot_product_MO
molecule.plot_electron_density = plot_electron_density
molecule.plot_dipole_moment = plot_dipole_moment
molecule.plot_transition_density = plot_transition_density
molecule.plot_fukui_function = plot_fukui_function
molecule.plot_diagonalized_kernel = plot_diagonalized_kernel
molecule.plot_analysis_kernel = plot_analysis_kernel


