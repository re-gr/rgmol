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




def plot_NOCV(self,list_fragments,grid_points=(100,100,100),delta=10,opacity=0.5,factor=1,with_radius=True,opacity_radius=1,factor_radius=.3,cutoff=.2,screenshot_button=True,window_size_screenshot=(1000,1000)):
    """
    plotsNOCV

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

    """

    if not "NOCV" in self.properties:
        self.calculate_NOCV(list_fragments,grid_points=grid_points,delta=delta)


    eigenvectors = self.properties["NOCV"]
    eigenvalues = self.properties["NOCV_eigenvalues"]

    plotter = pyvista.Plotter()
    if with_radius:
        self.plot(plotter,factor=factor_radius,opacity=opacity_radius)

    def create_mesh_NOCV(value,cutoff):
        vector_number = int(round(value))
        plot_cube(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],eigenvectors[vector_number-1],opacity=opacity,factor=factor,cutoff=cutoff)
        plotter.add_text(text=r"eigenvalue = "+'{:3.3f} (a.u.)'.format(eigenvalues[vector_number-1]),name="eigenvalue")

    slider_function = _slider(create_mesh_NOCV,1,cutoff)
    plotter.add_slider_widget(lambda value: slider_function("cutoff",value), [1e-6,1-1e-6],value=cutoff,title="Cutoff", fmt="%1.2f",pointa=(0.1,.9),pointb=(0.35,.9))

    slider_function = _slider(create_mesh_NOCV,1,cutoff)

    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)

    if screenshot_button:
        add_screenshot_button(plotter,window_size_screenshot)

    plotter.add_slider_widget(lambda value: slider_function("AO",value), [1, len(eigenvectors)],value=1,title="Eigenvector", fmt="%1.0f")
    plotter.show(full_screen=False)


molecule.plot_NOCV = plot_NOCV
