#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rgmol
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extraction, computation and graphical display of adf file')
    # parser.add_argument('-i', default="output_examples//methanes//bromomethane//bromomethane.out", help='Input file of adf')
    parser.add_argument('-i', default="output_examples//input.out", help='Input file of adf')
    # parser.add_argument('-i', default="output_examples//SP_alpha_glucose_CDFT_W_Fukui.out", help='Input file of adf')
    args = parser.parse_args()

    file=args.i
    # file = "output_examples//methanes//bromomethane//bromomethane.out"



    # mol = rgmol.extract_adf.extract_all(file)


    # mol.plot_plt(ax,factor=.5)
    # mol.plot_radius_plt(factor=.3)

    # mol.plot_radius_plotly(factor=.3)
    # mol.plot_diagonalized_kernel_plt(plotted_kernel="condensed linear response",opacity=0.3,factor=1)
    # fig=mol.plot_diagonalized_kernel_plotly(plotted_kernel="condensed linear response",opacity=0.3,factor=1)
    # fig=mol.plot_diagonalized_kernel_slider_plotly(plotted_kernel="condensed linear response",opacity=0.3,factor=1)
    # fig=mol.plot_property_plotly(plotted_property="dual",opacity=0.3,factor=2,opacity_radius=.7)


    # mol.plot_radius_pyvista(factor=.3)
    # plotter=mol.plot_diagonalized_kernel_slider_pyvista(plotted_kernel="condensed linear response",opacity=0.3,factor=2)
    # mol.plot_property_pyvista(plotted_property="dual",opacity=0.3,factor=1,opacity_radius=.9)
    #

    # file = "output_examples//n1_CNNH2//n1_CNNH2.mo71a_100.cube"
    # mol=rgmol.extract_cube.extract(file)
    # mol.plot_cube_pyvista()

    cube = np.zeros((40,40,40))
    voxel_origin=[-.5,-.5,-.5]
    voxel_matrix = .5/40*2
    nx,ny,nz = np.shape(cube)
    voxel_end = [0,0,0]
    voxel_end[0] = voxel_origin[0] + voxel_matrix*nx
    voxel_end[1] = voxel_origin[1] + voxel_matrix*nx
    voxel_end[2] = voxel_origin[2] + voxel_matrix*nx


    x = np.linspace(voxel_origin[0],voxel_end[0],nx)
    y = np.linspace(voxel_origin[1],voxel_end[1],ny)
    z = np.linspace(voxel_origin[2],voxel_end[2],nz)

    r = x.reshape((1,nx,1,1))*np.array([1.,0,0]).reshape((3,1,1,1)) + y.reshape((1,1,ny,1))*np.array([0,1.,0]).reshape((3,1,1,1)) + z.reshape((1,1,1,nz))*np.array([0,0,1.]).reshape((3,1,1,1))
    import pyvista
    import plot_pyvista
    import extract_excited_states
    import time

    a = [104.8551864200     ,30.2811436880       ,10.6513942670        ,3.8699456233    ]

    b= [261.4493931636 , 176.6993802449,  76.9545600188, 17.5937671387]

    A=time.time()
    cube = extract_excited_states.gaussian_d(r,a,b,[0,0,0])[-1]
    B=time.time()
    print(B-A)
    plotter = pyvista.Plotter()
    plot_pyvista.plot_isodensity(plotter,voxel_origin,[[3/20,0,0],[0,3/20,0],[0,0,3/20]],cube,opacity=.7)
    C=time.time()
    print(C-B)
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)
    plotter.show(full_screen=False)

    # plt.show()