#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rgmol
import plot_pyvista
import argparse
import numpy as np


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


    # file = "output_examples//n1_CNNH2//n1_CNNH2.mo71a_100.cube"
    # mol=rgmol.extract_cube.extract(file)
    # mol.plot_cube_pyvista()

    # file = "output_examples//br//br.molden.input"
    # file = "output_examples//hcl//hcl.molden.input"
    # file = "output_examples//chfclbr//chfclbr.molden.input"
    # file = "output_examples//CH3Br//CH3Br.molden.input"
    file = "output_examples//CH3Cl//CH3Cl.molden.input"
    # file = "output_examples//n1_CNNH2//n1_CNNH2.molden.input"
    # file = "output_examples//acrolein//acrolein.molden.input"
    # file = "output_examples//H2CO//H2CO.molden.input"
    mol = rgmol.extract_excited_states.extract_molden(file)
    # transition_energy,transition_list,transition_factor_list = rgmol.extract_excited_states.extract_transition_orca("output_examples//H2CO//H2CO.out")
    transition_energy,transition_list,transition_factor_list = rgmol.extract_excited_states.extract_transition_orca("output_examples//CH3Cl//CH3Cl.out")
    # transition_energy,transition_list,transition_factor_list = rgmol.extract_excited_states.extract_transition_orca("output_examples//CH3Br//CH3Br.out")

    mol.properties["transition_energy"] = transition_energy
    mol.properties["transition_list"] = transition_list
    mol.properties["transition_factor_list"] = transition_factor_list



    # mol.plot_AO_pyvista(delta=5,grid_points=(60,60,60))
    mol.plot_transition_density_pyvista(delta=5,grid_points=(100,100,100))




    # mol.plot_MO_pyvista(grid_points=(60,60,60))


    #
    # cube = np.zeros((40,40,40))
    # voxel_origin=[-.5,-.5,-.5]
    # voxel_matrix = .5/40*2
    # nx,ny,nz = np.shape(cube)
    #
    # import pyvista
    # import plot_pyvista
    # import extract_excited_states
    # import time
    # import calculate_mo
    #
    # voxel_end = [0,0,0]
    #
    # voxel_end[0] = voxel_origin[0] + voxel_matrix*nx
    # voxel_end[1] = voxel_origin[1] + voxel_matrix*ny
    # voxel_end[2] = voxel_origin[2] + voxel_matrix*nz
    #
    #
    # x = np.linspace(voxel_origin[0],voxel_end[0],nx)
    # y = np.linspace(voxel_origin[1],voxel_end[1],ny)
    # z = np.linspace(voxel_origin[2],voxel_end[2],nz)
    #
    # r = x.reshape((1,nx,1,1))*np.array([1.,0,0]).reshape((3,1,1,1)) + y.reshape((1,1,ny,1))*np.array([0,1.,0]).reshape((3,1,1,1)) + z.reshape((1,1,1,nz))*np.array([0,0,1.]).reshape((3,1,1,1))
    #
    #
    # a = [1238.4016938000 ,
    #   186.2900499200      ,
    #    42.2511763460       ,
    #    11.6765579320        ,
    #     3.5930506482         ]
    #
    # b= [   0.8208956682, 1.4766449896, 2.1526124058,2.1081722461,0.8290999933]
    # # a=[1238]
    # # b=[0.82]
    # a = [104.8551864200     ,30.2811436880       ,10.6513942670        ,3.8699456233    ]
    #
    # b= [261.4493931636 , 176.6993802449,  76.9545600188, 17.5937671387]
    # A=time.time()
    # cube = calculate_mo.gaussian_d(r,b,a,[0,0,0])[-1]
    # B=time.time()
    # print(B-A)
    # plotter = pyvista.Plotter()
    # plot_pyvista.plot_isodensity(plotter,voxel_origin,[[3/20,0,0],[0,3/20,0],[0,0,3/20]],cube,opacity=.7)
    # C=time.time()
    # print(C-B)
    # light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    # plotter.add_light(light)
    # plotter.show(full_screen=False)
    #
    # # plt.show()