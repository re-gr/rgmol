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



    mol = rgmol.extract_adf.extract_all(file)


    # mol.plot_plt(ax,factor=.5)
    # mol.plot_radius_plt(factor=.3)

    # mol.plot_radius_plotly(factor=.3)
    # mol.plot_diagonalized_kernel_plt(plotted_kernel="condensed linear response",opacity=0.3,factor=1)
    # fig=mol.plot_diagonalized_kernel_plotly(plotted_kernel="condensed linear response",opacity=0.3,factor=1)
    # fig=mol.plot_diagonalized_kernel_slider_plotly(plotted_kernel="condensed linear response",opacity=0.3,factor=1)
    # fig=mol.plot_property_plotly(plotted_property="dual",opacity=0.3,factor=2,opacity_radius=.7)


    # mol.plot_radius_pyvista(factor=.3)
    mol.plot_diagonalized_kernel_slider_pyvista(plotted_kernel="condensed linear response",opacity=0.3,factor=2)
    # mol.plot_property_pyvista(plotted_property="dual",opacity=0.3,factor=1,opacity_radius=.9)
    #

    # file = "output_examples//n1_CNNH2//n1_CNNH2.mo71a_100.cube"
    # mol=rgmol.extract_cube.extract(file)
    # mol.plot_cube_pyvista()






    # plt.show()