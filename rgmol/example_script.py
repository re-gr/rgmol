#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rgmol
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extraction, computation and graphical display of adf file')
    parser.add_argument('-i', default="output_examples//methanes//bromomethane//bromomethane.out", help='Input file of adf')
    # parser.add_argument('-i', default="output_examples//input.out", help='Input file of adf')
    # parser.add_argument('-i', default="output_examples//SP_alpha_glucose_CDFT_W_Fukui.out", help='Input file of adf')
    # parser.add_argument('-o',default="X",help="Choose do the 3d representation of a diagonalized matrix (format : matrix or matrix,vector) Matrices :  X : linear response, S* local softness, E* local hardness, f* fukui function (f(r)*f(r')/eta), XS* for the sum of X and S; with * being either : 0,+,-")
    # parser.add_argument('-s',default=0,help="Save the 3d representation in a .png (0) or a .svg (1) file (Default : -1)")
    args = parser.parse_args()

    file=args.i
    # file = "output_examples//methanes//bromomethane//bromomethane.out"



    # mol = rgmol.extract_adf.extract_all(file)
    # mol.plot_plt(ax,factor=.5)
    # mol.plot_diagonalized_kernel_plt(plotted_kernel="condensed linear response",transparency=0.3,factor=1)
    # fig=mol.plot_diagonalized_kernel_plotly(plotted_kernel="condensed linear response",transparency=0.3,factor=1)
    # fig=mol.plot_diagonalized_kernel_slider_plotly(plotted_kernel="condensed linear response",transparency=0.3,factor=1)
    # fig=mol.plot_property_plotly(plotted_property="dual",transparency=0.3,factor=2,transparency_radius=.7)
    #

    file = "output_examples//n1_CNNH2//n1_CNNH2.mo71a.cube"
    mol=rgmol.extract_cube.extract(file)
    mol.plot_cube_plotly()






    # plt.show()