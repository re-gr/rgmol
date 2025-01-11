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



    # mol = rgmol.extract_adf.extract(file)


    # mol.plot_plt(ax,factor=.5)
    # mol.plot_radius(factor=.3)

    # mol.plot_radius_plotly(factor=.3)
    # mol.plot_diagonalized_kernel_plt(plotted_kernel="condensed linear response",opacity=0.3,factor=1)
    # fig=mol.plot_diagonalized_kernel_plotly(plotted_kernel="condensed linear response",opacity=0.3,factor=1)
    # fig=mol.plot_diagonalized_kernel_slider_plotly(plotted_kernel="condensed linear response",opacity=0.3,factor=1)
    # fig=mol.plot_property_plotly(plotted_property="dual",opacity=0.3,factor=2,opacity_radius=.7)


    # mol.plot_radius(factor=.3)
    # plotter=mol.plot_diagonalized_kernel_slider(plotted_kernel="condensed linear response",opacity=0.3,factor=2)
    # mol.plot_property(plotted_property="dual",opacity=0.3,factor=1,opacity_radius=.9)


    # file = "output_examples//formaldehyde_huge//H2CO.mo31a.cube"
    # file = "output_examples//formaldehyde_huge//H2CO.ao4.cube"
    # mol=rgmol.extract_cube.extract(file)
    # mol.plot_cube()

    # file = "output_examples//br//br.molden.input"
    # file = "output_examples//hcl//hcl.molden.input"
    # file = "output_examples//chfclbr//chfclbr.molden.input"
    # file = "output_examples//chfclbr_huge//chfclbr.molden.input"
    # file = "output_examples//chfclbr//test.molden.input"
    # file = "output_examples//CH3Br//CH3Br.molden.input"
    # file = "output_examples//CH3Cl//CH3Cl.molden.input"
    # file = "output_examples//CH3Cl//CH3Cl-TD.molden"
    # file = "output_examples//n1_CNNH2//n1_CNNH2.molden.input"
    # file = "output_examples//acrolein//acrolein.molden.input"
    file = "output_examples//formaldehyde//H2CO.molden.input"
    # file = "output_examples//formaldehyde_pbepbe_all//H2CO.molden.input"
    # file = "output_examples//form_o//H2CO.molden.input"
    # file = "output_examples//formaldehyde_huge//H2CO.molden.input"
    # file = "output_examples//H2CO//H2CO.molden.input"
    # file = "output_examples//n1_CNNH2//n1_CNNH2.molden.input"
    mol = rgmol.extract_excited_states.extract_molden(file,do_find_bonds=1)
    rgmol.extract_excited_states.extract_transition_orca("output_examples//formaldehyde//H2CO.out",mol=mol)
    # transition_energy,transition_list,transition_factor_list = rgmol.extract_excited_states.extract_transition_orca("output_examples//formaldehyde_pbepbe_all//H2CO.out")
    # transition_energy,transition_list,transition_factor_list = rgmol.extract_excited_states.extract_transition_orca("output_examples//form_o//H2CO.out")
    # rgmol.extract_excited_states.extract_transition_orca("output_examples//formaldehyde_huge//H2CO.out",mol=mol)
    # transition_energy,transition_list,transition_factor_list = rgmol.extract_excited_states.extract_transition_orca("output_examples//br//br.out")

    # transition_energy,transition_list,transition_factor_list = rgmol.extract_excited_states.extract_transition_orca("output_examples//H2CO//H2CO.out")
    # transition_energy,transition_list,transition_factor_list = rgmol.extract_excited_states.extract_transition_orca("output_examples//chfclbr//chfclbr.out")
    # transition_energy,transition_list,transition_factor_list = rgmol.extract_excited_states.extract_transition_orca("output_examples//chfclbr_huge//chfclbr.out")
    # transition_energy,transition_list,transition_factor_list = rgmol.extract_excited_states.extract_transition_orca("output_examples//CH3Cl//CH3Cl_2.out")
    # transition_list,transition_factor_list = rgmol.extract_excited_states.extract_test("output_examples//CH3Cl//CH3Cl-TD.txt")





    # mol.plot_AO(delta=5,grid_points=(60,60,60),opacity_radius=.9,opacity=.8)
    # mol.plot_MO(delta=8,grid_points=(80,80,80),opacity_radius=.9)
    # mol.plot_transition_density(delta=5,grid_points=(60,60,60),opacity_radius=.4)
    # mol.plot_diagonalized_kernel(grid_points=(25,25,25),number_eigenvectors=40,delta=5)
    mol.plot_diagonalized_kernel(kernel="linear_response_function",grid_points=(50,50,50),number_eigenvectors=40,delta=10,cutoff=.05)





    # mol.plot_MO(grid_points=(60,60,60))
