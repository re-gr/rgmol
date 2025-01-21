#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script contains the examples presented throughout the tutorial.
"""

import rgmol
import rgmol.plot_pyvista
import rgmol.examples

if __name__=="__main__":
    ##For each of these examples, just change False to True

    ## Visualization of chloromethane from an ADF output
    if False:
        file = rgmol.examples.adf_CH3Cl
        mol = rgmol.extract_adf.extract(file)
        mol.plot_radius()

    ## Visualization of a property : the dual
    if False:
        file = rgmol.examples.adf_CH3Cl
        mol = rgmol.extract_adf.extract(file)
        mol.plot_property("dual",factor=2)

    ## Visualization of a condensed kernel : condensed linear response
    if False:
        file = rgmol.examples.adf_CH3Cl
        mol = rgmol.extract_adf.extract(file)
        mol.plot_diagonalized_condensed_kernel("condensed linear response")

    ## Visualization of a cube file
    if False:
        file = rgmol.examples.cube_H2CO_MO59
        mol = rgmol.extract_cube.extract(file,do_find_bonds=1)
        mol.plot_isodensity()

    ## Visualization of Atomic Orbitals
    if False:
        file = rgmol.examples.molden_H2CO
        mol = rgmol.extract_molden.extract(file,do_find_bonds=1)
        mol.plot_AO(delta=5,grid_points=(80,80,80))

    ## Visualization of Molecular Orbitals
    if False:
        file = rgmol.examples.molden_H2CO
        mol = rgmol.extract_molden.extract(file,do_find_bonds=1)
        mol.plot_MO(delta=8,grid_points=(80,80,80))

    ## Visualization of Transition Densities
    if False:
        file = rgmol.examples.molden_CH3Cl
        mol = rgmol.extract_molden.extract(file,do_find_bonds=1)
        rgmol.extract_orca.extract_transition(rgmol.examples.orca_CH3Cl,mol=mol)
        mol.plot_transition_density(delta=5,grid_points=(60,60,60),opacity_radius=.4)


    ## Visualization of Diagonalized Non-Condensed Kernel : Linear response function
    if False:
        file = rgmol.examples.molden_H2CO
        mol = rgmol.extract_molden.extract(file,do_find_bonds=1)
        rgmol.extract_orca.extract_transition(rgmol.examples.orca_H2CO,mol=mol)
        mol.plot_diagonalized_kernel(kernel="linear_response_function",grid_points=(50,50,50),number_eigenvectors=40,delta=10,cutoff=.2)
