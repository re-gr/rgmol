#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rgmol
import rgmol.plot_pyvista

if __name__=="__main__":
    ##For each of these examples, just change the False to True

    ## Visualization of chloromethane from an ADF output
    if False:

        file = "output_examples//ADF//chloromethane//chloromethane.out"
        mol = rgmol.extract_adf.extract(file)
        mol.plot_radius()

    ## Visualization of a property : the dual
    if False:
        file = "output_examples//ADF//chloromethane//chloromethane.out"
        mol = rgmol.extract_adf.extract(file)
        mol.plot_property("dual",factor=2)

    ## Visualization of a condensed kernel : condensed linear response
    if False:
        file = "output_examples//ADF//chloromethane//chloromethane.out"
        mol = rgmol.extract_adf.extract(file)
        mol.plot_diagonalized_condensed_kernel("condensed linear response")

    ## Visualization of a cube file
    if False:
        file = "output_examples//Cube//H2CO.mo8a.cube"
        mol = rgmol.extract_cube.extract(file,do_find_bonds=1)
        mol.plot_isodensity()

    ## Visualization of Atomic Orbitals
    if False:
        file = "output_examples//Orca//formaldehyde//H2CO.molden.input"
        mol = rgmol.extract_excited_states.extract_molden(file,do_find_bonds=1)
        mol.plot_AO(delta=5,grid_points=(80,80,80))

    ## Visualization of Molecular Orbitals
    if False:
        file = "output_examples//Orca//formaldehyde//H2CO.molden.input"
        mol = rgmol.extract_excited_states.extract_molden(file,do_find_bonds=1)
        mol.plot_MO(delta=8,grid_points=(80,80,80))

    ## Visualization of Transition Densities
    if False:
        file = "output_examples//Orca//CH3Cl//CH3Cl.molden.input"
        mol = rgmol.extract_excited_states.extract_molden(file,do_find_bonds=1)
        rgmol.extract_excited_states.extract_transition_orca("output_examples//Orca//CH3Cl//CH3Cl.out",mol=mol)
        mol.plot_transition_density(delta=5,grid_points=(60,60,60),opacity_radius=.4)

    ## Visualization of Diagonalized Non-Condensed Kernel : Linear response function
    if True:
        file = "output_examples//Orca//formaldehyde//H2CO.molden.input"
        mol = rgmol.extract_excited_states.extract_molden(file,do_find_bonds=1)
        rgmol.extract_excited_states.extract_transition_orca("output_examples//Orca//formaldehyde//H2CO.out",mol=mol)
        mol.plot_diagonalized_kernel(kernel="linear_response_function",grid_points=(50,50,50),number_eigenvectors=40,delta=10,cutoff=.05)





        # mol.plot_MO(grid_points=(60,60,60))
