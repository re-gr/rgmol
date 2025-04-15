#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script contains the functions used to extract data from orca files.
"""


import codecs
import numpy as np
from rgmol.objects import *
from rgmol.general_function import find_bonds


##########################
## Extraction functions ##
##########################


def _extract_mol(file):
    """
    extracts position and names of atoms
    """
    flag_coords = 0
    flag_getting_coords = 0

    for line in codecs.open(file,'r',encoding="utf-8"):
        if "CARTESIAN COORDINATES (A.U.)" in line:
            Pos = []
            Names = []
            flag_coords = 2
        elif flag_coords:
            flag_coords -= 1
            if not(flag_coords):
                flag_getting_coords = 1

        elif flag_getting_coords:
            if len(line)<3:
                flag_getting_coords = 0
            else:
                lsplit = line.split()
                Pos.append([float(lsplit[5]),float(lsplit[6]),float(lsplit[7])])
                Names.append(lsplit[1])
    return Names,Pos



def extract_properties(file,mol=None):
    """
    extract_properties(file,mol=None)

    Extract the Excited states calculations from an ORCA output and adds the properties to a molecule.


    Parameters
    ----------
        file : str
        mol : molecule object, optional
            If defined, the lists will be put inside the molecule properties

    Returns
    -------
        transition_energy : list
        transition : list
        transition_factor : list
            All the lists are sorted by energy
    """

    flag_states = 0
    flag_completing_state = 0
    flag_completed_state = 0

    flag_spec = 0
    flag_spec_in = 0

    flag_excited_states = 0

    for line in codecs.open(file, 'r',encoding="utf-8"):

        if "EXCITED STATES" in line:
            #This will ensure we only keep the last excited states calculations if there is a geom opt
            flag_excited_states = 1
            flag_states = 0
            flag_completing_state = 0
            flag_completed_state = 0
            transition_factor_list = []
            transition_list = []
            transition_energy = []
            flag_states = 1

        elif flag_states:
            lsplit = line.split()
            if len(line)>1 and lsplit[0] == "STATE":
                transition_energy.append(float(lsplit[3]))

                transition = []
                transition_factor = []
                flag_completing_state = 1

            elif flag_completing_state:
                if len(line)==1:
                    flag_completing_state = 0
                    flag_completed_state = 1
                    transition_list.append(transition)
                    #renormalize factors
                    transition_factor = np.array(transition_factor)
                    transition_factor = (transition_factor / np.sum(transition_factor**2)).tolist()

                    transition_factor_list.append(transition_factor)
                else:
                    if "b" in lsplit[0]:
                        raise ValueError("Unrestricted calculations not yet implemented")
                    transition.append([ int(lsplit[0][:-1]),int(lsplit[2][:-1])])
                    fact = lsplit[-1]
                    fact = fact.replace(")","")
                    transition_factor.append([ float(fact) ])

            elif len(line)==1 and flag_completed_state:
                flag_states = 0
                flag_completed_state = 0

        elif "Center of mass = " in line:
            lsplit = line.split()
            num_1,num_2,num_3 = lsplit[-3:]
            num_1, num_2, num_3 = num_1.replace("(",""), num_2.replace("(",""), num_3.replace("(","")
            num_1, num_2, num_3 = num_1.replace(")",""), num_2.replace(")",""), num_3.replace(")","")
            num_1, num_2, num_3 = num_1.replace(",",""), num_2.replace(",",""), num_3.replace(",","")
            num_1,num_2,num_3 = float(num_1),float(num_2),float(num_3)

            center_of_mass = [num_1,num_2,num_3]

        elif "SPECTRUM VIA TRANSITION" in line:
            flag_spec = 4

            if "ABSORPTION" in line and "ELECTRIC DIPOLE" in line:
                abs_ed,abs_vd,cd_ed,cd_vd = 1,0,0,0
                D = []
                state_transition = []
                energy_transition_spectra = []
            elif "ABSORPTION" in line and "VELOCITY DIPOLE" in line:
                abs_ed,abs_vd,cd_ed,cd_vd = 0,1,0,0
                P = []
            elif "CD" in line and "ELECTRIC DIPOLE" in line:
                abs_ed,abs_vd,cd_ed,cd_vd = 0,0,1,0
                M = []
            elif "CD" in line and "VELOCITY DIPOLE" in line:
                abs_ed,abs_vd,cd_ed,cd_vd = 0,0,0,1

        elif flag_spec:
            flag_spec -=1
            if flag_spec == 0: flag_spec_in = 1

        elif flag_spec_in:
            lsplit = line.split()
            if len(line)<3: flag_spec_in = 0
            else:
                if abs_ed:
                    state_transition.append(lsplit[0]+" "+lsplit[1]+" "+lsplit[2])
                    energy_transition_spectra.append(float(lsplit[3]))
                    D.append([float(lsplit[-3]),float(lsplit[-2]),float(lsplit[-1])])
                elif abs_vd:
                    P.append([float(lsplit[-3]),float(lsplit[-2]),float(lsplit[-1])])
                elif cd_ed:
                    M.append([float(lsplit[-3]),float(lsplit[-2]),float(lsplit[-1])])

    if not(flag_excited_states): return None

    array_sort = np.argsort(transition_energy)
    transition_energy_sorted = []
    transition_list_sorted = []
    transition_factor_list_sorted = []
    for sorting in array_sort:
        transition_energy_sorted.append(transition_energy[sorting])
        transition_list_sorted.append(transition_list[sorting])
        transition_factor_list_sorted.append(transition_factor_list[sorting])

    if mol:
        mol.properties["transition_energy"] = transition_energy_sorted
        mol.properties["transition_list"] = transition_list_sorted
        mol.properties["transition_factor_list"] = transition_factor_list_sorted

        mol.properties["state_transition"] = state_transition
        mol.properties["energy_transition_spectra"] = energy_transition_spectra
        mol.properties["center_of_mass"] = center_of_mass
        mol.properties["D"] = D
        mol.properties["P"] = P
        mol.properties["M"] = M

    return transition_energy_sorted,transition_list_sorted,transition_factor_list_sorted

def extract(file,do_order_bonds=0):
    """
    extract(file,do_order_bonds=0)

    Extracts and creates a molecule from an orca output.
    If there is TD-DFT calculations, the molecule will also have the transitions and the dipole moments in its properties.

    Parameters
    ----------
        file : str
            the name of the file
        do_order_bonds : bool, optional
            If the algorithm that finds bonds tries to find the order of the bonds.

    Retruns
    -------
        mol : molecule
            The moelcule
    """

    atom_names,atom_position = _extract_mol(file)


    list_atoms = []
    nicknaming = 0
    for name,pos in zip(atom_names,atom_position):
        atom_x = atom(name,pos,nickname=str(nicknaming))
        list_atoms.append(atom_x)
        nicknaming+=1

    mol = molecule(list_atoms,[],file=file)
    extract_properties(file,mol=mol)

    mol.bonds = find_bonds(mol,do_order_bonds=do_order_bonds)

    return mol

