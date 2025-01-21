#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script contains the functions used to extract data from molden files.
"""


import codecs
import numpy as np
from rgmol.objects import *
from rgmol.general_function import find_bonds


##########################
## Extraction functions ##
##########################


def _extract_molden_file(file):
    """
    _extract_molden_file(file)

    Extracts the data from a molden file

    Parameters
    ----------
        file : str

    Returns
    -------
        atom_names : list
        atom_position : ndarray
        AO_list : list
        AO_type_list : list
        MO_list : list
        MO_energy : list
    """

    flag_atoms = 0
    flag_gto = 0
    flag_mo = 0
    flag_change = 0
    flag_mo_lines = 4
    atom_names = []
    atom_position = []

    AO_list = []
    AO_type_list = []

    MO_list = []
    MO_energy = []
    spin = []
    occupancy = []
    D = np.arange(154)

    for line in codecs.open(file, 'r',encoding="utf-8"):
        lsplit = line.split()
        if "[Atoms]" in line:
            flag_atoms = 1
            flag_gto = 0
            flag_mo = 0
            flag_change = 1
        if "[GTO]" in line:
            flag_atoms = 0
            flag_gto = 1
            flag_mo = 0
            flag_change = 1

            flag_gto_atom = 1

        if '[MO]' in line:
            flag_atoms = 0
            flag_gto = 0
            flag_mo = 1
            flag_change = 1

        if "[5D]" in line or "[7F]" in line or "[9G]" in line :
            flag_atoms = 0
            flag_gto = 0
            flag_mo = 1
            flag_change = 1


        if flag_change:#Skip the line
            flag_change = 0

        elif flag_atoms:
            atom_names.append(lsplit[0])
            atom_position.append([float(lsplit[3]),float(lsplit[4]),float(lsplit[5])])

        elif flag_gto:
            if flag_gto_atom:#New atom
                AO_atom = []
                AO_atom_orbital_type = []
                flag_gto_atom = 0
                flag_gto_orb_num = 1

            elif flag_gto_orb_num:#New orbital name
                if len(line) <= 2:#Empty line meaning change of atom
                    flag_gto_orb_num = 0
                    flag_gto_atom = 1
                    AO_list.append(AO_atom)
                    AO_type_list.append(AO_atom_orbital_type)
                else:
                    AO_orbital = []
                    AO_atom_orbital_type.append(lsplit[0])
                    num_orb = int(lsplit[1])
                    flag_gto_orb_num = 0
                    flag_gto_orb = 1

            elif flag_gto_orb:#New orbital value
                AO_orbital.append([float(lsplit[0]),float(lsplit[1])])
                num_orb -= 1

                if num_orb == 0:#Last value
                    AO_atom.append(AO_orbital)

                    flag_gto_orb = 0
                    flag_gto_orb_num = 1

        elif flag_mo:
            if len(line) <= 2:#Last line for molecular orbitals
                MO_list.append(AO_contribution)

            #The first 4 lines contain details
            elif flag_mo_lines <= 4 and flag_mo_lines > 0:
                if lsplit[0]=="Ene=":
                    MO_energy.append(float(lsplit[1]))

                if lsplit[0] == "Spin=" and lsplit[1] == "Beta":
                    raise ValueError("Unrestricted calculations not currently implemented")
                    # spin.append(lsplit[1])
                if lsplit[0] == "Ocucp=":
                    occupancy.append(float(lsplit[1]))
                AO_contribution = []

                flag_mo_lines -= 1

            #The contribution of each AO
            else:
                if lsplit[0]=="Sym=" or lsplit[0] =="Ene=":#New MO
                    if lsplit[0]=="Ene=":
                        MO_energy.append(float(lsplit[1]))

                    if lsplit[0] == "Spin=" and lsplit[1] == "Beta":
                        raise ValueError("Unrestricted calculations not currently implemented")
                        # spin.append(lsplit[1])
                    if lsplit[0] == "Ocucp=":
                        occupancy.append(float(lsplit[1]))

                    flag_mo_lines = 3
                    MO_list.append(AO_contribution)
                else:
                    AO_contribution.append(float(lsplit[1]))

    return atom_names,atom_position,AO_list,AO_type_list,MO_list,MO_energy


def extract(file,do_find_bonds=0):
    """
    extract(file,do_find_bonds=0)

    Extract from molden input and create molecule object.
    As the bonds are not defined in a molden file, one can use do_find_bonds to
    use an algorithm that tries to find bonds. It is still in WIP

    Parameters
    ----------
        file : str
        do_find_bonds : bool, optional
            if one wants to use an algorithm to find the bonds (WIP)

    Returns
    -------
        mol : molecule
    """

    atom_names,atom_position,AO_list,AO_type_list,MO_list,MO_energy = _extract_molden_file(file)


    list_atoms = []
    nicknaming = 0
    for prop in zip(atom_names,atom_position):
        atom_x = atom(prop[0],prop[1],nickname=str(nicknaming))
        list_atoms.append(atom_x)
        nicknaming+=1

    mol = molecule(list_atoms,[],properties={"AO_list":AO_list,"AO_type_list":AO_type_list,"MO_list":MO_list,"MO_energy":MO_energy})

    if do_find_bonds:
        mol.bonds = find_bonds(mol)

    return mol


