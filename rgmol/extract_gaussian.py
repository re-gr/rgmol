#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script contains the functions used to extract data from gaussian files.
THIS IS ONLY PARTIALLY IMPLEMTENTED
"""


import codecs
import numpy as np
from rgmol.objects import *
from rgmol.general_function import find_bonds


##########################
## Extraction functions ##
##########################


def _extract_fchk(file):
    """Extracts from fchk file"""

    Pos = []
    Name = []
    AO_list = []
    AO_type_list = []
    MO_list = []
    MO_energy = []
    MO_occupancy = []


    flag_name = 0
    flag_pos = 0
    flag_shell_types = 0
    flag_primitive = 0
    flag_shell_to_atom_map = 0
    flag_number_primitives = 0
    flag_contraction = 0
    flag_AO_E = 0
    flag_MO_coef = 0

    Shell_types = []
    Number_primitives = []
    Shell_to_atom_map = []
    Primitive_expo = []
    Contaction_coef = []
    MO_list_raw = []
    Contraction_coef = []

    for line in codecs.open(file, 'r',encoding="utf-8"):
        l = line.split()
        if "Atomic numbers" in line:
            flag_name = 1
        elif flag_name:
            if "Nuclear charges" in  line:
                flag_name = 0
            else:
                for atom_number in l:
                    Name.append(int(atom_number))

        elif "Number of electrons" in line:
            number_of_electrons = int(l[-1])


        elif "Current cartesian coordinates" in line:
            flag_pos = 1
            N = int(l[-1])

        elif flag_pos:
            mod = N%3
            if N>4:
                if mod == 0:
                    Pos.append(l[:3])
                    pos_mem = l[3:]
                elif mod == 1:
                    pos_mem.append(l[0])
                    Pos.append(pos_mem)
                    Pos.append(l[1:4])
                    pos_mem = [l[4]]
                else:
                    for pos in l:
                        pos_mem.append(pos)
                    Pos.append(pos_mem)
                    Pos.append(l[2:])
            else:
                if mod == 0:
                    Pos.append(l)
                elif mod == 1:
                    pos_mem.append(l[0])
                    Pos.append(pos_mem)
                    if len(l)>2:
                        Pos.append(l[1:])
                else:
                    for pos in l:
                        pos_mem.append(pos)
                    Pos.append(pos_mem)

            N-=5
            if N<1: flag_pos = 0

        elif "Shell types" in line:
            flag_shell_types = 1
        elif flag_shell_types:
            if "Number of primitives per shell" in line:
                flag_number_primitives = 1
                flag_shell_types = 0
            else:
                for shell in l:
                    Shell_types.append(shell)
        elif flag_number_primitives:
            if "Shell to atom map" in line:
                flag_shell_to_atom_map = 1
                flag_number_primitives = 0
            else:
                for num_prim in l:
                    Number_primitives.append(int(num_prim))
        elif flag_shell_to_atom_map:
            if "Primitive exponents" in line:
                flag_shell_to_atom_map = 0
                flag_primitive = 1
            else:
                for atom_map in l:
                    Shell_to_atom_map.append(int(atom_map))
        elif flag_primitive:
            if "Contraction coefficients" in line:
                flag_primitive = 0
                flag_contraction = 1
            else:
                for primitive_expo in l:
                    Primitive_expo.append(float(primitive_expo))
        elif flag_contraction:
            if "Coordinates of each shell" in line:
                flag_contraction = 0
            else:
                for contraction_coef in l:
                    Contraction_coef.append(float(contraction_coef))


        elif "Alpha Orbital Energies" in line:
            flag_AO_E = 1
        elif flag_AO_E:
            if "Alpha MO coefficients" in line:
                flag_AO_E = 0
                flag_MO_coef = 1
            else:
                for MO in l:
                    MO_energy.append(float(MO))
        elif flag_MO_coef:
            if "Orthonormal basis" in line:
                flag_MO_coef = 0
            else:
                for MO in l:
                    MO_list_raw.append(float(MO))

    mem_atom = 1
    mem_primitives = 0
    AO = []
    AO_atom = []
    AO_type = []
    convert_shell_type = {"0":"s","1":"p","-2":"d","-3":"f"}
    for atom_map,shell_type,num_primitives in zip(Shell_to_atom_map,Shell_types,Number_primitives):
        if atom_map != mem_atom:
            mem_atom = atom_map
            AO_list.append(AO_atom)
            AO_type_list.append(AO_type)
            AO_atom = []
            AO_type = []

        AO = []
        for j in range(num_primitives):
            AO.append([float(Primitive_expo[mem_primitives + j]),float(Contraction_coef[mem_primitives + j])])
        mem_primitives += num_primitives
        try:
            convert_shell_type[shell_type]
        except:
            TypeError("The type of shell {} has not been implemented".format(shell_type))

        AO_atom.append(AO)
        AO_type.append(convert_shell_type[shell_type])

    AO_list.append(AO_atom)
    AO_type_list.append(AO_type)

    s = 0



    MO_list = np.array(MO_list_raw).reshape(len(MO_energy),len(MO_energy))

    MO_occupancy = np.zeros((len(MO_energy)))
    MO_occupancy[:number_of_electrons//2] = 2

    Pos = np.array(Pos).astype("float")

    return Name, Pos, AO_list, AO_type_list, MO_list, MO_energy, MO_occupancy




def extract_transition(file,mol=None):
    """Extracts from log"""

    conversion_ev_to_ha = 1/27.2114

    flag_excited_states = 0
    flag_doing_excited_states = 0
    flag_pause = 0

    transition_energy = []
    transition_factor_list = []
    transition_list = []

    for line in codecs.open(file, 'r',encoding="utf-8"):
        l = line.split()
        if "Excitation energies and oscillator strengths" in line:
            flag_excited_states = 1
            flag_pause = 1
        elif flag_pause:
            flag_pause = 0

        elif flag_excited_states:
            if "Excited State" in line:
                flag_doing_excited_states = 1
                transition_energy.append(float(l[4])*conversion_ev_to_ha)
                transition_MO = []
                transition_MO_coef = []
            elif flag_doing_excited_states:
                if "SavETr" in line:
                    flag_excited_states = 0
                    transition_factor_list.append(transition_MO_coef)
                    transition_list.append(transition_MO)
                elif "This state" in line or len(line)<3:
                    flag_doing_excited_states = 0
                    transition_factor_list.append(transition_MO_coef)
                    transition_list.append(transition_MO)
                else:
                    transition_MO.append([int(l[0])-1,int(l[2])-1])
                    transition_MO_coef.append([float(l[-1])])

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

    return  transition_energy_sorted, transition_list_sorted, transition_factor_list_sorted



def extract_fchk(file,do_find_bonds=0):
    """extracts from fchk and creates mol"""
    if file[-4:] == "fchk":
        atom_names, atom_position, AO_list, AO_type_list, MO_list, MO_energy, MO_occupancy = _extract_fchk(file)
    list_atoms = []
    nicknaming = 1

    for prop in zip(atom_names,atom_position):
        atom_x = atom(prop[0],prop[1],nickname=str(nicknaming))
        list_atoms.append(atom_x)
        nicknaming += 1

    mol = molecule(list_atoms,[],file=file,properties={"AO_list":AO_list,"AO_type_list":AO_type_list,"MO_list":MO_list,"MO_energy":MO_energy,"MO_occupancy":MO_occupancy})

    if do_find_bonds:
        mol.bonds = find_bonds(mol)

    return mol


