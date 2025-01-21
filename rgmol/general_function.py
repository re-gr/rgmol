#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script adds the function that allows to automatically find the bonds for a molecule.
WIP
"""


import numpy as np

dict_number_bonds = {
"H":	1,
"Li":	1,
"Be":	2,
"B":	3,
"C":	4,
"N":	3,
"O":	2,
"F":	1,
"Na":	1,
"Mg":	2,
"S":	2,
"Cl":	1,
"Br":   1,
"I":    1
}

dict_electronegativity = {
"H":	2.2,
"Li":	.98,
"Be":	1.57,
"B":	2.04,
"C":	2.55,
"N":	3.04,
"O":	3.44,
"F":	3.98,
"Na":	0.93,
"Mg":	1.31,
"S":	2.58,
"Cl":	3.16,
"Br":   2.96,
"I":    2.66
}


def find_atom_near(molecule,pos,factor_threshold=1.2):
    """
    function that returns the atoms that are near the one of interest
    """

    positions = []

    for atom_x in molecule.atoms:
        positions.append(atom_x.pos)

    positions = np.array(positions)
    distance = np.linalg.norm(positions - pos,axis=1)

    #remove the atom itself
    distance = distance + (distance==0.0)*1e9

    min_distance = np.min(distance)
    return distance < (min_distance * factor_threshold)


def are_bonded(bonds,atom_1,atom_2):
    for bond in range(len(bonds)):
        if bonds[bond][0] == atom_1 and bonds[bond][1] == atom_2:
            return True,bond
        if bonds[bond][0] == atom_2 and bonds[bond][1] == atom_1:
            return True,bond
    return False,0


def is_phenyl(mol,atoms_near):
    """
    Detect if is phenyl if H CC
    """
    H = 1
    C = 2
    flag = 1
    for atom_x in range(len(atoms_near)):
        if atoms_near[atom_x]:
            if mol.atoms[atom_x].name == "C":
                C -= 1
            elif mol.atoms[atom_x].name != "C" and flag:
                #Check the other atom, usually H on the phenyl but can be other. But if a fourth atom, not a phenyl
                H -= 1
                flag = 0
            else: return False
    if C==0 and H==0:
        return True
    return False



def find_bonds(mol):
    """
    Function that tries to find the bonds of a molecule
    This script is still in WIP
    """

    bonds = []
    number_bonds = [10 for k in range(len(mol.atoms))]
    list_atom_near = []
    list_is_phenyl = []

    #Find number of bonds for each atom
    for atom_x in range(len(mol.atoms)):
        if mol.atoms[atom_x].name in dict_number_bonds:
            number_bonds[atom_x] = dict_number_bonds[mol.atoms[atom_x].name]

    #First pass
    for atom_x in range(len(mol.atoms)):
        atoms_near = find_atom_near(mol,mol.atoms[atom_x].pos)
        list_atom_near.append(atoms_near)

        if is_phenyl(mol,atoms_near):
            for atom_near in range(len(atoms_near)):
                if atoms_near[atom_near] and not(are_bonded(bonds,atom_x,atom_near)[0]):
                    if mol.atoms[atom_near].name=="C":
                        bonds.append([atom_x,atom_near,1.5])
                        number_bonds[atom_x] = number_bonds[atom_x]-1.5
                        number_bonds[atom_near] =number_bonds[atom_near]- 1.5
                    else:
                        bonds.append([atom_x,atom_near,1])
                        number_bonds[atom_x] = number_bonds[atom_x]-1
                        number_bonds[atom_near] = number_bonds[atom_near]-1
        else:
            for atom_near in range(len(atoms_near)):
                if atoms_near[atom_near] and not(are_bonded(bonds,atom_x,atom_near)[0]):
                    bonds.append([atom_x,atom_near,1])
                    number_bonds[atom_x] = number_bonds[atom_x]-1
                    number_bonds[atom_near] = number_bonds[atom_near]-1
    #Second pass where the atoms that only have one neighbor are solved
    for atom_1 in range(len(number_bonds)):
        if mol.atoms[atom_1].name in dict_number_bonds:
            if number_bonds[atom_1] > 0:
                atom_can_bond = []
                for atom_2 in range(len(number_bonds)):
                    are_bond, bond_index = are_bonded(bonds,atom_1,atom_2)
                    if are_bond and number_bonds[atom_2]>0:
                        atom_can_bond.append([atom_2,bond_index])

                if len(atom_can_bond) == 1:
                    #Only one atom, has to be bonded
                    bonds[atom_can_bond[0][1]][2] += number_bonds[atom_1]
                    number_bonds[atom_can_bond[0][0]] -= number_bonds[atom_1]
                    number_bonds[atom_1] = 0

    #Third pass where the bonds that are left are divided equally
    for atom_1 in range(len(number_bonds)):
        if mol.atoms[atom_1].name in dict_number_bonds:
            if number_bonds[atom_1] > 0:
                atom_can_bond = []
                for atom_2 in range(len(number_bonds)):
                    are_bond, bond_index = are_bonded(bonds,atom_1,atom_2)
                    if are_bond and number_bonds[atom_2]>0:
                        atom_can_bond.append([atom_2,bond_index])
                if len(atom_can_bond):
                    bond_number = number_bonds[atom_1] / len(atom_can_bond)
                    number_bonds[atom_1] = 0
                    for atom_2 in range(len(atom_can_bond)):
                        bonds[atom_can_bond[atom_2][1]][2] += bond_number
                        number_bonds[atom_can_bond[atom_2][0]] -= bond_number


    for atom_1 in range(len(number_bonds)):
        if number_bonds[atom_1] < 0:
            print("The atom {} has too much bonds".format(mol.atoms[atom_1].nickname))
    #add 1 into bonds indexes
    for bond in range(len(bonds)):
        bonds[bond][0]+=1
        bonds[bond][1]+=1


    return bonds









