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


##########################
## Extraction functions ##
##########################

def extract_transition(file,mol=None):
    """
    extract_transition(file,mol=None)

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

    transition_factor_list = []
    transition_list = []
    transition_energy = []


    for line in codecs.open(file, 'r',encoding="utf-8"):

        if "EXCITED STATES" in line:
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
                    transition_factor.append([ float(lsplit[-1][:-1]) ])

            elif len(line)==1 and flag_completed_state:
                flag_states = 0
                flag_completed_state = 0

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

    return transition_energy_sorted,transition_list_sorted,transition_factor_list_sorted


