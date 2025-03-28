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

    flag_spec = 0
    flag_spec_in = 0

    for line in codecs.open(file, 'r',encoding="utf-8"):

        if "EXCITED STATES" in line:
            #This will ensure we only keep the last excited states calculations if there is a geom opt
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
                    transition_factor.append([ float(lsplit[-1][:-1]) ])

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
