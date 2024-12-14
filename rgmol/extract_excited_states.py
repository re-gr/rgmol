#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
import numpy as np
from objects import *
from general_function import find_bonds


##########################
## Extraction functions ##
##########################


def gaussian_s(r,contraction_coefficients,exponent_primitives,r0):
    n,nx,ny,nz = np.shape(r)
    sum_gaussian = np.zeros((nx,ny,nz))

    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0
    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian += coefs[0] * np.exp(-coefs[1]*r_r)

    return sum_gaussian

def gaussian_p(r,contraction_coefficients,exponent_primitives,r0):
    n,nx,ny,nz = np.shape(r)
    sum_gaussian_x = np.zeros((nx,ny,nz))
    sum_gaussian_y = np.zeros((nx,ny,nz))
    sum_gaussian_z = np.zeros((nx,ny,nz))

    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0

    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian_x += x*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_y += y*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_z += z*coefs[0] * np.exp(-coefs[1]*r_r)

    return sum_gaussian_x


def gaussian_d(r,contraction_coefficients,exponent_primitives,r0):
    n,nx,ny,nz = np.shape(r)
    sum_gaussian_xy = np.zeros((nx,ny,nz))
    sum_gaussian_xz = np.zeros((nx,ny,nz))
    sum_gaussian_yz = np.zeros((nx,ny,nz))
    sum_gaussian_xx_yy = np.zeros((nx,ny,nz))
    sum_gaussian_zz = np.zeros((nx,ny,nz))
    x,y,z = r
    x0,y0,z0 = r0
    x,y,z = x-x0,y-y0,z-z0
    r0_reshaped = np.array(r0).reshape(3,1,1,1)

    x_x = x*x
    y_y = y*y
    r_r = x**2 + y**2 + z**2

    for coefs in zip(contraction_coefficients,exponent_primitives):
        sum_gaussian_xy += (x*y)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xz += (x*z)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_yz += (y*z)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_xx_yy += (y_y-x_x)*coefs[0] * np.exp(-coefs[1]*r_r)
        sum_gaussian_zz += (2*z*z-x_x-y_y)*coefs[0] * np.exp(-coefs[1]*r_r)

    return sum_gaussian_xy, sum_gaussian_xz, sum_gaussian_yz,sum_gaussian_xx_yy,sum_gaussian_zz



def extract_molden(file):
    """
    Extracts the global descriptors from an molden file format

    Input : file (str)

    Outputs :
    """
    flag_atoms = 0
    flag_gto = 0
    flag_mo = 0
    flag_change = 0
    atoms_names = []
    atoms_pos = []

    gaussian_coef = []
    gaussian_type = []


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
            atoms_names.append(lsplit[0])
            atoms_pos.append([float(lsplit[3]),float(lsplit[4]),float(lsplit[5])])

        elif flag_gto:
            if flag_gto_atom:#New atom
                gauss_atom = []
                gauss_atom_orbital_type = []
                flag_gto_atom = 0
                flag_gto_orb_num = 1

            elif flag_gto_orb_num:#New orbital name
                if len(line) == 1:#Empty line meaning change of atom
                    flag_gto_orb_num = 0
                    flag_gto_atom = 1
                    gaussian_coef.append(gauss_atom)
                    gaussian_type.append(gauss_atom_orbital_type)
                else:
                    gauss_orbital = []
                    gauss_atom_orbital_type.append(lsplit[0])
                    num_orb = int(lsplit[1])
                    flag_gto_orb_num = 0
                    flag_gto_orb = 1

            elif flag_gto_orb:#New orbital value
                gauss_orbital.append([float(lsplit[0]),float(lsplit[1])])
                num_orb -= 1

                if num_orb == 0:#Last value
                    gauss_atom.append(gauss_orbital)

                    flag_gto_orb = 0
                    flag_gto_orb_num = 1






        elif flag_mo:
            pass

    return atoms_names,atoms_pos,gaussian_coef