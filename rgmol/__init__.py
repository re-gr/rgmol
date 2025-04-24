#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rgmol
=====

This package provides visualization and computation of various chemical properties from DFT softwares.
Its main purpose is to compute and visualize the condensed and non-condensed linear response function from CDFT.

Documentation
-------------

The documentation can be found on `rgmol wiki <re-gr.github.io/rgmol_wiki>`_
"""


import numpy as np
import os
from rgmol.objects import *
import rgmol.extract_adf
import rgmol.extract_cube
import rgmol.extract_molden
import rgmol.extract_orca
import rgmol.extract_gaussian
import rgmol.molecular_calculations
import rgmol.calculate_CDFT
import rgmol.read_save
import rgmol.plot_pyvista
import rgmol.potential

__version__ = "0.1.3.1"

nprocs = 1

def set_number_procs(nprocs):
    """
    set_number_procs(nprocs)

    This functions sets the number maximum of processors that can be used by rgmol.
    Because of python shenanigans, the maximum number of processors that should be used is 8.
    Higher number of processors will slow down the calculations

    Parameters
    ----------
        nprocs : int
            the maximum number of processors

    Returns
    -------
        None
    """

    if type(nprocs) is int:
        rgmol.nprocs = nprocs
    else:
        raise TypeError('The number of processors used should be an integer. You gave a {}'.format(type(nprocs)))

set_nprocs = set_number_procs


if __name__=="__main__":
    pass


