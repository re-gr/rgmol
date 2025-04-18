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
# from write import *
from rgmol.objects import *
import rgmol.extract_adf
import rgmol.extract_cube
import rgmol.extract_molden
import rgmol.extract_orca
import rgmol.extract_gaussian
import rgmol.molecular_calculations
import rgmol.calculate_CDFT
import rgmol.write
import rgmol.plot_pyvista

__version__ = "0.1.2.5"


##


if __name__=="__main__":
    pass


