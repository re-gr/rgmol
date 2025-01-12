#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
# from write import *
from rgmol.objects import *
import rgmol.extract_adf
import rgmol.extract_cube
import rgmol.extract_excited_states
import rgmol.molecular_calculations

def create_newS(L,f):
    """
    Compute the condensed local softness using the Parr-Berkowitz formula : s(r,r') = -X(r,r') + f(r)f(r')/eta
    With s local softness, X the linear response, f fukui function and eta the global softness

    Input : L (ndarray)     condensed linear response kernel (2D)
            f (ndarray)     fukui function (1D) contains the square root of the global softness term

    Outputs :
            s (ndarray)     local softness
    """
    return -L+f*f.reshape((len(f),1))

__version__ = "0.1.0.10"


##


if __name__=="__main__":
    pass


