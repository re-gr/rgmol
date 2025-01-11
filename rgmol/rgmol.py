#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
# from write import *
from objects import *
import extract_adf
import extract_cube
import extract_excited_states
import calculate_orbitals

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


##


if __name__=="__main__":
    pass


