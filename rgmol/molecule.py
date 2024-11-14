#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import periodictable as pt
import numpy as np
from dicts import *
from plot_plt import *

class atom(object):
    """
    Atom object

    """

    def __init__(self,name_or_atomic_number,pos,**kwargs):
        """
        Constructs atom object from properties

        Parameters
        ----------
            name_or_atomic_number : int or float
            pos : list or array of 3 elements; position of the atom
            *args :
                color :
        """

        if type(name_or_atomic_number) is int:
            self.atomic_number = name_or_atomic_number
            self.name = pt.elements[6]

        elif type(name_or_atomic_number) is str:
            self.name = name_or_atomic_number
            self.atomic_number = pt.element[name_or_atomic_number].number

        else:
            raise TypeError("The name_or_atomic_number should be an int or a string")

        if type(pos) is list or type(pos) is np.ndarray:
            if len(pos) == 3:
                self.pos=np.array(pos)
            else: raise TypeError("The position should be a 3D list or array")
        else: raise TypeError("The position should be a 3D list or array")

        is_color = kwargs.get("color")
        if is_color:
            ##NEED TO ADD EXCEPTIONS AND REFORMATING
            self.color=is_color
        else: self.color=dict_color[self.name]

        is_nickname = kwargs.get("nickname")
        if type(is_nickname) is str:
            self.nickname=is_nickname
        else:
            self.nickname=name

        self.property = {"radius":dict_radii[self.name]}


    def plot_plt(self,ax,plotted_property="radius"):
        """plot the atom on the figure"""
        plot_atom(ax,self,plotted_property)






class molecule(object):
    """
    Molecule object
    """

    def __init__(self,atoms,bonds,properties={}):
        """
        Constructs a molecule from atoms

        Parameters
        ----------

        atoms : list of atoms objects
        bonds : list of
        properties : dict of properties Default : {}
        """

        self.atoms=atoms
        self.bonds=bonds
        self.properties=properties




