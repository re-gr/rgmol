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
            **kwargs :
                color :
                prop :
        """

        if type(name_or_atomic_number) is int:
            self.atomic_number = name_or_atomic_number
            self.name = pt.elements[name_or_atomic_number]

        elif type(name_or_atomic_number) is str:
            self.name = name_or_atomic_number
            self.atomic_number = pt.elements.symbol(name_or_atomic_number).number

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
            self.nickname=self.name

        self.property = {"radius":dict_radii[self.name]}

        is_prop = kwargs.get("prop")
        if is_prop:
            if type(is_prop) is not dict:
                raise TypeError("The properties should be listed in a dict")
            for prop in is_prop:
                self.property[prop] = is_prop[prop]


    def plot_plt(self,ax,plotted_property="radius",transparency=1):
        """plot the atom on the figure"""
        plot_atom(ax,self,plotted_property,transparency=transparency)






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
        if type(properties) is not dict:
            raise TypeError("The properties should be listed in a dict")
        self.properties=properties
        Pos = []
        for atom_x in atoms:
            Pos.append(atom_x.pos)

    def list_property(self,chosen_property):
        """
        returns a list for a chosen property of all the atoms
        """
        L=[]
        for atom_x in self.atoms:
            L.append(atom_x[chosen_property])
        return L



    def plot_plt(self,ax,plotted_property="radius",transparency=1,show_bonds=1,factor=1):
        """
        Plot the entire molecule
        """
        for atom_x in self.atoms:
            atom_x.plot_plt(self,ax,plotted_property=plotted_property,transparency=transparency,factor=factor)

        if show_bonds:
            bonds_plotting(ax,bonds,self.list_property("pos"),self.list_property(plotted_property),factor=factor)





