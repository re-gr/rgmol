#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script contains the definitions of the classes used in the whole package.
"""


import periodictable as pt
import numpy as np
from rgmol.dicts import *


class atom(object):
    """
    Atom object
    """

    def __init__(self,name_or_atomic_number,pos,properties={},**kwargs):
        """
        atom(name_or_atomic_number, pos, properties={}, **kwargs)

        Constructs atom object from its name or atomic number and positions.
        Properties can also be added in a properties dictionnary.
        If not specified, a color and a radius will be automatically attributed for the atom depending on its atomic number.

        Attributes
        ----------
            atomic_number : int
                The atomic number of the atom
            name : str
                The name of the atom
            pos : list
                The position of the atom
            nickname : str
                The nickname of the atom
            color : list
                The color of the atom used in the representations
            properties : list
                The list of the properties of the atom

        Parameters
        ----------
        name_or_atomic_number : int or str
            The atomic number or the name of the element
        pos : list
            The position of the atom
        properties : dict, optional
            The properties of the atom, if the radius is not provided, it will be taken from an already defined dictionnary.
        **kwargs :
            nickname : str
                The nickname of the atom
            color : list
                The color of the atom. If not provided, an already defined color will be used.

        Returns
        -------
            atoms : atom
                The atom object.


        """
        if type(name_or_atomic_number) is int:
            self.atomic_number = name_or_atomic_number
            self.name = pt.elements[name_or_atomic_number].symbol

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
            self.color=is_color
        else:
            self.color=(np.array(dict_color[self.name])/255).tolist()


        is_nickname = kwargs.get("nickname")
        if type(is_nickname) is str:
            self.nickname=is_nickname
        else:
            self.nickname=self.name

        self.properties = {"radius":dict_radii[self.name]}

        is_prop = kwargs.get("properties")
        if is_prop:
            if type(is_prop) is not dict:
                raise TypeError("The properties should be listed in a dict")
            for prop in is_prop:
                self.properties[prop] = is_prop[prop]

    def list_properties(self):
        """
        list_properties()

        Returns the list of the name of the properties of the atom

        Parameters
        ----------
            None

        Returns
        -------
            list_properties
                The list of the name of the properties
        """
        return list(self.properties.keys())





class molecule(object):
    """
    molecule(atoms,bonds,properties={},name=None)

    Constructs a molecule from atoms.

    Parameters
    ----------
        atoms : list
            List of the atoms inside the molecule
        bonds : list
            List of the bonds
        properties : dict
            Dictionnary containing the properties of the molecules. By default : {}
        name : str, optional
            Name of the molecule

    Returns
    -------
        molecule
            molecule object

    Attributes
    ----------
        name : str
        atoms : list
        bonds : list
        properties : dict
    """

    def __init__(self,atoms,bonds,properties={},name=None):
        """
        molecule(atoms,bonds,properties={},name=None)

        Constructs a molecule from atoms.

        Parameters
        ----------
            atoms : list
                List of the atoms inside the molecule
            bonds : list
                List of the bonds
            properties : dict
                Dictionnary containing the properties of the molecules. By default : {}
            name : str, optional
                Name of the molecule

        Returns
        -------
            molecule
                molecule object

        Attributes
        ----------
            name : str
            atoms : list
            bonds : list
            properties : dict
        """

        self.name = name
        self.atoms = atoms
        self.bonds = bonds

        if type(properties) is not dict:
            raise TypeError("The properties should be listed in a dict")
        self.properties = properties
        Pos = []
        for atom_x in atoms:
            Pos.append(atom_x.pos)

    def list_properties(self):
        """
        list_properties()

        Returns the list of the name of the properties of the molecule

        Parameters
        ----------
            None

        Returns
        -------
            list_properties
                The list of the name of the properties
        """
        return list(self.properties.keys())


    def list_property(self,chosen_property):
        """
        list_property(chosen_property)

        Returns the list of a chosen property for all atoms.
        A list of the properties can be obtained using the list_properties() method.
        If the chosen property is "pos", it will list the position of the atoms

        Parameters
        ----------
            chosen_property : str
                The name of the chosen property. Can be "pos" for the position

        Returns
        -------
            list_property : list
                The list of the chosen property
        """
        list_property=[]
        if chosen_property=="pos":
            for atom_x in self.atoms:
                list_property.append(atom_x.pos)
            return np.array(list_property)
        for atom_x in self.atoms:
            list_property.append(atom_x.properties[chosen_property])
        return np.array(list_property)



class group_molecules(object):
    """ group of moelcules object"""

    def __init__(self,list_molecules):
        """creates
        """
        self.molecules=list_molecules



    def plot_kernel_plt(self,plotted_kernel="condensed linear response",opacity=0.5,factor=1):
        """
        Plot kernel
        """

        nrows = len(self.molecules)

        for mol in range(len(self.molecules)):
            if len(self.molecules[mol].properties[plotted_kernel]) > ncols:
                ncols = len(self.molecules[mol].properties[plotted_kernel])


        fig=plt.figure(figsize=(3.1*ncols,2.8*nrows),dpi=200)
        for mol in range(len(self.molecules)):


            for vec in range(len(self.molecules[mol].atoms)):
                ax=fig.add_subplot(nrows,ncols,mol*nrows+1,projection="3d",aspect="equal")


    def plot_diagonalized_kernel_plt(self,plotted_kernel="condensed linear response",opacity=0.5,factor=1):
        """
        Plot kernel
        """

        nrows = len(self.molecules)
        ncols = 0
        for mol in range(len(self.molecules)):
            if len(self.molecules[mol].properties[plotted_kernel]) > ncols:
                ncols = len(self.molecules[mol].properties[plotted_kernel])

        fig=plt.figure(figsize=(3.1*ncols,2.8*nrows),dpi=200)
        plt.rcParams.update({'font.size': 13})
        plt.rcParams['svg.fonttype'] = 'none'

        #remove grid background
        plt.rcParams['axes.edgecolor'] = 'none'
        plt.rcParams['axes3d.xaxis.panecolor'] = 'none'
        plt.rcParams['axes3d.yaxis.panecolor'] = 'none'
        plt.rcParams['axes3d.zaxis.panecolor'] = 'none'

        for mol in range(len(self.molecules)):
            X = self.molecules[mol].properties[plotted_kernel]
            Xvp,XV= np.linalg.eigh(X)

            #Add box to the graph with a title
            ax=fig.add_subplot(nrow,1,nrow+1)
            if self.name:
                title = self.name
                ax.set_title("{}".format(title),fontsize=20)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['bottom'].set_color('black') #add box
            ax.spines['top'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.set_facecolor("none")



            for vec in range(len(XV)):
                ax=fig.add_subplot(nrows,ncols,mol*nrows+vec+1,projection="3d",aspect="equal")
                #Remove grid and ticks because prettier
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                self.molecules.plot_vector_plt(ax,XV,opacity=opacity,factor=factor)















