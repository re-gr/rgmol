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

        self.properties = {"radius":dict_radii[self.name]}

        is_prop = kwargs.get("properties")
        if is_prop:
            if type(is_prop) is not dict:
                raise TypeError("The properties should be listed in a dict")
            for prop in is_prop:
                self.properties[prop] = is_prop[prop]


    def plot_plt(self,ax,plotted_property="radius",transparency=1,factor=1):
        """plot the atom on the figure"""
        plot_atom(ax,self,plotted_property=plotted_property,transparency=transparency,factor=factor)


    def plot_eigen_plt(self,ax,eigenvector,eigenvalue,transparency=1,factor=1):
        """plot the atom on the figure"""
        plot_eigen_atom(ax,self,eigenvector,eigenvalue,transparency=transparency,factor=factor)






class molecule(object):
    """
    Molecule object
    """

    def __init__(self,atoms,bonds,properties={},name=None):
        """
        Constructs a molecule from atoms

        Parameters
        ----------

        atoms : list of atoms objects
        bonds : list of
        properties : dict of properties Default : {}
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

    def list_property(self,chosen_property):
        """
        returns a list for a chosen property of all the atoms
        """
        L=[]
        if chosen_property=="pos":
            for atom_x in self.atoms:
                L.append(atom_x.pos)
            return np.array(L)
        for atom_x in self.atoms:
            L.append(atom_x.properties[chosen_property])
        return np.array(L)



    def plot_plt(self,ax,plotted_property="radius",transparency=1,show_bonds=1,factor=1):
        """
        Plot the entire molecule
        """
        for atom_x in self.atoms:
            atom_x.plot_plt(ax,plotted_property=plotted_property,transparency=transparency,factor=factor)
        if show_bonds:
            bonds_plotting(ax,self.bonds,self.list_property("pos"),self.list_property(plotted_property),factor=factor)

    def plot_vector_plt(self,ax,vector,transparency=1,factor=1):
        """
        Plot the entire molecule
        """
        for atom_x in self.atoms:
            atom_x.plot_eigen_plt(ax,vector,transparency=transparency,factor=factor)

    def plot_eigen_plt(self,ax,eigenvector,eigenvalue,transparency=1,factor=1):
        """
        Plot the entire molecule
        """
        for atom_x in self.atoms:
            atom_x.plot_eigen_plt(ax,eigenvector,eigenvalue,transparency=transparency,factor=factor)


    def plot_diagonalized_kernel_plt(self,plotted_kernel="condensed linear response",transparency=0.5,factor=1,factor_radius=.3,with_radius=1):
        """
        Plot kernel
        """
        X = self.properties[plotted_kernel]
        Xvp,XV = np.linalg.eigh(X)
        ncols = len(X)

        fig=plt.figure(figsize=(3.1*ncols,2.8),dpi=200)
        plt.rcParams.update({'font.size': 13})
        plt.rcParams['svg.fonttype'] = 'none'

        #remove grid background
        plt.rcParams['axes.edgecolor'] = 'none'
        plt.rcParams['axes3d.xaxis.panecolor'] = 'none'
        plt.rcParams['axes3d.yaxis.panecolor'] = 'none'
        plt.rcParams['axes3d.zaxis.panecolor'] = 'none'


        #Add box to the graph with a title
        ax=fig.add_subplot(1,1,1)
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
            ax=fig.add_subplot(1,ncols,vec+1,projection="3d",aspect="equal")
            #Remove grid and ticks because prettier
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            if with_radius:
                self.plot_plt(ax,factor=factor_radius)

            self.plot_eigen_plt(ax,XV[vec],Xvp,transparency=transparency,factor=factor)


class group_molecules(object):
    """ group of moelcules object"""

    def __init__(self,list_molecules):
        """creates
        """
        self.molecules=list_molecules



    def plot_kernel_plt(self,plotted_kernel="condensed linear response",transparency=0.5,factor=1):
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


    def plot_diagonalized_kernel_plt(self,plotted_kernel="condensed linear response",transparency=0.5,factor=1):
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
                self.molecules.plot_vector_plt(ax,XV,transparency=transparency,factor=factor)















