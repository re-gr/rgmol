#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import periodictable as pt
import numpy as np
from dicts import *
import plot_plt
import plot_plotly
import plot_pyvista
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyvista

class atom(object):
    """
    Atom object
    """

    def __init__(self,name_or_atomic_number,pos,**kwargs):
        """
        atom(name_or_atomic_number,pos,**kwargs)

        Constructs atom object from properties

        Parameters
        ----------
            name_or_atomic_number : int or float
                The atomic number or the name of the element
            pos : list or array_like of 3 elements
                The position of the atom

        Returns
        -------
            atom
                The atom object

        Other Parameters
        ----------------
            **kwargs :
                nickname : str
                    The nickname of the atom
                color : list or array_like of 3 elements
                    The color of the atom. If not provided, an already defined color will be used.
                properties : dict
                    The properties of the atom, if the radius is not provided, it will be taken from an already defined dictionnary.
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


    def plot_plt(self,ax,plotted_property="radius",opacity=1,factor=1):
        """
        plot_plt(ax,plotted_property="radius",opacity=1,factor=1)

        Plot a property of the atom on the ax using matplotlib

        Parameters
        ----------
            ax : matplotlib.axes
                The ax object from matplotlib on which the atom will be plotted. It can be easily defined using ax = fig.add_subplot(nrows,nncols,index)
            plotted_property : string, optional
                The property to be plotted. By default the radius is plotted.
            opacity : float, optional
                The opacity of the plot. By default equals to 1
            factor : float, optional
                The factor by which the plotted_property will be multiplied. By default equals to 1

        Returns
        -------
            None
                The atom is plotted on the ax object
        """
        plot_plt.plot_atom(ax,self,plotted_property=plotted_property,opacity=opacity,factor=factor)


    def plot_vector_plt(self,ax,vector,opacity=1,factor=1):
        """
        plot_vector_plt(ax,plotted_property="radius",opacity=1,factor=1)

        Plot a value of a vector on the position of the atom on the ax using matplotlib

        Parameters
        ----------
            ax : matplotlib.axes
                The ax object from matplotlib on which the atom will be plotted. It can be easily defined using ax = fig.add_subplot(nrows,nncols,index)
            vector : float
                The value to be plotted
            opacity : float, optional
                The opacity of the plot. By default equals to 1
            factor : float, optional
                The factor by which the vector will be multiplied. By default equals to 1

        Returns
        -------
            None
                The atom is plotted on the ax object
        """
        plot_plt.plot_vector_atom(ax,self,vector,opacity=opacity,factor=factor)


    def plot_plotly(self,Surfaces,plotted_property="radius",opacity=1,factor=1):
        """
        plot_plotly(Surfaces,plotted_property="radius",opacity=1,factor=1)

        Plot a value of a vector on the position of the atom using plotly

        Parameters
        ----------
            Surfaces : list
                A list of plotly.graph_object.surface
            plotted_property : string, optional
                The property to be plotted. By default the radius is plotted.
            opacity : float, optional
                The opacity of the plot. By default equals to 1
            factor : float, optional
                The factor by which the plotted_property will be multiplied. By default equals to 1

        Returns
        -------
            Surfaces
                The updated Surfaces list
        """
        return plot_plotly.plot_atom(Surfaces,self,plotted_property=plotted_property,opacity=opacity,factor=factor)


    def plot_vector_plotly(self,Surfaces,vector,opacity=1,factor=1):
        """
        plot_vector_plotly(Surfaces,plotted_property="radius",opacity=1,factor=1)

        Plot a value of a vector on the position of the atom using plotly

        Parameters
        ----------
            Surfaces : list
                A list of plotly.graph_object.surface
            vector : float
                The value to be plotted
            opacity : float, optional
                The opacity of the plot. By default equals to 1
            factor : float, optional
                The factor by which the vector will be multiplied. By default equals to 1

        Returns
        -------
            Surfaces
                The updated Surfaces list
        """
        return plot_plotly.plot_vector_atom(Surfaces,self,vector,opacity=opacity,factor=factor)


    def plot_pyvista(self,plotter,plotted_property="radius",opacity=1,factor=1):
        """
        plot_pyvista(plotter,plotted_property="radius",opacity=1,factor=1)

        Plot a property of the atom on the plotter using pyvista

        Parameters
        ----------
            plotter : pyvista.plotter
                The plotter object from pyvita on which the atom will be plotted. It can be easily defined using plotter = pyvista.Plotter()
            plotted_property : string, optional
                The property to be plotted. By default the radius is plotted.
            opacity : float, optional
                The opacity of the plot. By default equals to 1
            factor : float, optional
                The factor by which the plotted_property will be multiplied. By default equals to 1

        Returns
        -------
            None
                The atom is plotted on the plotter object
        """
        plot_pyvista.plot_atom(plotter,self,plotted_property=plotted_property,opacity=opacity,factor=factor)



    def plot_vector_pyvista(self,plotter,vector,opacity=1,factor=1):
        """
        plot_vector_pyvista(plotter,plotted_property="radius",opacity=1,factor=1)

        Plot a value of a vector on the position of the atom using pyvista

        Parameters
        ----------
            plotter : pyvista.plotter
                The plotter object from pyvita on which the atom will be plotted. It can be easily defined using plotter = pyvista.Plotter()
            vector : float
                The value to be plotted
            opacity : float, optional
                The opacity of the plot. By default equals to 1
            factor : float, optional
                The factor by which the plotted_property will be multiplied. By default equals to 1

        Returns
        -------
            None
                The atom is plotted on the plotter object
        """
        plot_pyvista.plot_vector_atom(plotter,self,vector,opacity=opacity,factor=factor)







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


    def plot_plt(self,ax,plotted_property="radius",opacity=1,show_bonds=1,factor=1):
        """
        Plot the entire molecule
        """
        for atom_x in self.atoms:
            atom_x.plot_plt(ax,plotted_property=plotted_property,opacity=opacity,factor=factor)
        if show_bonds:
            plot_plt.bonds_plotting(ax,self.bonds,self.list_property("pos"),self.list_property(plotted_property),factor=factor)

    def plot_vector_plt(self,ax,vector,opacity=1,factor=1):
        """
        Plot the entire molecule
        """
        for atom_x in range(len(self.atoms)):
            self.atoms[atom_x].plot_vector_plt(ax,vector[atom_x],opacity=opacity,factor=factor)


    def plot_radius_plt(self,opacity=1,show_bonds=1,factor=1):
        """
        Plot kernel
        """


        fig=plt.figure(figsize=(3.1,2.8),dpi=200)
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

        ax=fig.add_subplot(1,1,1,projection="3d",aspect="equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        for atom_x in self.atoms:
            atom_x.plot_plt(ax,opacity=opacity,factor=factor)
        if show_bonds:
            plot_plt.bonds_plotting(ax,self.bonds,self.list_property("pos"),self.list_property("radius"),factor=factor)
        plot_plt.axes_equal(ax)
        plt.show()


    def plot_diagonalized_kernel_plt(self,plotted_kernel="condensed linear response",opacity=0.5,factor=1,factor_radius=.3,with_radius=1):
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
            ax.set_title(r"$\mathrm{\lambda}$"+" = {:3.2f}".format(Xvp[vec]),y=1.0,pad=-6)
            if with_radius:
                self.plot_plt(ax,factor=factor_radius,opacity=0.8)
            self.plot_vector_plt(ax,XV[:,vec],opacity=opacity,factor=factor)
            plot_plt.axes_equal(ax)
        plt.show()



    def plot_plotly(self,Surfaces,plotted_property="radius",opacity=1,show_bonds=1,factor=1):
        """
        Plot the entire molecule
        """
        for atom_x in self.atoms:
            Surfaces=atom_x.plot_plotly(Surfaces,plotted_property=plotted_property,opacity=opacity,factor=factor)
        if show_bonds:
            Surfaces=plot_plotly.bonds_plotting(Surfaces,self.bonds,self.list_property("pos"),self.list_property(plotted_property),factor=factor)
        return Surfaces

    def plot_vector_plotly(self,Sufaces,vector,opacity=1,factor=1):
        """
        Plot the entire molecule
        """
        for atom_x in range(len(self.atoms)):
            Surfaces=self.atoms[atom_x].plot_vector_plotly(Sufaces,vector[atom_x],opacity=opacity,factor=factor)
        return Surfaces


    def plot_radius_plotly(self,opacity=1,show_bonds=1,factor=1):
        """
        Plot the entire molecule
        """
        Surfaces=[]
        for atom_x in self.atoms:
            Surfaces=atom_x.plot_plotly(Surfaces,opacity=opacity,factor=factor)
        if show_bonds:
            Surfaces=plot_plotly.bonds_plotting(Surfaces,self.bonds,self.list_property("pos"),self.list_property("radius"),factor=factor)
        fig = go.Figure(data=Surfaces)
        fig.update_layout(scene = {"xaxis": {"showticklabels":False,"title":"","showbackground":False},"yaxis": {"showticklabels":False,"title":"","showbackground":False},"zaxis": {"showticklabels":False,"title":"","showbackground":False}})

        fig.write_html("plot.html", auto_open=True)


    def plot_property_plotly(self,plotted_property,opacity=1,factor=1,with_radius=1,opacity_radius=.8,factor_radius=.3):
        """
        Plot the entire molecule
        """
        X = self.properties[plotted_property]

        Surfaces = []

        if with_radius:
            Surfaces=self.plot_plotly(Surfaces,factor=factor_radius,opacity=opacity_radius)
        Surfaces=self.plot_vector_plotly(Surfaces,X,opacity=opacity,factor=factor)

        fig = go.Figure(data=Surfaces)
        fig.update_layout(scene = {"xaxis": {"showticklabels":False,"title":"","showbackground":False},"yaxis": {"showticklabels":False,"title":"","showbackground":False},"zaxis": {"showticklabels":False,"title":"","showbackground":False}})

        fig.write_html("plot.html", auto_open=True)


    def plot_diagonalized_kernel_plotly(self,plotted_kernel="condensed linear response",opacity=0.5,factor=1,with_radius=1,opacity_radius=.8,factor_radius=.3):
        """
        Plot kernel
        """
        X = self.properties[plotted_kernel]
        Xvp,XV = np.linalg.eigh(X)
        ncols = len(X)

        scene_dict = {"type":"scene"}


        titles_subplots = []
        for vec in range(len(XV)):
            titles_subplots.append(r"$\mathrm{\lambda}"+"= {:3.2f}$".format(Xvp[vec]))

        fig = make_subplots(rows=1,cols=len(XV),specs=[[scene_dict for k in range(len(XV))] for j in range(1)],subplot_titles=titles_subplots)


        for vec in range(len(XV)):
            Surfaces = []

            if with_radius:
                Surfaces=self.plot_plotly(Surfaces,factor=factor_radius,opacity=opacity_radius)
            Surfaces=self.plot_vector_plotly(Surfaces,XV[:,vec],opacity=opacity,factor=factor)

            for k in Surfaces:
                fig.add_trace(k,row=1,col=vec+1)

        dict_layout={"xaxis": {"showticklabels":False,"title":"","showbackground":False},"yaxis": {"showticklabels":False,"title":"","showbackground":False},"zaxis": {"showticklabels":False,"title":"","showbackground":False},"dragmode":'orbit'}
        fig["layout"]["scene"] = dict_layout
        for k in range(2,len(XV)*1+1):
            fig["layout"]["scene"+str(k)] = dict_layout


        fig.write_html("plot.html", auto_open=True)

        # fig.show()


    def plot_diagonalized_kernel_slider_plotly(self,plotted_kernel="condensed linear response",opacity=0.5,factor=1,with_radius=1,opacity_radius=1,factor_radius=.3):
        """
        Plot kernel
        """
        X = self.properties[plotted_kernel]
        Xvp,XV = np.linalg.eigh(X)
        ncols = len(X)

        number_items = len(self.atoms) + len(self.bonds)
        number_vectors = len(self.atoms)*len(X)
        number_atoms = len(self.atoms)

        Surfaces = []
        if with_radius:
            Surfaces=self.plot_plotly(Surfaces,factor=factor_radius,opacity=opacity_radius)
        for vec in range(len(XV)):
            Surfaces=self.plot_vector_plotly(Surfaces,XV[:,vec],opacity=opacity,factor=factor)

        fig = go.Figure(data=Surfaces)
        fig.update_traces(visible=False)
        fig.update_layout(scene = {"xaxis": {"showticklabels":False,"title":"","showbackground":False},"yaxis": {"showticklabels":False,"title":"","showbackground":False},"zaxis": {"showticklabels":False,"title":"","showbackground":False},"dragmode":'orbit'})
        #Toggle the first eigenvector visible
        for j in range(number_items+len(self.atoms)):
            fig["data"][j]["visible"] = True
        steps = []
        for i in range(len(X)):
            step = dict(method="update",args=[{"visible": [True]*number_items + [False] * (number_vectors)}])
            for j in range(number_atoms):
                step["args"][0]["visible"][i*number_atoms+j+number_items] = True  # Toggle i'th trace to "visible"
            steps.append(step)
        sliders = [dict(active=0,currentvalue={"prefix": "Eigenvector: "},pad={"t": 1},steps=steps)]
        fig.update_layout(sliders=sliders)
        fig.write_html("plot.html", auto_open=True)

    def plot_cube_plotly(self,plotted_isodensity="cube",opacity=0.5,factor=1,with_radius=1,opacity_radius=1,factor_radius=1):
        """
        Plot cube
        """

        Surfaces = []

        if with_radius:
            Surfaces=self.plot_plotly(Surfaces,factor=factor_radius,opacity=opacity_radius,show_bonds=False)
        Surfaces=plot_plotly.plot_isodensity(Surfaces,self.properties["voxel_origin"],self.properties["voxel_matrix"],self.properties["cube"],opacity=opacity,factor=factor)
        fig = go.Figure(data=Surfaces)
        fig.update_layout(scene = {"xaxis": {"showticklabels":False,"title":"","showbackground":False},"yaxis": {"showticklabels":False,"title":"","showbackground":False},"zaxis": {"showticklabels":False,"title":"","showbackground":False},"dragmode":'orbit'})

        fig.write_html("plot.html", auto_open=True)





    def plot_pyvista(self,plotter,plotted_property="radius",opacity=1,show_bonds=1,factor=1):
        """
        Plot the entire molecule
        """
        for atom_x in self.atoms:
            atom_x.plot_pyvista(plotter,plotted_property=plotted_property,opacity=opacity,factor=factor)
        if show_bonds:
            plot_pyvista.bonds_plotting(plotter,self.bonds,self.list_property("pos"),self.list_property(plotted_property),factor=factor)
        return

    def plot_vector_pyvista(self,plotter,vector,opacity=1,factor=1):
        """
        Plot the entire molecule
        """
        for atom_x in range(len(self.atoms)):
            self.atoms[atom_x].plot_vector_pyvista(plotter,vector[atom_x],opacity=opacity,factor=factor)
        return


    def plot_radius_pyvista(self,opacity=1,show_bonds=1,factor=1):
        """
        Plot the entire molecule
        """
        plotter = pyvista.Plotter()
        for atom_x in self.atoms:
            atom_x.plot_pyvista(plotter,opacity=opacity,factor=factor)
        if show_bonds:
            plot_pyvista.bonds_plotting(plotter,self.bonds,self.list_property("pos"),self.list_property("radius"),factor=factor)
        light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
        plotter.add_light(light)
        plotter.show(full_screen=True)


    def plot_property_pyvista(self,plotted_property,opacity=1,factor=1,with_radius=1,opacity_radius=.8,factor_radius=.3):
        """
        Plot the entire molecule
        """
        X = self.properties[plotted_property]
        plotter = pyvista.Plotter()

        if with_radius:
            self.plot_pyvista(plotter,factor=factor_radius,opacity=opacity_radius)
        self.plot_vector_pyvista(plotter,X,opacity=opacity,factor=factor)
        light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
        plotter.add_light(light)
        plotter.show(full_screen=False)

    def plot_diagonalized_kernel_slider_pyvista(self,plotted_kernel="condensed linear response",opacity=0.5,factor=1,with_radius=1,opacity_radius=1,factor_radius=.3):
        """
        Plot kernel
        """
        X = self.properties[plotted_kernel]
        Xvp,XV = np.linalg.eigh(X)
        ncols = len(X)

        plotter = pyvista.Plotter()
        if with_radius:
            self.plot_pyvista(plotter,factor=factor_radius,opacity=opacity_radius)
        def create_mesh_diagonalized_kernel(value):
            vector_number = int(round(value))
            self.plot_vector_pyvista(plotter,XV[:,vector_number],opacity=opacity,factor=factor)

        # light = pyvista.Light((0,10,0),(0,0,0),"white",light_type="camera light",intensity=.3)
        # plotter.add_light(light)
        plotter.add_slider_widget(create_mesh_diagonalized_kernel, [0, len(XV)-1],value=0,title="Eigenvector", fmt="%1.0f")

        plotter.show(full_screen=False)


    def plot_cube_pyvista(self,plotted_isodensity="cube",opacity=0.5,factor=1,with_radius=1,opacity_radius=1,factor_radius=.5):
        """
        Plot cube
        """
        plotter = pyvista.Plotter()
        if with_radius:
            self.plot_pyvista(plotter,factor=factor_radius,opacity=opacity_radius,show_bonds=True)
        plot_pyvista.plot_isodensity(plotter,self.properties["voxel_origin"],self.properties["voxel_matrix"],self.properties["cube"],opacity=opacity,factor=factor)

        light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
        plotter.add_light(light)
        plotter.show(full_screen=False)





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















