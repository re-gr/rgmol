#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script adds functions, and methods to the molecule objects.
These methods allow the plotting of chemical properties using plotly.
"""

import numpy as np
from rgmol.objects import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


########################################
## Adding Plotting Methods for Atoms  ##
########################################


def plot(self,Surfaces,plotted_property="radius",opacity=1,factor=1):
    """
    plot(Surfaces,plotted_property="radius",opacity=1,factor=1)

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
    return plot.plot_atom(Surfaces,self,plotted_property=plotted_property,opacity=opacity,factor=factor)


def plot_vector(self,Surfaces,vector,opacity=1,factor=1):
    """
    plot_vector(Surfaces,plotted_property="radius",opacity=1,factor=1)

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
    return plot.plot_vector_atom(Surfaces,self,vector,opacity=opacity,factor=factor)


atom.plot = plot
atom.plot_vector = plot_vector


############################################
## Adding Plotting Methods for Molecules  ##
############################################


def plot(self,Surfaces,plotted_property="radius",opacity=1,show_bonds=1,factor=1):
    """
    Plot the entire molecule
    """
    for atom_x in self.atoms:
        Surfaces=atom_x.plot(Surfaces,plotted_property=plotted_property,opacity=opacity,factor=factor)
    if show_bonds:
        Surfaces=plot.bonds_plotting(Surfaces,self.bonds,self.list_property("pos"),self.list_property(plotted_property),factor=factor)
    return Surfaces

def plot_vector(self,Sufaces,vector,opacity=1,factor=1):
    """
    Plot the entire molecule
    """
    for atom_x in range(len(self.atoms)):
        Surfaces=self.atoms[atom_x].plot_vector(Sufaces,vector[atom_x],opacity=opacity,factor=factor)
    return Surfaces


def plot_radius(self,opacity=1,show_bonds=1,factor=1):
    """
    Plot the entire molecule
    """
    Surfaces=[]
    for atom_x in self.atoms:
        Surfaces=atom_x.plot(Surfaces,opacity=opacity,factor=factor)
    if show_bonds:
        Surfaces=plot.bonds_plotting(Surfaces,self.bonds,self.list_property("pos"),self.list_property("radius"),factor=factor)
    fig = go.Figure(data=Surfaces)
    fig.update_layout(scene = {"xaxis": {"showticklabels":False,"title":"","showbackground":False},"yaxis": {"showticklabels":False,"title":"","showbackground":False},"zaxis": {"showticklabels":False,"title":"","showbackground":False}})

    fig.write_html("plot.html", auto_open=True)


def plot_property(self,plotted_property,opacity=1,factor=1,with_radius=1,opacity_radius=.8,factor_radius=.3):
    """
    Plot the entire molecule
    """
    X = self.properties[plotted_property]

    Surfaces = []

    if with_radius:
        Surfaces=self.plot(Surfaces,factor=factor_radius,opacity=opacity_radius)
    Surfaces=self.plot_vector(Surfaces,X,opacity=opacity,factor=factor)

    fig = go.Figure(data=Surfaces)
    fig.update_layout(scene = {"xaxis": {"showticklabels":False,"title":"","showbackground":False},"yaxis": {"showticklabels":False,"title":"","showbackground":False},"zaxis": {"showticklabels":False,"title":"","showbackground":False}})

    fig.write_html("plot.html", auto_open=True)


def plot_diagonalized_kernel(self,plotted_kernel="condensed linear response",opacity=0.5,factor=1,with_radius=1,opacity_radius=.8,factor_radius=.3):
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
            Surfaces=self.plot(Surfaces,factor=factor_radius,opacity=opacity_radius)
        Surfaces=self.plot_vector(Surfaces,XV[:,vec],opacity=opacity,factor=factor)

        for k in Surfaces:
            fig.add_trace(k,row=1,col=vec+1)

    dict_layout={"xaxis": {"showticklabels":False,"title":"","showbackground":False},"yaxis": {"showticklabels":False,"title":"","showbackground":False},"zaxis": {"showticklabels":False,"title":"","showbackground":False},"dragmode":'orbit'}
    fig["layout"]["scene"] = dict_layout
    for k in range(2,len(XV)*1+1):
        fig["layout"]["scene"+str(k)] = dict_layout


    fig.write_html("plot.html", auto_open=True)

    # fig.show()


def plot_diagonalized_condensed_kernel(self,plotted_kernel="condensed linear response",opacity=0.5,factor=1,with_radius=1,opacity_radius=1,factor_radius=.3):
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
        Surfaces=self.plot(Surfaces,factor=factor_radius,opacity=opacity_radius)
    for vec in range(len(XV)):
        Surfaces=self.plot_vector(Surfaces,XV[:,vec],opacity=opacity,factor=factor)

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

def plot_isodensity(self,plotted_isodensity="cube",opacity=0.5,factor=1,with_radius=1,opacity_radius=1,factor_radius=1):
    """
    Plot cube
    """

    Surfaces = []

    if with_radius:
        Surfaces=self.plot(Surfaces,factor=factor_radius,opacity=opacity_radius,show_bonds=False)
    Surfaces=plot.plot_cube(Surfaces,self.properties["voxel_origin"],self.properties["voxel_matrix"],self.properties["cube"],opacity=opacity,factor=factor)
    fig = go.Figure(data=Surfaces)
    fig.update_layout(scene = {"xaxis": {"showticklabels":False,"title":"","showbackground":False},"yaxis": {"showticklabels":False,"title":"","showbackground":False},"zaxis": {"showticklabels":False,"title":"","showbackground":False},"dragmode":'orbit'})

    fig.write_html("plot.html", auto_open=True)



molecule.plot = plot
molecule.plot_vector = plot_vector
molecule.plot_radius = plot_radius
molecule.plot_property = plot_property
molecule.plot_diagonalized_condensed_kernel = plot_diagonalized_condensed_kernel
molecule.plot_isodensity = plot_isodensity




##############################
## Representation functions ##
##############################


def Rz(alpha):
    """3D Rotation matrix around the z axis"""
    return np.array([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
def Ry(beta):
    """3D Rotation matrix around the y axis"""
    return np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
def Rx(gamma):
    """3D Rotation matrix around the x axis"""
    return np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])

def corr_angle(angle,x,y):
    """Corrects the angle given by arctan to be the proper angle."""
    if x>0 and y<0:
        return -angle
    if x<0 and y>0:
        return np.pi-angle
    if x<0 and y<0:
        return np.pi+angle
    return angle



def rota_bonds(Vec,x,y,z):
    """
    Roates the bonds
    """
    #Gets the two angles alpha and beta of the vector
    Vec=Vec/np.linalg.norm(Vec)
    alpha=np.arctan(np.abs(Vec[1]/Vec[0]))
    alpha=corr_angle(alpha,Vec[0],Vec[1])
    Vec2=Rz(-alpha).dot(Vec)
    beta=np.arctan(abs(Vec2[0]/Vec2[2]))
    beta=corr_angle(beta,Vec2[2],Vec2[0])
    Rota=Rz(alpha).dot(Ry(beta))#The rotation matrix to convert to a vector of the x axis to a colinear vector of Vec the starting vector
    #Rotates the bonds
    x2,y2,z2=[],[],[]
    for k in range(len(x)):
        xt,yt,zt=[],[],[]
        for j in range(len(x[0])):
            pos3=Rota.dot(np.array([x[k][j],y[k][j],z[k][j]]))
            xt.append(pos3[0])
            yt.append(pos3[1])
            zt.append(pos3[2])
        x2.append(xt)
        y2.append(yt)
        z2.append(zt)
    return np.array(x2),np.array(y2),np.array(z2)


def find_angle(Vec1,Vec2):
    """
    Finds the angle between two vectors
    """
    cross=np.cross(Vec1,Vec2) #Calculate the sign of the angle
    cross= cross*(abs(cross)>1e-5)#Remove almost zero components
    scross=np.sign(np.prod(cross[np.where(cross!=0)]))
    cosangle=(Vec1.dot(Vec2) / np.linalg.norm(Vec1)/np.linalg.norm(Vec2))#cos angle
    if cosangle>1: cosangle=1 #If the two vectors are colinear, in some cases the scalar product gives 1 + 1e-15
    if cosangle<-1: cosangle=-1
    return ( np.arccos(cosangle))*scross

def rota_mol(Pos,Vec1,Vec2):
    """
    Roates the molecule
    """

    #Find the angles of the vector we want the other vector to be colinear to
    if Vec1[0]==0:
        alpha1=np.pi/2
    else: alpha1=np.arctan(np.abs(Vec1[1]/Vec1[0]))
    alpha1=corr_angle(alpha1,Vec1[0],Vec1[1])
    Vec12=Rz(-alpha1).dot(Vec1)
    if Vec12[2]==0:
        beta1=np.pi/2
    else: beta1=np.arctan(abs(Vec12[0]/Vec12[2]))
    beta1=corr_angle(beta1,Vec12[2],Vec12[0])
    Vec13=Ry(-beta1).dot(Vec12)

    if Vec2[0]==0:
        alpha2=np.pi/2
    else: alpha2=np.arctan(np.abs(Vec2[1]/Vec2[0]))
    alpha2=corr_angle(alpha2,Vec2[0],Vec2[1])
    Vec22=Rz(-alpha2).dot(Vec2)
    if Vec22[2]==0:
        beta2=np.pi/2
    else: beta2=np.arctan(abs(Vec22[0]/Vec22[2]))
    beta2=corr_angle(beta2,Vec22[2],Vec22[0])
    Vec23=Ry(-beta2).dot(Vec22)

    #V2= Rz.Ry.v2 => v2= RyT.RzT.V2
    #=> v = Rz1.Ry1.RyT.RzT.V2
    Rota=Rz(alpha1).dot(Ry(beta1).dot(Ry(-beta2).dot(Rz(-alpha2))))
    Rota_ax=Rz(alpha1).dot(Ry(beta1))
    x2,y2,z2=[],[],[]#Rotates the bonds
    # for k in range(len(Pos)):
    #     Pos[k]=Rota.dot(Pos[k])
    Pos=Rota.dot(Pos.transpose()).transpose()
    return Pos,Rota_ax



def orthonormal_basis(Pos,B,k):
    """
    Creates an orthonormal basis centered on an atom with the "z" axis perpendicular to the surface created by the 3 atoms and the "y" axis perpendicular to the bond but on the same surface created by the 3 atoms

    Inputs :    Pos (ndarray)
                B (list)
                k (int)

    Output :
                z (ndarray dim(3)) z axis
                y (ndarray dim(3)) y axis
    """


    ind=int(B[k][1])-1
    ind2=int(B[k][0])-1


    Vec=(Pos[ind2]-Pos[ind])/np.linalg.norm((Pos[ind]-Pos[ind2]))

    #In order to create an orthonormal basis, the strategy is : taking two vectors linked to two adjacent atoms that are not colinear, one can claculate the cross product wihch gives a perpendicular vector. And by taking the cross product with the last vector and one of the two first, the result is an orthonormal basis that is centered on the atom of interest
    for j in B:

        if int(j[0])-1==ind and int(j[1])-1!=ind2:

            num=int(j[1])-1
            Dist=(Pos[ind]-Pos[num])/np.linalg.norm((Pos[ind]-Pos[num]))
            angl=find_angle(Dist,Vec)

            if angl<0: Dist=-Dist
            if abs(Dist.dot(Vec))<0.95:
                per=np.cross(Vec,Dist)
                return per,np.cross(Vec,per)

        if int(j[1])-1==ind and int(j[0])-1!=ind2:

            num=int(j[0])-1
            Dist=(Pos[ind]-Pos[num])/np.linalg.norm((Pos[ind]-Pos[num]))
            angl=find_angle(Dist,Vec)

            if angl<0: Dist=-Dist
            if abs(Dist.dot(Vec))<0.95:
                per=np.cross(Vec,Dist)
                return per,np.cross(Vec,per)


        if int(j[0])-1==ind2 and int(j[1])-1!=ind:

            num=int(j[1])-1
            Dist=(Pos[ind2]-Pos[num])/np.linalg.norm((Pos[ind2]-Pos[num]))
            angl=find_angle(Dist,Vec)

            if angl<0: Dist=-Dist
            if abs(Dist.dot(Vec))<0.95:
                per=np.cross(Vec,Dist)
                return per,np.cross(Vec,per)

        if int(j[1])-1==ind2 and int(j[0])-1!=ind:

            num=int(j[0])-1
            Dist=(Pos[ind2]-Pos[num])/np.linalg.norm((Pos[ind2]-Pos[num]))
            angl=find_angle(Dist,Vec)

            if angl<0: Dist=-Dist
            if abs(Dist.dot(Vec))<0.95:
                per=np.cross(Vec,Dist)
                return per,np.cross(Vec,per)

    #Linear => we take a random vector that is not colinear
    if Vec[1]!=0 or Vec[2]!=0:
        aVec=np.copy(Vec+np.array([1,0,0]))
    else: aVec=np.copy(Vec+np.array([0,0,1]))
    aVec/=np.linalg.norm(aVec)
    per=np.cross(Vec,aVec)
    return per,np.array([1,-1,-1.3])#linear




def bonds_plotting(Surfaces,bonds,Pos,Vec,factor=1):
    """
    Plot the bonds
    """
    #initial values for the parameters of the bonds
    Radbond=0.05
    u=np.linspace(0,2*np.pi,30)#base of the cylinder
    v=np.linspace(0,np.pi,2) #height of the cylinder

    for k in range(len(bonds)):
        one,two=int(bonds[k][0])-1,int(bonds[k][1])-1
        order = bonds[k][2]
        Vect=Pos[one]-Pos[two]

        dist=np.linalg.norm(Pos[one]-Pos[two])

        x=Radbond*(np.outer(np.cos(u),np.ones(np.size(v))))
        y=Radbond*(np.outer(np.sin(u),np.ones(np.size(v))))
        z=(np.outer(np.ones(np.size(u)),np.linspace((abs(Vec[two]*factor)-1/20),(dist-abs(Vec[one]*factor)+1/20),np.size(v))))
        x,y,z=rota_bonds(Vect,x,y,z)
        x,y,z=x+Pos[two][0],y+Pos[two][1],z+Pos[two][2]

        if order==1:
            Surfaces.append(go.Surface(x=x,y=y,z=z,colorscale="gray",showscale=False))


        elif order==1.5:
            zpe,pe=orthonormal_basis(Pos,bonds,k)#Get a orthonormal vector in order to distance the two cylinders
            pe=pe/np.linalg.norm(pe)/15
            Surfaces.append(go.Surface(x=x-pe[0],y=y-pe[1],z=z-pe[2],colorscale="gray",showscale=False))
            Surfaces.append(go.Surface(x=x+pe[0],y=y+pe[1],z=z+pe[2],colorscale="gray",opacity=.5,showscale=False))


        elif order==2:
            zpe,pe=orthonormal_basis(Pos,bonds,k)
            pe=pe/np.linalg.norm(pe)/15
            Surfaces.append(go.Surface(x=x-pe[0],y=y-pe[1],z=z-pe[2],colorscale="gray",showscale=False))
            Surfaces.append(go.Surface(x=x+pe[0],y=y+pe[1],z=z+pe[2],colorscale="gray",showscale=False))
        else:
            zpe,pe=orthonormal_basis(Pos,bonds,k)
            pe=pe/np.linalg.norm(pe)/12
            Surfaces.append(go.Surface(x=x-pe[0],y=y-pe[1],z=z-pe[2],colorscale="gray",showscale=False))
            Surfaces.append(go.Surface(x=x+pe[0],y=y+pe[1],z=z+pe[2],colorscale="white",showscale=False))
            Surfaces.append(go.Surface(x=x,y=y,z=z,colorscale="gray",showscale=False))
    return Surfaces



def convert_color_to(color):
    """convert color to plotly"""
    c0,c1,c2 = color
    c3=max(c0-.2,0)
    c4=max(c1-.2,0)
    c5=max(c2-.2,0)
    return [[0,'rgb({},{},{})'.format(c0,c1,c2)],[1,'rgb({},{},{})'.format(c3,c4,c5)]]


def plot_atom(Surfaces,atom,plotted_property="radius",opacity=1,factor=1):
    """plot atom as a sphere"""

    Norm = atom.properties[plotted_property]

    u=np.linspace(0,2*np.pi,30)
    v=np.linspace(0,np.pi,30)

    x=Norm*factor*(np.outer(np.cos(u),np.sin(v)))+atom.pos[0]
    y=Norm*factor*(np.outer(np.sin(u),np.sin(v)))+atom.pos[1]
    z=Norm*factor*(np.outer(np.ones(np.size(u)),np.cos(v)))+atom.pos[2]


    Surfaces.append(go.Surface(x=x,y=y,z=z,showscale=False,opacity=opacity,colorscale=convert_color_to(atom.color),name=atom.name))
    return Surfaces


def plot_vector_atom(Surfaces,atom,vector,opacity=1,factor=1):
    """plot atom as a sphere"""

    u=np.linspace(0,2*np.pi,30)
    v=np.linspace(0,np.pi,30)
    colors={"red":[[0,"rgb(150,30,30)"],[1,"rgb(200,0,0)"]] ,"white":[[0,"rgb(50,50,50)"],[1,"rgb(255,255,255)"]]}

    x=abs(vector)*factor*(np.outer(np.cos(u),np.sin(v)))+atom.pos[0]
    y=abs(vector)*factor*(np.outer(np.sin(u),np.sin(v)))+atom.pos[1]
    z=abs(vector)*factor*(np.outer(np.ones(np.size(u)),np.cos(v)))+atom.pos[2]
    Surfaces.append(go.Surface(x=x,y=y,z=z,colorscale=colors[["red","white"][(vector>0)*1]],opacity=opacity,showscale=False,name=atom.name))
    return Surfaces



def plot_cube(Surfaces,voxel_origin,voxel_matrix,cube,cutoff=0.2,opacity=1,factor=1):
    """plot atom as a sphere"""

    nx,ny,nz = np.shape(cube)
    voxel_end = [0,0,0]
    voxel_end[0] = voxel_origin[0] + voxel_matrix[0][0]*nx
    voxel_end[1] = voxel_origin[1] + voxel_matrix[1][1]*nx
    voxel_end[2] = voxel_origin[2] + voxel_matrix[2][2]*nx

    #Calculate cube density
    cube_density = cube**2 * voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]

    #Calculate renormalization as for some reason some cube files are not normalized
    cube_density = cube_density / np.sum(cube_density)


    array_sort = np.argsort(cube_density,axis=None)[::-1]

    cube_sorted = cube_density.flatten()[array_sort]
    cube_values_sorted = np.cumsum(cube_sorted)

    #Find how to unsort the array. There should be a more efficient way to do this
    indexes = np.arange(len(array_sort),dtype=int)
    array_unsort = np.zeros(len(array_sort),dtype=int)

    for k in range(len(array_sort)):
        array_unsort[array_sort[k]] = indexes[k]

    cube_values = cube_values_sorted[array_unsort]

    # #Multiply by the sign to keep the positive and negative aspect
    # cube_nozero = (cube + (cube==0)*1e-15).flatten()
    #
    # cube_values = cube_values * np.sign(cube_nozero)

    x = np.linspace(voxel_origin[0],voxel_end[0],nx)
    y = np.linspace(voxel_origin[1],voxel_end[1],ny)
    z = np.linspace(voxel_origin[2],voxel_end[2],nz)

    #This order is important to keep z as the inner, y as the middle and x as the outer coordinates
    y,x,z=np.meshgrid(y,x,z)
    x,y,z=x.flatten(),y.flatten(),z.flatten()


    Surfaces.append(go.Volume(x=x,y=y,z=z,value=cube_values,isomin=-1+cutoff,isomax=1-cutoff,opacity=.5,showscale=False))

    return Surfaces

