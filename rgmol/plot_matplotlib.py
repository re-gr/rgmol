#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from objects import *


########################################
## Adding Plotting Methods for Atoms  ##
########################################


def plot(self,ax,plotted_property="radius",opacity=1,factor=1):
    """
    plot(ax,plotted_property="radius",opacity=1,factor=1)

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
    plot.plot_atom(ax,self,plotted_property=plotted_property,opacity=opacity,factor=factor)


def plot_vector(self,ax,vector,opacity=1,factor=1):
    """
    plot_vector(ax,plotted_property="radius",opacity=1,factor=1)

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
    plot.plot_vector_atom(ax,self,vector,opacity=opacity,factor=factor)


atom.plot = plot
atom.plot_vector = plot_vector


############################################
## Adding Plotting Methods for Molecules  ##
############################################




def plot(self,ax,plotted_property="radius",opacity=1,show_bonds=1,factor=1):
    """
    Plot the entire molecule
    """
    for atom_x in self.atoms:
        atom_x.plot(ax,plotted_property=plotted_property,opacity=opacity,factor=factor)
    if show_bonds:
        plot.bonds_plotting(ax,self.bonds,self.list_property("pos"),self.list_property(plotted_property),factor=factor)

def plot_vector(self,ax,vector,opacity=1,factor=1):
    """
    Plot the entire molecule
    """
    for atom_x in range(len(self.atoms)):
        self.atoms[atom_x].plot_vector(ax,vector[atom_x],opacity=opacity,factor=factor)


def plot_radius(self,opacity=1,show_bonds=1,factor=1):
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
        atom_x.plot(ax,opacity=opacity,factor=factor)
    if show_bonds:
        plot.bonds_plotting(ax,self.bonds,self.list_property("pos"),self.list_property("radius"),factor=factor)
    plot.axes_equal(ax)
    plt.show()


def plot_diagonalized_condensed_kernel(self,plotted_kernel="condensed linear response",opacity=0.5,factor=1,factor_radius=.3,with_radius=1):
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
            self.plot(ax,factor=factor_radius,opacity=0.8)
        self.plot_vector(ax,XV[:,vec],opacity=opacity,factor=factor)
        plot.axes_equal(ax)
    plt.show()



molecule.plot = plot
molecule.plot_vector = plot_vector
molecule.plot_radius = plot_radius
molecule.plot_diagonalized_condensed_kernel = plot_diagonalized_condensed_kernel


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



def axes_equal(ax):
    """
    Makes axes equal
    """

    boundaries = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()])
    ranges = np.diff(boundaries,axis=1)
    middles = np.sum(boundaries,axis=1)/2
    max_range = np.max(ranges)/2

    ax.set_xlim3d([middles[0]-max_range,middles[0]+max_range])
    ax.set_ylim3d([middles[1]-max_range,middles[1]+max_range])
    ax.set_zlim3d([middles[2]-max_range,middles[2]+max_range])





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




def transf_pos(Pos,B,Ord,Rota_arb,row,Scale):
    """
    Do the rotation of the atoms in order to have a good representation of the molecule
    """

    Pos[:,0]-=np.mean(Pos[:,0]) #Center the Pos
    Pos[:,1]-=np.mean(Pos[:,1])
    Pos[:,2]-=np.mean(Pos[:,2])
    #rotates the molecule to be perpendicular to the default projection
    Zmean=np.array([0.,0.,0.])
    Zdefault=np.array([1.,-1.,1.])#The default projection axis from matplotlib
    if len(Ord)!=0: #If there are bonds, rotate the molecule
        # Mord=np.max(Ord)
        # Mord=2
        for k in range(len(B)):
            # if Ord[k]==Mord:
            zpe,ype=orthonormal_basis(Pos,B,k)
            Zmean+=zpe
        Zmean=Zmean/np.linalg.norm(Zmean)
        Zdefault=Zdefault/np.linalg.norm(Zdefault)

        Pos,Rota=rota_mol(Pos,Zdefault,Zmean)
        for k in range(len(Pos)):
            Pos[k]=Rota.transpose().dot(Pos[k])#Rotation to xyz
        #Scale the molecule ONLY USE 1 or -1 TO MIRROR
        if Scale!=[]:
            Pos[:,0]=Scale[row][0]*Pos[:,0]
            Pos[:,1]=Scale[row][1]*Pos[:,1]
            Pos[:,2]=Scale[row][2]*Pos[:,2]
    for k in range(len(Pos)):
        Pos[k]=Rz(Rota_arb).dot(Pos[k]) #Rotation around the z axis
    return Pos


def bonds_plotting(ax,bonds,Pos,Vec,factor=1):
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
        if order==1: ax.plot_surface(x,y,z,color="gray")

        elif order==1.5:
            zpe,pe=orthonormal_basis(Pos,bonds,k)#Get a orthonormal vector in order to distance the two cylinders
            pe=pe/np.linalg.norm(pe)/15
            ax.plot_surface(x-pe[0],y-pe[1],z-pe[2],color="gray")
            ax.plot_surface(x+pe[0],y+pe[1],z+pe[2],color="white")

        elif order==2:
            zpe,pe=orthonormal_basis(Pos,bonds,k)
            pe=pe/np.linalg.norm(pe)/15
            ax.plot_surface(x-pe[0],y-pe[1],z-pe[2],color="gray")
            ax.plot_surface(x+pe[0],y+pe[1],z+pe[2],color="gray")

        else:
            zpe,pe=orthonormal_basis(Pos,bonds,k)
            pe=pe/np.linalg.norm(pe)/12
            ax.plot_surface(x-pe[0],y-pe[1],z-pe[2],color="gray")
            ax.plot_surface(x+pe[0],y+pe[1],z+pe[2],color="gray")
            ax.plot_surface(x,y,z,color="gray")



def plot_atom(ax,atom,plotted_property="radius",opacity=1,factor=1):
    """plot atom as a sphere"""

    Norm = atom.properties[plotted_property]
    Pos = atom.pos

    u=np.linspace(0,2*np.pi,20)
    v=np.linspace(0,np.pi,15)

    x=Norm*factor*(np.outer(np.cos(u),np.sin(v)))+Pos[0]
    y=Norm*factor*(np.outer(np.sin(u),np.sin(v)))+Pos[1]
    z=Norm*factor*(np.outer(np.ones(np.size(u)),np.cos(v)))+Pos[2]
    ax.plot_surface(x,y,z,color=atom.color,alpha=opacity)


def plot_vector_atom(ax,atom,vector,opacity=1,factor=1):
    """plot atom as a sphere"""


    u=np.linspace(0,2*np.pi,10)
    v=np.linspace(0,np.pi,15)
    colors=["r","w"]

    if vector>0:
        col=colors[0]
    else: col=colors[1]
    x=abs(vector)*factor*(np.outer(np.cos(u),np.sin(v)))+atom.pos[0]
    y=abs(vector)*factor*(np.outer(np.sin(u),np.sin(v)))+atom.pos[1]
    z=abs(vector)*factor*(np.outer(np.ones(np.size(u)),np.cos(v)))+atom.pos[2]
    ax.plot_surface(x,y,z,color=col,alpha=opacity)




