#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyvista


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
    # print(ind,ind2)
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

    #Linear => we take a random vector that is not colinear
    if Vec[1]!=0 or Vec[2]!=0:
        aVec=np.copy(Vec+np.array([1,0,0]))
    else: aVec=np.copy(Vec+np.array([0,0,1]))
    aVec/=np.linalg.norm(aVec)
    per=np.cross(Vec,aVec)
    return per,np.array([1,-1,-1.3])#linear



def bonds_plotting(plotter,bonds,Pos,Vec,factor=1):
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
            grid = pyvista.StructuredGrid(x,y,z)
            plotter.add_mesh(grid,name="bond_{}_{}".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)


        elif order==1.5:
            zpe,pe=orthonormal_basis(Pos,bonds,k)#Get a orthonormal vector in order to distance the two cylinders
            pe=pe/np.linalg.norm(pe)/15
            grid = pyvista.StructuredGrid(x-pe[0],y-pe[1],z-pe[2])
            grid2 = pyvista.StructuredGrid(x+pe[0],y+pe[1],z+pe[2])
            plotter.add_mesh(grid,name="bond_{}_{}_1".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
            plotter.add_mesh(grid2,name="bond_{}_{}_2".format(one,two),color="white",pbr=True,roughness=.2,metallic=.7,opacity=.5)



        elif order==2:
            zpe,pe=orthonormal_basis(Pos,bonds,k)
            pe=pe/np.linalg.norm(pe)/15
            grid = pyvista.StructuredGrid(x-pe[0],y-pe[1],z-pe[2])
            grid2 = pyvista.StructuredGrid(x+pe[0],y+pe[1],z+pe[2])
            plotter.add_mesh(grid,name="bond_{}_{}_1".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
            plotter.add_mesh(grid2,name="bond_{}_{}_2".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
        else:
            zpe,pe=orthonormal_basis(Pos,bonds,k)
            pe=pe/np.linalg.norm(pe)/12
            grid = pyvista.StructuredGrid(x,y,z)
            grid2 = pyvista.StructuredGrid(x-pe[0],y-pe[1],z-pe[2])
            grid3 = pyvista.StructuredGrid(x+pe[0],y+pe[1],z+pe[2])
            plotter.add_mesh(grid,name="bond_{}_{}_1".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
            plotter.add_mesh(grid2,name="bond_{}_{}_2".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
            plotter.add_mesh(grid3,name="bond_{}_{}_3".format(one,two),color="gray",pbr=True,roughness=.2,metallic=.7)
    return





def plot_atom(plotter,atom,plotted_property="radius",opacity=1,factor=1):
    """plot atom as a sphere"""
    Norm = atom.properties[plotted_property]
    atom_sphere = pyvista.Sphere(radius=Norm*factor, phi_resolution=100, theta_resolution=100,center=atom.pos)
    plotter.add_mesh(atom_sphere,name=atom.nickname,color=atom.color,pbr=False,roughness=0.0,metallic=0.0,diffuse=1,opacity=opacity)
    return


def plot_vector_atom(plotter,atom,vector,opacity=1,factor=1):
    """plot atom as a sphere"""

    colors=[[255,0,0],[255,255,255]]
    atom_sphere = pyvista.Sphere(radius=abs(vector)*factor, phi_resolution=100, theta_resolution=100,center=atom.pos)
    plotter.add_mesh(atom_sphere,name=atom.nickname+"vector",color=colors[(vector>0)*1],pbr=True,roughness=.4,metallic=.4,diffuse=1,opacity=opacity)

    return


def plot_isodensity(plotter,voxel_origin,voxel_matrix,cube,cutoff=0.2,opacity=1,factor=1):
    """plot atom as a sphere"""
    cube = np.transpose(cube,(2,1,0))
    nx,ny,nz = np.shape(cube)
    grid = pyvista.ImageData(dimensions=(ny,nx,nz),spacing=(voxel_matrix[0][0], voxel_matrix[1][1], voxel_matrix[2][2]),origin=voxel_origin)

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

    cube_values_positive = cube_values + (cube<0).flatten() * (1-cube_values)
    cube_values_negative = cube_values + (cube>0).flatten() * (1-cube_values)

    contour_positive = grid.contour(isosurfaces=2,scalars=cube_values_positive,rng=[0,1-cutoff])
    contour_negative = grid.contour(isosurfaces=2,scalars=cube_values_negative,rng=[0,1-cutoff])


    plotter.add_mesh(contour_positive,name="isosurface_cube_positive",opacity=opacity,pbr=True,roughness=.5,metallic=1,color="blue")
    plotter.add_mesh(contour_negative,name="isosurface_cube_negative",opacity=opacity,pbr=True,roughness=.5,metallic=1,color="red")

    return

