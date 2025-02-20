#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script adds functions, and methods to the molecule objects.
These methods allow the plotting of chemical properties using pyvista.
"""

import numpy as np
import pyvista
from rgmol.objects import *
import rgmol.molecular_calculations




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
    if Vec[0]==0:
        alpha = -np.pi/2
    else:
        alpha=np.arctan(np.abs(Vec[1]/Vec[0]))
    alpha=corr_angle(alpha,Vec[0],Vec[1])
    Vec2=Rz(-alpha).dot(Vec)
    if Vec2[2] == 0:
        beta = -np.pi/2
    else:
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


def plot_cube(plotter,voxel_origin,voxel_matrix,cube,cutoff=0.1,opacity=1,factor=1,add_name=""):
    """plot atom as a sphere"""

    nx,ny,nz = np.shape(cube)
    cube_transposed = np.transpose(cube,(2,1,0))

    grid = pyvista.ImageData(dimensions=(nx,ny,nz),spacing=(voxel_matrix[0][0], voxel_matrix[1][1], voxel_matrix[2][2]),origin=voxel_origin)

    #Calculate cube density
    # cube_density = cube_transposed**2 * voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]
    cube_density = abs(cube_transposed) * voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]

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

    cube_values_positive = cube_values + (cube_transposed<0).flatten() * (1-cube_values)
    cube_values_negative = cube_values + (cube_transposed>0).flatten() * (1-cube_values)

    contour_positive = grid.contour(isosurfaces=2,scalars=cube_values_positive,rng=[0,1-cutoff])
    contour_negative = grid.contour(isosurfaces=2,scalars=cube_values_negative,rng=[0,1-cutoff])


    if len(contour_positive.point_data["Contour Data"]):
        plotter.add_mesh(contour_positive,name="isosurface_cube_positive"+add_name,opacity=opacity,pbr=True,roughness=.5,metallic=.2,color="red")
    else:
        plotter.remove_actor("isosurface_cube_positive"+add_name)
    if len(contour_negative.point_data["Contour Data"]):
        plotter.add_mesh(contour_negative,name="isosurface_cube_negative"+add_name,opacity=opacity,pbr=True,roughness=.5,metallic=.2,color="blue")
    else:
        plotter.remove_actor("isosurface_cube_negative"+add_name)



def plot_cube_multiple_isodensities(plotter,voxel_origin,voxel_matrix,cube,number_isodensities=10,opacity=1,factor=1,add_name=""):
    """plot multiple isodensities"""

    nx,ny,nz = np.shape(cube)
    cube_transposed = np.transpose(cube,(2,1,0))

    grid = pyvista.ImageData(dimensions=(nx,ny,nz),spacing=(voxel_matrix[0][0], voxel_matrix[1][1], voxel_matrix[2][2]),origin=voxel_origin)

    #Calculate cube density
    cube_density = abs(cube_transposed) * voxel_matrix[0][0] * voxel_matrix[1][1] * voxel_matrix[2][2]

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

    cube_values_positive = cube_values + (cube_transposed<0).flatten() * (1-cube_values)
    cube_values_negative = cube_values + (cube_transposed>0).flatten() * (1-cube_values)

    # for cutoff in cutoffs
    contour_positive = grid.contour(isosurfaces=number_isodensities,scalars=cube_values_positive,rng=[0,0.9])
    contour_negative = grid.contour(isosurfaces=number_isodensities,scalars=cube_values_negative,rng=[0,0.9])

    import matplotlib.colors as mpc

    empty = np.ones((256,3))
    cmap_pos = mpc.ListedColormap(empty * np.array([1.,0.,0.]*np.linspace(.5,1,256).reshape((256,1))))
    cmap_neg = mpc.ListedColormap(empty * np.array([0.,0.,1.]*np.linspace(.5,1,256).reshape((256,1))))


    point_positive = contour_positive.points
    point_negative = contour_negative.points

    if len(contour_positive.point_data["Contour Data"]):

        plotter.add_mesh(contour_positive,name="isosurface_cube_positive"+add_name,opacity=1-contour_positive["Contour Data"],scalars=contour_positive["Contour Data"],pbr=True,roughness=.5,metallic=.2,cmap=cmap_pos,show_scalar_bar=False)
    else:
        plotter.remove_actor("isosurface_cube_positive"+add_name)
    if len(contour_negative.point_data["Contour Data"]):
        plotter.add_mesh(contour_negative,name="isosurface_cube_negative"+add_name,opacity=1-contour_negative["Contour Data"],scalars=contour_negative["Contour Data"],pbr=True,roughness=.5,metallic=.2,cmap=cmap_neg,show_scalar_bar=False)
    else:
        plotter.remove_actor("isosurface_cube_negative"+add_name)



def plot_cube_volume(plotter,voxel_origin,voxel_matrix,cube,opacity=1,factor=1,add_name=""):
    """plot multiple isodensities"""

    nx,ny,nz = np.shape(cube)
    cube_transposed = np.transpose(cube,(2,1,0))

    coordinates_kept = np.zeros((nz,ny,nx),dtype="bool")
    coordinates_ravelled = np.arange(nx*ny*nz)

    cube_flatten = abs(cube_transposed).flatten()

    sorting_array = np.argsort(cube_flatten,axis=None)[::-1]
    norm_cube = np.sum(cube_flatten)
    cube_cumsum = np.cumsum(cube_flatten[sorting_array])

    coordinates_sorted = coordinates_ravelled[sorting_array]

    coordinates_ravelled_kept = coordinates_sorted[cube_cumsum < (norm_cube*0.99)]

    for coordinates in coordinates_ravelled_kept:
        coordinates_kept[np.unravel_index(coordinates,(nx,ny,nz))] = True

    coordinates = np.arange(nx*ny*nz)[coordinates_kept.flatten()]
    minx,miny,minz,maxx,maxy,maxz = 100,100,100,0,0,0
    for co in coordinates:
        x,y,z = np.unravel_index(co,(nx,ny,nz))
        minx = min(x,minx)
        maxx = max(x,maxx)
        miny = min(y,miny)
        maxy = max(y,maxy)
        minz = min(z,minz)
        maxz = max(z,maxz)

    new_nx,new_ny,new_nz = maxx-minx+1,maxy-miny+1,maxz-minz+1
    # print(new_nx,new_ny,new_nz)

    voxel_origin_0 = voxel_origin[0] + minx * voxel_matrix[0][0]
    voxel_origin_1 = voxel_origin[1] + miny * voxel_matrix[1][1]
    voxel_origin_2 = voxel_origin[2] + minz * voxel_matrix[2][2]

    grid = pyvista.ImageData(dimensions=(new_nx,new_ny,new_nz),spacing=(voxel_matrix[0][0], voxel_matrix[1][1], voxel_matrix[2][2]),origin=(voxel_origin_0,voxel_origin_1,voxel_origin_2))

    cube_data = cube_flatten[coordinates_kept.flatten()]

    cube = pyvista.Cube(x_length = voxel_matrix[0][0],y_length = voxel_matrix[1][1],z_length = voxel_matrix[2][2])
    glyphs = grid.glyph(orient=False, geom=cube, scale=False)

    scalars = abs((cube_transposed[minz:maxz+1,miny:maxy+1,minx:maxx+1]).reshape((new_nx,new_ny,new_nz,1)) * np.ones((1,1,1,6)))

    plotter.add_mesh(glyphs,scalars=scalars,opacity="linear",show_scalar_bar=False,name="volume"+add_name)





def print_contribution_transition_density(plotter,vector_number,contrib_eigenvectors,divy=1):
    """Prints the contribution of each transition density for an eigenvector"""

    plotter.add_text(text=r"Contribution of tranisiton densities",name="contrib_name",position=(0,(plotter.window_size[1]/divy-130)),font_size=18/divy)

    array_sort = np.argsort(abs(contrib_eigenvectors[vector_number-1]))[::-1]
    contrib_sorted = contrib_eigenvectors[vector_number-1][array_sort]
    contrib_indices = np.arange(1,len(contrib_eigenvectors[vector_number-1])+1)[array_sort]

    text_contrib = ""
    for contrib in range(len(contrib_sorted)):
        if abs(contrib_sorted[contrib])<0.1:
            break
        text_contrib += r"C_"+"{}".format(contrib_indices[contrib])+": {:3.3f}\n".format(contrib_sorted[contrib])
    plotter.add_text(text=text_contrib,name="contrib",font_size=14/divy,position=(20.0,plotter.window_size[1]/divy-130-16*(contrib)))



def print_occupancy(plotter,MO_occupancy,MO_number,divy=1):
    """Prints the contribution of each transition density for an eigenvector"""

    LUMO = np.argmin(MO_occupancy)

    # plotter.add_text(text=r"Occupancy : "+'{:1.1f}'.format(MO_occupancy[MO_number-1]),name="mo occupancy",position=(plotter.window_size[0]*(divx) + 20,plotter.window_size[1]*(divy)-100))
    plotter.add_text(text=r"Occupancy : "+'{:1.1f}'.format(MO_occupancy[MO_number-1]),name="mo occupancy",position=(20,plotter.window_size[1]/divy-100),font_size=18/divy)
    if MO_number-1 < LUMO:
        if MO_number-1 == LUMO-1:
            plotter.add_text(text=r"HOMO",name="mo occupancy lumo",position=(20.0,plotter.window_size[1]/divy-130),font_size=18/divy)
        else:
            plotter.add_text(text=r"HOMO - {}".format(LUMO-MO_number),name="mo occupancy lumo",position=(20.0,plotter.window_size[1]/divy-130),font_size=18/divy)
    else:
        if MO_number-1 == LUMO:
            plotter.add_text(text=r"LUMO",name="mo occupancy lumo",position=(20.0,plotter.window_size[1]/divy-130),font_size=18/divy)
        else:
            plotter.add_text(text=r"LUMO + {}".format(MO_number-1-LUMO),name="mo occupancy lumo",position=(20.0,plotter.window_size[1]/divy-130),font_size=18/divy)
