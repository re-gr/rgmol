#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script adds methods for creating videos of vibrations
"""

import numpy as np
import scipy as sp
import pyvista
from rgmol.objects import *
import rgmol.molecular_calculations
import rgmol.grid
import rgmol.plot_pyvista._functions as rgmol_funcs




def Rz(alpha):
    """3D Rotation matrix around the z axis"""
    return np.array([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
def Ry(beta):
    """3D Rotation matrix around the y axis"""
    return np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
def Rx(gamma):
    """3D Rotation matrix around the x axis"""
    return np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])


def vibration_video(self,file_name,eddm,norm=3,nb_turns=10,nb_points=300,grid_points=(80,80,80)):
    coords, El = rgmol.rectilinear_grid_reconstruction.reconstruct_electron_density(self,grid_points=grid_points)
    coords_0 = coords

    x,y,z = coords

    dV = (x[1]-x[0])*(y[1]-y[0])*(z[1]-z[0])
    coef = self.properties["contribution_linear_response_eigenvectors"][eddm]
    # TD = rgmol.rectilinear_grid_reconstruction.reconstruct_transition_density(self,grid_points=(80,80,80),delta=5)[1]
    Eigvec = self.properties["Reconstructed_linear_response_eigenvectors"][eddm]
    E = np.array(self.properties["transition_energy"])

    Eigvec = Eigvec/np.sum(Eigvec**2*dV)**(1/2)*0.6



    plotter = pyvista.Plotter(window_size=(1300,1300))
    light = pyvista.Light((0,1,0),(0,0,0),"white",light_type="camera light",attenuation_values=(0,0,0))
    plotter.add_light(light)

    ##Perp
    def do_vid(value):
        plotter.add_text(text="",name="vid",position=(90.,20.))
        for button in plotter.button_widgets:
            repres = button.GetRepresentation()
            repres.VisibilityOff()
            button.SetRepresentation(repres)


        plotter.open_gif(file_name + ".gif",fps=30)
        p = plotter.camera.position
        plotter.off_screen = True
        plotter.enable_parallel_projection()
        path = plotter.generate_orbital_path(2.0,n_points=nb_points,viewup=[0.2,0.2,1])

        for k in range(nb_turns):
            for p,j in zip(path.points,range(len(path.points))):

                t = j+k*len(path.points)

                plotter.camera.position = p
                coss = np.cos(t*2*np.pi/(nb_points)*10)

                # print(np.shape(coss),np.shape(coef),np.shape(TD),np.shape(El))
                cube = El + Eigvec * coss * norm
                # cube = El + np.einsum("l,lijk->ijk",coef,TD) * coss

                rgmol_funcs.plot_cube(plotter,coords,cube,cutoff=0.2,opacity=0.8,factor=1,add_name="")
                plotter.write_frame()


        plotter.close()


    # cm = mol.properties["center_of_mass"]
    # coords = coords_0

    mol.plot(plotter,factor=.3,opacity=1)

    plotter.add_text(text="Video",name="vid",position=(90.,20.))
    plotter.add_checkbox_button_widget(do_vid,size=80,color_off="blue")

    plotter.show()


molecule.vibration_video = vibration_video
