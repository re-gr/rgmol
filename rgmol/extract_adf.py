#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

This script contains the functions used to extract data from ADF files.
"""


import codecs
import numpy as np
from rgmol.objects import *


##########################
## Extraction functions ##
##########################

def _is_CDFT(file):
    """
    Finds if the file contains CDFT
    """
    for line in codecs.open(file,"r",encoding="utf-8"):
        if  "C O N C E P T U A L  D F T" in line:
            return True
    return False



def _rename(file):
    """
    Rename the file in order to have the .out and the .run files
    """
    if ".out" in file:
        return file,file[:-3]+"run",file[:-4]
    else: return file+".out", file+".run",file



def extract_global_descriptors(file):
    """
    Extracts the global descriptors from an adf output

    Parameters
    ----------
        file : str

    Returns
    -------
        mu      (float)    electronic chemical potential
        mu+     (float)
        mu-     (float)
        chi     (float)    electronegativity (-mu)
        eta     (float)    Hardness
        S       (float)    Softness (1/eta)
        gamma   (float)    Hyperhardness
        w       (float)    Electrophilicity index
        DEn     (float)    Dissociation energy (nucleofuge) (eV)
        DEe     (float)    Dissociation energy (electrofuge) (eV)
        w-      (float)    Electrodonating power
        w+      (float)    Electroaccepting power
        NE      (float)    Net Electrophilicity
        List of the name of the descriptors
    """
    flag,flag2=0,0
    global_desc_dict={}
    L=[]
    Name=[]
    global_desc=["mu","mu+","mu-","chi","eta","S","gamma","w","DEn","DEe","w-","w+","NE"]
    for line in codecs.open(file, 'r',encoding="utf-8"):
        #Collect data for the global descriptors. Uses flags to check when the descriptors are reached
        if "GLOBAL DESCRIPTORS" in line:
            flag=1
        elif flag==1 and "." in line:
            flag=2
            a=line.split()
            if a[-1]=="(eV)":
                L.append(float(a[-2]))
                Name.append(a[:-3])
            else:
                L.append(float(a[-1]))
                Name.append(a[:-2])
        elif flag==2 and len(line)>3:
            a=line.split()
            if a[-1]=="(eV)":
                L.append(float(a[-2]))
                Name.append(a[:-3])
            else:
                if "*" in a[-1]: L.append(None)
                else: L.append(float(a[-1]))
                Name.append(a[:-2])
        elif flag==2: break

    if len(L)==0:
        raise ImportError("The file does not contain the global descriptors")
    if len(L) != len(global_desc):
        raise ImportError("The file contains more or less descriptors than expected. ADF update ?")

    for desc in range(len(L)):
        global_desc_dict[global_desc[desc]] = L[desc]

    return L,global_desc_dict,Name

def extract_condensed_kernel(file):
    """
    Extracts the condensed kernels from an adf output

    Parameters
    ----------
        file : str

    Returns
    -------
        Linear response : ndarray
            The condensed linear response kernel
        Softness Kernel : ndarray
            The condensed softness kernel
        Name : list of str
            The name of the atoms
    """
    flag,flag2=0,0
    flag_long,flag_long2=0,0
    Lin,Sof=[],[]
    Name=[]
    for line in codecs.open(file, 'r',encoding="utf-8"):
        #Collect data for the condensed kernels. Uses flags to check when the descriptors are reached
        if "CONDENSED LINEAR RESPONSE FUNCTION" in line:
            flag=1
        elif flag==1 and "1" in line:
            flag=2
            Lin.append(line.split()[3:])
            Name.append(line.split()[:2])
        elif flag==2 and len(line)>3 and flag_long>0:
            Lin[-1]+=line.split()
            flag_long-=1
        elif flag==2 and len(line)>3:
            if int(line.split()[0])>10:
                flag_long=(int(line.split()[0])-1)//10
            Lin.append(line.split()[3:])
            Name.append(line.split()[:2])
        elif flag==2: flag=3

        if "SOFTNESS KERNEL" in line:
            flag2=1
        elif flag2==1 and "1" in line:
            flag2=2
            Sof.append(line.split()[3:])
        elif flag2==2 and len(line)>3 and flag_long2>0:
            Sof[-1]+=line.split()
            flag_long2-=1
        elif flag2==2 and len(line)>3:
            if int(line.split()[0])>10:
                flag_long2=(int(line.split()[0])-1)//10
            Sof.append(line.split()[3:])
        elif flag2==2: flag2=3
    L,S=np.zeros((len(Lin[-1]),len(Lin[-1]))),np.zeros((len(Sof[-1]),len(Sof[-1])))
    for k in range(len(Sof)):
        for j in range(k+1):

            L[j][k]=float(Lin[k][j])
            L[k][j]=float(Lin[k][j])
            S[j][k]=float(Sof[k][j])
            S[k][j]=float(Sof[k][j])
    if len(Lin)==0:
        raise ImportError("The file does not contain the kernels")

    return L,S,Name

def extract_pos(file):
    """
    Extracts the positions and the radius of the nucleus from an adf output

    Parameters
    ----------
        file : str

    Returns
    -------
        Positions : ndarray
        Radius : ndarray
        Names : list of str
    """
    flag,flag2=0,0
    Pos,Rad=[],[]
    Name=[]
    for line in codecs.open(file, 'r',encoding="utf-8"):
        #Collect data for the position and the radius. Uses flags to check when the descriptors are reached
        if "Index Symbol" in line:
            flag=1
        elif flag==1 and "1" in line:
            flag=2
            Pos.append(line.split()[2:5])
            Name.append(line.split())
        elif flag==2 and len(line)>3:
            Pos.append(line.split()[2:5])
            Name.append(line.split())
        elif flag==2: flag=3
    # print(Pos)

    return np.array(Pos,dtype="float"),Name


def extract_fukui(file):
    """
    Extracts the fukui functions from an adf output

    Parameters
    ----------
        file : str

    Returns
    -------
        f+ : ndarray
        f- : ndarray
        f0 : ndarray
        f2 : ndarray
        Name : list of str
    """
    flag,flag2=0,0
    fp,fm,f0,f2=[],[],[],[]
    Name=[]
    for line in codecs.open(file, 'r',encoding="utf-8"):
        #Collect data for the fukui function. Uses flags to check when the descriptors are reached
        if "ATOMIC DESCRIPTORS: CANONICAL ENSEMBLE" in line:
            flag=1
        elif flag==1 and "1" in line:
            flag=2
            fp.append(line.split()[3])
            fm.append(line.split()[4])
            f0.append(line.split()[5])
            f2.append(line.split()[6])
            Name.append(line.split())
        elif flag==2 and not ("--" in line):
            fp.append(line.split()[3])
            fm.append(line.split()[4])
            f0.append(line.split()[5])
            f2.append(line.split()[6])
            Name.append(line.split())

        elif flag==2: break

    return np.array(fp,dtype="float"),np.array(fm,dtype="float"),np.array(f0,dtype="float"),np.array(f2,dtype="float"),Name


def extract_bonds(file):
    """
    Extracts the bonds from an adf input

    Parameters
    ----------
        file : str

    Returns
    -------
        Bonds : list
            List of bonds
        Order_bonds : list
            List of the order of bonds
    """
    flag,flag2=0,0
    list_bonds=[]

    for line in codecs.open(file, 'r',encoding="utf-8"):
        #Collect data for the bonds. Uses flags to check when the descriptors are reached
        if "BondOrders" in line:
            flag=1
        elif flag==1 and "1" in line:
            flag=2
            list_bonds.append(line.split()[:-1]+[float(line.split()[-1])])
        elif flag==2 and not ("End" in line):
            list_bonds.append(line.split()[:-1]+[float(line.split()[-1])])
        elif flag==2: flag=3
    return list_bonds


def extract(file):
    """
    extract(file)

    Extracts all the implemented information from an adf output and input

    Parameters
    ----------
        file : str

    Returns
    -------
        mol : molecule
            molecule object
    """

    fileout,filerun,file=_rename(file)
    list_atoms=[]

    pos,Name=extract_pos(fileout)
    list_bonds=extract_bonds(filerun)
    if _is_CDFT(fileout):
        L,global_desc_dict,Name=extract_global_descriptors(fileout)
        X,S,Name=extract_condensed_kernel(fileout)
        fp,fm,f0,f2,Name=extract_fukui(fileout)
        for prop in zip(Name,pos,X,S,fp,fm,f0,f2):
            dict_properties = {"condensed linear response":prop[2],"softness kernel":prop[3],"fukui plus":prop[4],"fukui minus":prop[5],"fukui":prop[6],"dual":prop[7]}
            atom_x = atom(prop[0][1],prop[1],properties=dict_properties,nickname=prop[0][0]+prop[0][1])
            list_atoms.append(atom_x)

        global_desc_dict["condensed linear response"]=X
        global_desc_dict["softness kernel"]=S
        global_desc_dict["fukui"]=f0
        global_desc_dict["dual"]=f2

        mol = molecule(list_atoms,list_bonds,properties=global_desc_dict)

    else:
        for prop in zip(Name,pos):
            atom_x = atom(prop[0],prop[1])
            list_atoms.append(atom_x)
        mol = molecule(list_atoms,list_bonds)


    return mol

