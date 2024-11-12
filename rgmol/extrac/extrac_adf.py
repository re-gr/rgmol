#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
import numpy as np



##########################
## Extraction functions ##
##########################

def glob_desc(file):
    """
    Extracts the global descriptors from an adf output

    Input : file (str)

    Outputs :   List of descriptors :
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
    L=[]
    Name=[]
    for line in codecs.open(file, 'r',encoding="utf-8"):
        #Collect data for the global descriptors. Uses flags to check when the descriptors are reached
        if "GLOBAL DESCRIPTORS" in line:
            flag=1
        elif flag==1 and "1" in line:
            flag=2
            a=line.split()
            if a[-1]=="(eV)":
                L.append(float(a[-2]))
                Name.append(a[:-3])
            else:
                L.append(float(a[-1]))
                Name.append(a[:-2])
        elif flag==2 and len(line)!=1:
            a=line.split()
            if a[-1]=="(eV)":
                L.append(float(a[-2]))
                Name.append(a[:-3])
            else:
                if "*" in a[-1]: L.append(None)
                else: L.append(float(a[-1]))
                Name.append(a[:-2])
        elif flag==2: flag=3
    return L,Name

def ker(file):
    """
    Extracts the kernel from an adf output

    Input : file (str)

    Outputs :   Linear response (ndarray)
                Softness Kernel (ndarray)
                Name of the atoms (list of str)
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
        elif flag==2 and len(line)!=1 and flag_long>0:
            Lin[-1]+=line.split()
            flag_long-=1
        elif flag==2 and len(line)!=1:
            if int(line.split()[0])>10:
                flag_long=int(line.split()[0])//10
            Lin.append(line.split()[3:])
            Name.append(line.split()[:2])
        elif flag==2: flag=3

        if "SOFTNESS KERNEL" in line:
            flag2=1
        elif flag2==1 and "1" in line:
            flag2=2
            Sof.append(line.split()[3:])
        elif flag2==2 and len(line)!=1 and flag_long2>0:
            Sof[-1]+=line.split()
            flag_long2-=1
        elif flag2==2 and len(line)!=1:
            if int(line.split()[0])>10:
                flag_long2=int(line.split()[0])//10
            Sof.append(line.split()[3:])
        elif flag2==2: flag2=3
    L,S=np.zeros((len(Lin[-1]),len(Lin[-1]))),np.zeros((len(Sof[-1]),len(Sof[-1])))
    for k in range(len(Sof)):
        for j in range(k+1):

            L[j][k]=float(Lin[k][j])
            L[k][j]=float(Lin[k][j])
            S[j][k]=float(Sof[k][j])
            S[k][j]=float(Sof[k][j])
    return L,S,Name

def pos_rad(file):
    """
    Extracts the positions and the radius of the nucleus from an adf output

    Input : file (str)

    Outputs :   Positions (ndarray, 2dim)
                Radius (ndarray, 2dim)
                Name of the atoms (list of str)
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
            Pos.append(line.split()[2:])
            Name.append(line.split()[:2])
        elif flag==2 and len(line)!=2:
            Pos.append(line.split()[2:])
            Name.append(line.split()[:2])
        elif flag==2: flag=3

        if "Electron Density at Nuclei" in line:
            flag2=1
        elif flag2==1 and "1" in line:
            flag2=2
            Rad.append(line.split()[2])
        elif flag2==2 and len(line)!=1:
            Rad.append(line.split()[2])
        elif flag2==2: flag2=3
    return np.array(Pos,dtype="float"),np.array(Rad,dtype="float"),Name


def fukui(file,eta=1):
    """
    Extracts the fukui functions from an adf output

    Input :     file    (str)
                eta     (float,default=1)    square of the scaling factor used to do the comparison using the Parr-Berkowitz formula

    Outputs :   f+      (ndarray)
                f-      (ndarray)
                f0      (ndarray)
                f2      (ndarray)
                Name of the atoms (list of str)
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
            Name.append(line.split()[:2])
        elif flag==2 and not ("--" in line):
            fp.append(line.split()[3])
            fm.append(line.split()[4])
            f0.append(line.split()[5])
            f2.append(line.split()[6])
            Name.append(line.split()[:2])
        elif flag==2: flag=3
    et=eta**(1/2)
    return np.array(fp,dtype="float")/et,np.array(fm,dtype="float")/et,np.array(f0,dtype="float")/et,np.array(f2,dtype="float")/et,Name

def bonds(file):
    """
    Extracts the bonds from an adf input

    Input :     file    (str)

    Outputs :   B       (list) list of bonds
                O       (list) order of the bonds
    """
    flag,flag2=0,0
    B,O=[],[]
    for line in codecs.open(file, 'r',encoding="utf-8"):
        #Collect data for the bonds. Uses flags to check when the descriptors are reached
        if "BondOrders" in line:
            flag=1
        elif flag==1 and "1" in line:
            flag=2
            B.append(line.split()[:-1])
            O.append(float(line.split()[-1]))

        elif flag==2 and not ("End" in line):
            B.append(line.split()[:-1])
            O.append(float(line.split()[-1]))
        elif flag==2: flag=3
    return B,O
