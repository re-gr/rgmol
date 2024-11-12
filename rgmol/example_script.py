#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rgmol
import argparse
import os
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extraction, computation and graphical display of adf file')
    parser.add_argument('-i', default="output_examples//methanes//bromomethane//bromomethane.out", help='Input file of adf')
    parser.add_argument('-o',default="X",help="Choose do the 3d representation of a diagonalized matrix (format : matrix or matrix,vector) Matrices :  X : linear response, S* local softness, E* local hardness, f* fukui function (f(r)*f(r')/eta), XS* for the sum of X and S; with * being either : 0,+,-")
    parser.add_argument('-s',default=0,help="Save the 3d representation in a .png (0) or a .svg (1) file (Default : -1)")
    args = parser.parse_args()

    file=args.i
    fileout,filerun,file=rgmol.rename(file)


    if not(os.path.isfile(fileout)):
        raise ValueError("The output file could not be found")
    if not(os.path.isfile(filerun)):
        print("The input file could not be found")
        B,Ord=[],[]
    else: B,Ord=rgmol.extrac_adf.bonds(filerun)

    eta=rgmol.extrac_adf.glob_desc(fileout)[0][4]
    X,S0,Name=rgmol.extrac_adf.ker(fileout)
    fp,fm,f0,f2,Name=rgmol.extrac_adf.fukui(fileout,eta=eta)

    # Xvp,XV,S0vp,S0V,Spvp,SpV,Smvp,SmV,XS0vp,XS0V,XSpvp,XSpV,XSmvp,XSmV,E0vp,E0V,Epvp,EpV,Emvp,EmV,F2vp=outputV(file+".diagV",X,S0,Sp,Sm,Name,f2)
    # Fpvp,FpV,Fmvp,FmV,F0vp,F0V,F2vp,F2V=outputF(file+".diagF",fp,fm,f0,f2,Name)
    # outputA(file+".diagA",X,S0,Name)
    # output_comp(file+".diagComp",X,S0,Sp,Sm,fp,fm,f0,f2,Name)


    # Papier 95 Chattaraj
    # E0=np.linalg.inv(S0)
    # f0=np.array([0.4586 ,0.4586 ,0.0414,0.0414])
    # print(np.(f0.dot(E0)))


    # for j in range(len(XV)):
    # for j in range(1):
    #     vec=XV[:,j]
    #     # print(vec,np.sum(vec**2))
    #     summ=0
    #     for i in range(len(X)):
    #         for k in range(len(X)):
    #             a,b=X[i][i],X[k][k]
    #             c=X[k][i]
    #             if a*b==c*c: Cmut=a
    #             else: Cmut=(a*b-c*c)/(a+b+2*c)
    #             if i!=k:
    #                 summ+=XV[i,j]*XV[k,j]*Cmut
    #                 print("mult",i,k,XV[i,j]*XV[k,j],Cmut,XV[i,j]*XV[k,j]*Cmut)
    #     print(summ)
    # print(Xvp/2)
    #
    # for i in range(len(X)):
    #     for j in range(len(X)):
    #         som=0
    #         for k in range(len(X)):
    #             som+=Xvp[k]*XV[i][k]*XV[j][k]
    #         print(i,j,som)


    #There must be a quicker way, but I don't want to think too much about it as it is fast enough
    # L=X
    # alph1,alph2,alph3=np.zeros((len(L),len(L))),np.zeros((len(L),len(L))),np.zeros((len(L),len(L)))
    # for k in range(len(L)):
    #     for j in range(len(L[0])):
    #         OverlineX = (L[k][k] * L[j][j])**(1/2)
    #         alph1[k][j]=L[k][j]/OverlineX
    #         alph2[k][j]=1-L[k][j]/OverlineX
    #         alph3[k][j]=(1-L[k][j]/OverlineX)/2
    #
    # print("-"*10)
    # for n in range(len(X)):
    #     somme=0
    #     for k in range(len(X)):
    #         if k!=n:
    #             print("diff",k,n,(X[k][k] - X[n][n]))
    #             somme+=alph1[n][k]**2 * X[k][k] * X[n][n] / (-X[k][k] + X[n][n])
    #     print(n,somme+X[n][n])
    # print(Xvp)

    Pos,Rad,Name=rgmol.extrac_adf.pos_rad(fileout)
    O=args.o
    save=int(args.s)

    Xvp,XV = np.linalg.eigh(X)
    rgmol.plot_all_3d(Pos,Rad,XV,Xvp,Name,B,Ord,file+"X",save)
