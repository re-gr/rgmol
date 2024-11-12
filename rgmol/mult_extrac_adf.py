#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import extrac_adf as eadf
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def find(folder,r,e,s):
    """
    Finds the output files in a folder, it descends recursively into folders.
    """
    F=os.listdir(folder)
    flag=0 #Check if there was a output file in the folder : do not descend if already found an output file
    Lexcl=e.split(",")
    Folders=[]
    List_out=[]
    for k in F:
        file=folder+"//"+k
        flagexcl=0
        for exc in Lexcl:
            if exc in k:
                flagexcl=1
        if os.path.isdir(file): Folders.append(file)
        elif flagexcl==0 and k[-3:]==r:
            flag=1
            List_out.append(file)

    if flag==0:
        for f in Folders:
            List_out=List_out+find(f,r,e,s) #If folder descend in it
    return List_out


def no_plot(List_out):
    """
    If you choose not to plot the eigenvectors, the diag function will stil be produced
    """
    for file in List_out:
        fileout,filerun,file=eadf.rename(file)
        eta=eadf.glob_desc(fileout)[0][4]
        X,S0,Name=eadf.ker(fileout)
        fp,fm,f0,f2,Name=eadf.fukui(fileout,eta=eta)
        Sp,Sm=eadf.create_newS(X,fp),eadf.create_newS(X,fm)
        Pos,Rad,Name=eadf.pos_rad(fileout)
        B,Ord=eadf.bonds(filerun)
        Xvp,XV,S0vp,S0V,Spvp,SpV,Smvp,SmV,XS0vp,XS0V,XSpvp,XSpV,XSmvp,XSmV,E0vp,E0V,Epvp,EpV,Emvp,EmV,F2vp=eadf.outputV(file+".diagV",X,S0,Sp,Sm,Name,f2)
        Fpvp,FpV,Fmvp,FmV,F0vp,F0V,F2vp,F2V=eadf.outputF(file+".diagF",fp,fm,f0,f2,Name)
        eadf.outputA(file+".diagA",X,S0,Name)



def plot_indiv(List_out,s):
    """
    Plots for each molecules individually
    """
    for file in List_out:
        fileout,filerun,file=eadf.rename(file)
        eta=eadf.glob_desc(fileout)[0][4]
        X,S0,Name=eadf.ker(fileout)
        fp,fm,f0,f2,Name=eadf.fukui(fileout,eta=eta)
        Sp,Sm=eadf.create_newS(X,fp),eadf.create_newS(X,fm)
        Pos,Rad,Name=eadf.pos_rad(fileout)
        B,Ord=eadf.bonds(filerun)
        Xvp,XV,S0vp,S0V,Spvp,SpV,Smvp,SmV,XS0vp,XS0V,XSpvp,XSpV,XSmvp,XSmV,E0vp,E0V,Epvp,EpV,Emvp,EmV,F2vp=eadf.outputV(file+".diagV",X,S0,Sp,Sm,Name,f2)
        Fpvp,FpV,Fmvp,FmV,F0vp,F0V,F2vp,F2V=eadf.outputF(file+".diagF",fp,fm,f0,f2,Name)
        eadf.outputA(file+".diagA",X,S0,Name)
        eadf.plot_all_3d(Pos,Rad,XV,Xvp,Name,B,Ord,file+"X",s)
        # eadf.plot_all_3d(Pos,Rad,S0V,S0vp,Name,B,Ord,file+"S0",s)
        # eadf.plot_all_3d(Pos,Rad,SpV,Spvp,Name,B,Ord,file+"S+",s)
        # eadf.plot_all_3d(Pos,Rad,SmV,Smvp,Name,B,Ord,file+"S-",s)
        # eadf.plot_all_3d(Pos,Rad,E0V,E0vp,Name,B,Ord,file+"E0",s)
        # eadf.plot_all_3d(Pos,Rad,EpV,Epvp,Name,B,Ord,file+"E+",s)
        # eadf.plot_all_3d(Pos,Rad,EmV,Emvp,Name,B,Ord,file+"E-",s)
        # eadf.plot_all_3d(Pos,Rad,F0V,F0vp,Name,B,Ord,file+"f0",s)
        # eadf.plot_all_3d(Pos,Rad,FpV,Fpvp,Name,B,Ord,file+"f+",s)
        # eadf.plot_all_3d(Pos,Rad,FmV,Fmvp,Name,B,Ord,file+"f-",s)
        # eadf.plot_all_3d(Pos,Rad,XS0V,XS0vp,Name,B,Ord,file+"XS0",s)
        # eadf.plot_all_3d(Pos,Rad,XSpV,XSpvp,Name,B,Ord,file+"XS+",s)
        # eadf.plot_all_3d(Pos,Rad,XSmV,XSmvp,Name,B,Ord,file+"XS-",s)


def plot_all_in_line(List_out,save,modevec,Rota_arb=[],Scale=[],List_name=[]):
    """
    plots all vectors for all molecules in line
    """
    plot_in_line(List_out,save,"X",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"S0",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"S+",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"S-",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"E0",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"E+",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"E-",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"f0",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"f+",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"f-",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"XS0",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"XS+",modevec,Rota_arb,Scale,List_name)
    # plot_in_line(List_out,save,"XS-",modevec,Rota_arb,Scale,List_name)



def plot_all_one_vec(List_out,nvec,save,modevec,Rota_arb=[],Scale=[],List_name=[]):
    """
    plots only one vector for all molecules
    """
    plot_one_vec(List_out,nvec,save,"X",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"S0",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"S+",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"S-",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"E0",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"E+",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"E-",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"f0",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"f+",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"f-",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"XS0",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"XS+",modevec,Rota_arb,Scale,List_name)
    # plot_one_vec(List_out,nvec,save,"XS-",modevec,Rota_arb,Scale,List_name)



def plot_in_line(List_out,save,vec,modevec,Rota_arb,Scale,List_name):
    """
    Plots the vectors in line
    """
    ncols,nrows=0,len(List_out)
    # print(vec)
    for file in List_out:
        fileout,filerun,file=eadf.rename(file)
        eta=eadf.glob_desc(fileout)[0][4]
        X,S0,Name=eadf.ker(fileout)
        if len(X)>ncols:
            ncols=len(X)
    if modevec=="2":
        ncols-=1
    # ncols=6
    if save>=0:
        fig=plt.figure(figsize=(3.1*ncols,2.8*nrows),dpi=200) #x: ~3.1 for 1 vector; y: ~2.8 for one molecule
    else:
        fig=plt.figure()
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['svg.fonttype'] = 'none'

    #remove grid background
    plt.rcParams['axes.edgecolor'] = 'none'
    plt.rcParams['axes3d.xaxis.panecolor'] = 'none'
    plt.rcParams['axes3d.yaxis.panecolor'] = 'none'
    plt.rcParams['axes3d.zaxis.panecolor'] = 'none'


    for row in range(len(List_out)):
        file=List_out[row]
        fileout,filerun,file=eadf.rename(file)
        eta=eadf.glob_desc(fileout)[0][4]
        X,S0,Name=eadf.ker(fileout)
        fp,fm,f0,f2,Name=eadf.fukui(fileout,eta=eta)
        Sp,Sm=eadf.create_newS(X,fp),eadf.create_newS(X,fm)
        Pos,Rad,Name=eadf.pos_rad(fileout)
        B,Ord=eadf.bonds(filerun)
        Xvp,XV,S0vp,S0V,Spvp,SpV,Smvp,SmV,XS0vp,XS0V,XSpvp,XSpV,XSmvp,XSmV,E0vp,E0V,Epvp,EpV,Emvp,EmV,F2vp=eadf.outputV(file+".diagV",X,S0,Sp,Sm,Name,f2)
        eadf.outputA(file+".diagA",X,S0,Name)
        if modevec=="2":
            XV=XV[:,:-1]
            Xvp=Xvp[:-1]
        # if row==0 or row==1:
        #     XV=XV[:,:5]
        #     Xvp=Xvp[:5]

        # else:
        #     Xvp=Xvp[:6]
        #     XV=XV[:,:6]
        Fpvp,FpV,Fmvp,FmV,F0vp,F0V,F2vp,F2V=eadf.outputF(file+".diagF",fp,fm,f0,f2,Name)
        eadf.output_comp(file+".diagComp",X,S0,Sp,Sm,fp,fm,f0,f2,Name)

        if Rota_arb==[]:
            Rota=0
        else: Rota=Rota_arb[row]
        if vec=="X": eadf.plot_line_3d(Pos,Rad,XV,Xvp,Name,B,Ord,file+"X",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="S0": eadf.plot_line_3d(Pos,Rad,S0V,S0vp,Name,B,Ord,file+"S0",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="S+": eadf.plot_line_3d(Pos,Rad,SpV,Spvp,Name,B,Ord,file+"S+",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="S-": eadf.plot_line_3d(Pos,Rad,SmV,Smvp,Name,B,Ord,file+"S-",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="E0": eadf.plot_line_3d(Pos,Rad,E0V,E0vp,Name,B,Ord,file+"E0",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="E+": eadf.plot_line_3d(Pos,Rad,EpV,Epvp,Name,B,Ord,file+"E+",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="E-": eadf.plot_line_3d(Pos,Rad,EmV,Emvp,Name,B,Ord,file+"E-",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="f0": eadf.plot_line_3d(Pos,Rad,F0V,F0vp,Name,B,Ord,file+"f0",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="f+": eadf.plot_line_3d(Pos,Rad,FpV,Fpvp,Name,B,Ord,file+"f+",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="f-": eadf.plot_line_3d(Pos,Rad,FmV,Fmvp,Name,B,Ord,file+"f-",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="XS0": eadf.plot_line_3d(Pos,Rad,XS0V,XS0vp,Name,B,Ord,file+"XS0",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="XS+": eadf.plot_line_3d(Pos,Rad,XSpV,XSpvp,Name,B,Ord,file+"XS+",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
        if vec=="XS-": eadf.plot_line_3d(Pos,Rad,XSmV,XSmvp,Name,B,Ord,file+"XS-",nrows,ncols,row,fig,modevec,Rota,Scale,List_name)
    if save==0:
        plt.savefig(file.split('//')[0]+"//"+file.split('//')[0]+vec+".png",transparent=True,bbox_inches="tight", pad_inches=0.1)
    elif save==1:
        plt.savefig(file.split('//')[0]+"//"+file.split('//')[0]+vec+".svg",transparent=True,bbox_inches="tight", pad_inches=0.1)
    else: plt.show()
    plt.close()

def plot_one_vec(List_out,nvec,save,vec,modevec,Rota_arb,Scale,List_name):
    """
    Plots only one vector
    """
    ncols,nrows=len(List_out),1
    if modevec=="1":
        ncols-=1
    if save>=0:
        fig=plt.figure(figsize=(20,4),dpi=200)
    else:
        fig=plt.figure()
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['svg.fonttype'] = 'none'

    #remove grid background
    plt.rcParams['axes.edgecolor'] = 'none'
    plt.rcParams['axes3d.xaxis.panecolor'] = 'none'
    plt.rcParams['axes3d.yaxis.panecolor'] = 'none'
    plt.rcParams['axes3d.zaxis.panecolor'] = 'none'


    for col in range(len(List_out)):
        file=List_out[col]
        fileout,filerun,file=eadf.rename(file)
        eta=eadf.glob_desc(fileout)[0][4]
        X,S0,Name=eadf.ker(fileout)
        fp,fm,f0,f2,Name=eadf.fukui(fileout,eta=eta)
        Sp,Sm=eadf.create_newS(X,fp),eadf.create_newS(X,fm)
        Pos,Rad,Name=eadf.pos_rad(fileout)
        B,Ord=eadf.bonds(filerun)
        Xvp,XV,S0vp,S0V,Spvp,SpV,Smvp,SmV,XS0vp,XS0V,XSpvp,XSpV,XSmvp,XSmV,E0vp,E0V,Epvp,EpV,Emvp,EmV,F2vp=eadf.outputV(file+".diagV",X,S0,Sp,Sm,Name,f2)

        Fpvp,FpV,Fmvp,FmV,F0vp,F0V,F2vp,F2V=eadf.outputF(file+".diagF",fp,fm,f0,f2,Name)
        if Rota_arb==[]:
            Rota=0
        else: Rota=Rota_arb[col]

        if modevec=="1":
            XV=XV[:,:-1]
            Xvp=Xvp[:-1]
        if vec=="X": eadf.plot_one_vec_3d(Pos,Rad,XV,Xvp,nvec,Name,B,Ord,file+"X",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="S0": eadf.plot_one_vec_3d(Pos,Rad,S0V,S0vp,nvec,Name,B,Ord,file+"S0",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="S+": eadf.plot_one_vec_3d(Pos,Rad,SpV,Spvp,nvec,Name,B,Ord,file+"S+",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="S-": eadf.plot_one_vec_3d(Pos,Rad,SmV,Smvp,nvec,Name,B,Ord,file+"S-",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="E0": eadf.plot_one_vec_3d(Pos,Rad,E0V,E0vp,nvec,Name,B,Ord,file+"E0",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="E+": eadf.plot_one_vec_3d(Pos,Rad,EpV,Epvp,nvec,Name,B,Ord,file+"E+",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="E-": eadf.plot_one_vec_3d(Pos,Rad,EmV,Emvp,nvec,Name,B,Ord,file+"E-",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="f0": eadf.plot_one_vec_3d(Pos,Rad,F0V,F0vp,nvec,Name,B,Ord,file+"f0",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="f+": eadf.plot_one_vec_3d(Pos,Rad,FpV,Fpvp,nvec,Name,B,Ord,file+"f+",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="f-": eadf.plot_one_vec_3d(Pos,Rad,FmV,Fmvp,nvec,Name,B,Ord,file+"f-",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="XS0": eadf.plot_one_vec_3d(Pos,Rad,XS0V,XS0vp,nvec,Name,B,Ord,file+"XS0",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="XS+": eadf.plot_one_vec_3d(Pos,Rad,XSpV,XSpvp,nvec,Name,B,Ord,file+"XS+",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)
        if vec=="XS-": eadf.plot_one_vec_3d(Pos,Rad,XSmV,XSmvp,nvec,Name,B,Ord,file+"XS-",nrows,ncols,col,fig,modevec,Rota,Scale,List_name)


    ax=fig.add_subplot(1,1,1)
    ax.set_title("Eigenvector n°{} multiplied by the eigenvalues".format(nvec),fontsize=20)
    # ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_color('black') #add box
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set_facecolor("none")

    if save==0:
        plt.savefig(file.split('//')[0]+"//"+file.split('//')[0]+vec+"_{}".format(nvec)+".png",transparent=True,bbox_inches="tight", pad_inches=0.1)
    elif save==1:
        plt.savefig(file.split('//')[0]+"//"+file.split('//')[0]+vec+"_{}".format(nvec)+".svg",transparent=True,bbox_inches="tight", pad_inches=0.1)
    else: plt.show()
    plt.close()

def find_rota(List_out,Rota_arb,Scale,bond,mol):
    """
    find the rotation in order to have bonds at the same angle

    bond : format : [[num_bond_mol,num_bond], [...]]

    """

    LVec=[]
    if Rota_arb==[]:
        Rota_arb=[0]*len(List_out)
    if len(bond)==1:
        bond=bond*len(List_out)
    file=List_out[mol]
    fileout,filerun,file=eadf.rename(file)
    Pos,Rad,Name=eadf.pos_rad(fileout)
    B_mol,Ord=eadf.bonds(filerun)
    Pos_mol=eadf.transf_pos(Pos,B_mol,Ord,Rota_arb[mol],mol,Scale)
    for molecule in range(len(List_out)):
        file=List_out[molecule]
        fileout,filerun,file=eadf.rename(file)
        Pos,Rad,Name=eadf.pos_rad(fileout)
        B,Ord=eadf.bonds(filerun)
        if B!=[]:
            a,b=B[bond[molecule][1]]
            amol,bmol=B_mol[bond[molecule][0]]
            Pos=eadf.transf_pos(Pos,B,Ord,Rota_arb[molecule],molecule,Scale)
            LVec.append( [Pos_mol[int(bmol)-1]-Pos_mol[int(amol)-1],Pos[int(b)-1]-Pos[int(a)-1] ] )
        else: LVec.append(-1) #There are no bonds

    for k in range(len(List_out)):
        if LVec[k]!=-1: #If there is a bond
            cross=np.cross(LVec[k],(LVec[mol])) #Calculate the sign of the angle
            cross= cross*(abs(cross)>1e-5)#Remove almost zero components
            scross=np.sign(np.prod(cross[np.where(cross!=0)]))
            cosangle=(LVec[k][1].dot(LVec[k][0]) / np.linalg.norm(LVec[k][0])/np.linalg.norm(LVec[k][1]))#cos angle
            if cosangle>1: cosangle=1 #If the two vectors are colinear, in some cases the scalar product gives 1 + 1e-15 which produces an error
            if cosangle<-1: cosangle=-1
            Rota_arb[k]+=( np.arccos(cosangle))*scross
    return Rota_arb





if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extraction, computation and saving representation of a folder containing adf files')
    parser.add_argument('-f', default="frag", help='Folder containing the adf folder / files (default: A)')
    parser.add_argument('-r',default="out",help="adf ouptut files format (default:out)")
    parser.add_argument('-e',default="opt,mout",help="adf ouptut files containing these strings will be skipped format : 'extension1,extension2,...'")
    parser.add_argument('-s',default="0",help="Save the 3d representation in a .png (0) or a .svg (1) file (Default : -1)")
    parser.add_argument('-m',default="2",help="Mode 0 : no plot; 1 : plot individually; 2 : plot on the same graph; 3 : plot one vec on the same graph (Default : 0)")
    parser.add_argument('-mv',default="3",help="Mode vec 1 : normal; 2 : remove last eigenvector; 3 : multiply eigenvectors by eigenvalues (Default : 1)")
    args = parser.parse_args()

    folder=args.f
    List_out=find(folder,args.r,args.e,int(args.s))
    Scal=[]
    Rota_arb=[]
    bond=[[0,0]]
    List_name=[]


    #The Rota_arb list is a arbitrary rotation of the molecules on the plane perpendicual to the visualization axis
    #Scal is a scaling list, which principal usefulness is to do 180 rotation along an axis
    #The format for the Bond list is : bond [ [...], [the index of the bond of the first molecule to be parallel to, the index of the bond of this molecule to rotate], [...] ]
    ## The following commented lines are examples of inputs used ##



    # List_out=["form//formamide//formamide.out","form//formiatemet//formiatemet.out","form//acideformic//acideformic.out","form//formaldehyde//formaldehyde.out","form//formylchloride//formylchloride.out"]
    # List_name=["Formamide","Methyl Formate","Formic Acid","Formaldehyde","Formyl Chloride"]
    # Scal=[[1,1,1],[1,1,1],[1,1,1],[1,1,1],[-1,1,1]]
    # Rota_arb=[0,0,0,0,0]


    # List_out=["form//formylchloride//formylchloride.out"]
    # List_name=["Formyl Chloride"]
    # Rota_arb=[0]



    List_out=["methanes//methane//methane.out","methanes//fluoromethane//fluoromethane.out","methanes//chloromethane//chloromethane.out","methanes//bromomethane//bromomethane.out","methanes//iodomethane//iodomethane.out"]
    Rota_arb=[-np.pi/2.5,0,0,0,np.pi]


    # List_out=["frag//frag2//ethane//ethane.out","frag//frag2//ch3//ch3.out","frag//frag2//h2//h2.out"]
    # bond=[[0,0],[0,0],[5,2],[0,0]]



    # List_out=["phen//benzene//benzene.out","phen//phenol//phenol.out","phen//benzaldehyde//benzaldehyde.out","phen//nitrobenzene//nitrobenzene.out"]
    # List_name=["Benzene","Phenol","Benzaldehyde","Nitrobenzene"]
    # bond=[[0,0],[0,0],[0,0]]


    # List_out=['ac//acideacrylic//acideacrylic.out','ac//acroleine//acroleine.out','ac//acrylonitrile//acrylonitrile.out','ac//nitroethylene//nitroethylene.out']
    # List_name=[]
    # bond=[[0,0],[0,0],[2,0],[0,1]]



    # List_out=["form//acetaldehyde//acetaldehyde.out","frag//ch3//ch3.out","frag//h3+//h3+.out","frag//h2//h2.out"]
    # bond=[[0,0],[0,0],[5,2]]
    # List_out=["frag//HCCH//HCCH.out","frag//HOOH/HOOH.out","frag//difluorodiazene//difluorodiazene.out"]
    # List_name=['Acetylene',"Hydrogen Peroxide","Difluorodiazene"]
    # Rota_arb=[-np.pi/32,0,0,0]


    # List_out=["frag//Ethane//ethane2.out","frag//Ethane//ethane3.out","frag//Ethane//ch3.out"]
    # List_name=["Ethane",r"$CH_3$"]
    # Rota_arb=[0,0]


    # List_out=["A//nitrone//nitrone.out"]
    # List_name=[r"Formaldonitrone"]
    # Rota_arb=[-0.13]

    if args.m=="0":
        no_plot(List_out)
    if args.m=="1":
        plot_indiv(List_out,int(args.s))
    if args.m=="2":
        Rota_arb=find_rota(List_out,Rota_arb,Scal,bond,0)
        plot_all_in_line(List_out,int(args.s),args.mv,Rota_arb=Rota_arb,Scale=Scal,List_name=List_name)
    if args.m=="3":
        Rota_arb=find_rota(List_out,Rota_arb,Scal,bond,0)
        plot_all_one_vec(List_out,0,int(args.s),args.mv,Rota_arb=Rota_arb,Scale=Scal,List_name=List_name)
    # print(List_out)