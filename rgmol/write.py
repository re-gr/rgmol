#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os

#######################
## Writing functions ##
#######################
def write_m(f,Name,M):
    """
    Writes the matrix M inside the file f
    Input :     f       (str) file
                Name    (list) the name of the atoms
                M       (2D array or list) the matrix

    Outputs :   None
    """
    for k in range(len(M)):
        f.write(Name[k][1]+Name[k][0]+" "*(3-len(Name[k][1]+Name[k][0])))
        for j in range(len(M[k])):
            f.write(" "*(9-len("{:3.3f}".format(M[k][j])))+"{:3.3f}".format(M[k][j]))
        f.write("\n")
    f.write("\n")
    return

def write_lrls(file,L,S0,Sp,Sm,Name,f2):
    """
    Write the linear response and local softness kernels, their diagonalization and the latex table in a .txt file
    Outputs the eigenvals and eigenvectors of the functions

    Input : file (str)
            L   (ndarray)    condensed linear response kernel
            S0  (ndarray)    condensed local softness kernel S0
            Sp  (ndarray)    condensed local softness kernel S+
            Sm  (ndarray)    condensed local softness kernel S-
            Name (list of str); name of the atoms

    Outputs :   Lvp (ndarray); eigenvals of L
                LV (ndarray); eigenvectors of L
                Svp (ndarray); eigenvals of S
                SV (ndarray); eigenvectors of S
    """

    def write_S(S,num):
        """write"""
        Svp,SV=np.linalg.eigh(S)
        Svp=Svp[::-1]
        SV=SV[:,::-1]
        LSvp,LSV=np.linalg.eigh(L+S)


        f.write("Diagonalized S{} :\n".format(num))
        f.write("\n")

        write_m(f,Name,SV)
        f.write("vp ")
        for k in range(len(Svp)):
            f.write(" "*(9-len("{:3.3f}".format(Svp[k])))+"{:3.3f}".format(Svp[k]))

        return Svp,SV

    def write_Eta(Eta,num):
        """write"""
        Evp,EV=np.linalg.eigh(Eta)
        f.write("Diagonalized Eta{} :\n".format(num))
        f.write("\n")

        write_m(f,Name,EV)
        f.write("vp ")
        for k in range(len(Evp)):
            f.write(" "*(9-len("{:3.3f}".format(Evp[k])))+"{:3.3f}".format(Evp[k]))

        return Evp,EV

    def export_X_latex(X,Name):
        delimit=" & "
        a="\\begin{table}[h!t] \n\\centering\n\\begin{tabular}{c@{\\hskip 0.07\\textwidth} *"+str(len(X)-1)+"{S[table-format=1.3]@{\\hskip 0.07\\textwidth}} S[table-format=1.3]}\n\\toprule\nAtoms(k)"
        for k in range(len(Name)):
            a+=delimit + "\\text{" + Name[k][1]+Name[k][0] + "}"
        a+="\\\\\n"
        a+="\\midrule\n"
        for k in range(len(X)):
            a+=Name[k][1]+Name[k][0]
            for j in range(len(X)):
                a+=delimit+"{:3.3f}".format(X[k][j])
            a+="\\\\\n"
        a+="\\toprule\n\\end{tabular}\n\\caption{Condensed Linear Response function for }\n\\end{table}"
        return a

    def export_XV_latex(X,Xvp,Name):
        delimit=" & "
        a="\\begin{table}[h!t] \n\\centering\n\\begin{tabular}{c@{\\hskip 0.07\\textwidth} *"+str(len(X)-1)+"{S[table-format=1.3]@{\\hskip 0.07\\textwidth}} S[table-format=1.3]}\n\\toprule\nAtoms(k)"
        for j in range(len(X)):
            a+="& \\text{$\\chi_" +str(j)+"(k)$}"
        a+="\\\\\n\\midrule\n"
        for k in range(len(X)):
            a+=Name[k][1]+Name[k][0]
            for j in range(len(X)):
                a+=delimit+"{:3.3f}".format(X[k][j])
            a+="\\\\\n"
        a+="\\midrule\n$\\lambda(a.u.)$"
        for k in Xvp:
            a+=delimit+"{:3.3f}".format(k)
        a+="\\\\\n\\toprule\n\\end{tabular}\n\\caption{Linear Response Eigenvectors and eigenvalues for }\n\\end{table}"
        return a
    Lvp,LV=np.linalg.eigh(L)
    Eta0=np.linalg.inv(S0)
    Etap=np.linalg.inv(Sp)
    Etam=np.linalg.inv(Sm)

    with open(file,"w") as f:
        f.write("============================================\n")
        f.write("|| Diagonalization of the linear response ||\n")
        f.write("||     and the local softness kernels     ||\n")
        f.write("============================================\n")
        f.write("\n")
        f.write("Condensed linear response :\n")
        f.write("\n")

        for k in range(len(Name)):
            f.write(" "*(9-len(Name[k][1]+Name[k][0]))+Name[k][1]+Name[k][0])
        f.write("\n")

        write_m(f,Name,L)


        f.write("\n")
        f.write("Diagonalized linear response kernel :\n")
        f.write("\n")

        write_m(f,Name,LV)
        f.write("vp ")
        for k in range(len(Lvp)):
            f.write(" "*(9-len("{:3.3f}".format(Lvp[k])))+"{:3.3f}".format(Lvp[k]))
        f.write("\n\n\n")
        f.write("Contribution of each atom on the EDDMs (vp_i*XV_i**2):\n")
        f.write("\n")
        write_m(f,Name,abs(np.reshape(Lvp,(1,len(Lvp))))*(LV**2))
        f.write("\n")

        f.write("\n")
        f.write("S0 :\n")
        f.write("\n")

        for k in range(len(Name)):
            f.write(" "*(9-len(Name[k][1]+Name[k][0]))+Name[k][1]+Name[k][0])
        f.write("\n")

        write_m(f,Name,S0)


        S0vp,S0V=write_S(S0,"0")
        f.write("\n\n\n")
        f.write("Contribution of each atom on the eigenvectors of S0 (vp_i*S0V_i**2):\n")
        f.write("\n")
        write_m(f,Name,abs(np.reshape(S0vp,(1,len(S0vp))))*(S0V**2))
        f.write("\n")


        f.write("\n\n\n")
        Spvp,SpV=write_S(Sp,"+")
        f.write("\n\n\n")
        Smvp,SmV=write_S(Sm,"-")

        f.write("\n\n\n")
        E0vp,E0V=write_Eta(Eta0,"0")
        f.write("\n\n\n")
        Epvp,EpV=write_Eta(Etap,"+")
        f.write("\n\n\n")
        Emvp,EmV=write_Eta(Etam,"-")

        f.write("\n\n\n")
        f.write("Projection of the dual operator on the eigenvector of the linear response :\n\n")
        F2vp=LV.dot(f2)
        f.write("   ")
        for k in range(len(F2vp)):
            f.write(" "*(9-len("{:3.3f}".format(F2vp[k])))+"{:3.3f}".format(F2vp[k]))

        f.write("\n\n\n")
        f.write("E(2) = {:3.6f} eV".format(-1/2*np.sum(F2vp[:-1]**2/abs(Lvp[:-1])))) #the projection of the dual descriptor on the constant potential should be zero
        f.write('\n\n\n')
        f.write("==========================================================\n")
        f.write("||                      Latex Codes                     ||\n")
        f.write("|| The packages used are tabularx, siunitx and booktabs ||\n")
        f.write("==========================================================\n")
        f.write("\n\n")
        f.write("LaTeX Code for the non diagonalized condensed linear response\n\n")
        f.write(export_X_latex(L,Name))
        f.write("\n\n\n")
        f.write("LaTeX Code for the diagonalized condensed linear response\n\n")
        f.write(export_XV_latex(LV,Lvp,Name))
        f.write("\n\n\n")
        f.write("LaTeX Code for the contribution on the EDDMs\n\n")
        f.write(export_X_latex(LV**2*abs(Lvp),Name))
        f.write("\n\n")
        f.write("LaTeX Code for the non diagonalized S0\n\n")
        f.write(export_X_latex(S0,Name))
        f.write("\n\n\n")
        f.write("LaTeX Code for the diagonalized S0\n\n")
        f.write(export_XV_latex(S0V,S0vp,Name))
        f.write("\n\n\n")
        f.write("LaTeX Code for the contribution on the eigenvectors of S0\n\n")
        f.write(export_X_latex(S0V**2*abs(S0vp),Name))


    LS0vp,LS0V=np.linalg.eigh(L+S0)
    LSpvp,LSpV=np.linalg.eigh(L+Sp)
    LSmvp,LSmV=np.linalg.eigh(L+Sm)
    return Lvp,LV,S0vp,S0V,Spvp,SpV,Smvp,SmV,LS0vp,LS0V,LSpvp,LSpV,LSmvp,LSmV,E0vp,E0V,Epvp,EpV,Emvp,EmV,F2vp


def write_fukui(file,fp,fm,f0,f2,Name):
    """
    Write the product of the fukui functions and their diagonalization in a .txt file
    Outputs the eigenvals and eigenvalue of the functions

    Input : file (str)
            fp (ndarray); f+
            fm (ndarray); f-
            f0 (ndarray)
            f2 (ndarray); f(2)
            Name (list of str); name of the atoms

    Outputs :   Fpvp (ndarray); eigenvals of f+
                FpV (ndarray); eigenvectors of f+
                Fmvp (ndarray); eigenvals of f-
                FmV (ndarray); eigenvectors of f-
                F0vp (ndarray); eigenvals of f0
                F0V (ndarray); eigenvectors of f0
                F2vp (ndarray); eigenvals of f2
                F2V (ndarray); eigenvectors of f2
    """

    with open(file,"w") as f:
        f.write("=============================================\n")
        f.write("||     Products of the fukui functions     ||\n")
        f.write("||        and their diagonalization        ||\n")
        f.write("=============================================\n")
        f.write("\n")


        Fp=fp*fp.reshape((len(fp),1))
        f.write("Product of f+ :\n")
        f.write("\n")
        write_m(f,Name,Fp)

        f.write("Diagonalized f+ :\n")
        f.write("\n")

        Fpvp,FpV=np.linalg.eigh(Fp)
        write_m(f,Name,FpV)
        f.write("vp ")
        for k in range(len(Fpvp)):
            f.write(" "*(9-len("{:3.3f}".format(Fpvp[k])))+"{:3.3f}".format(Fpvp[k]))


        f.write("\n\n\n\n")

        Fm=fm*fm.reshape((len(fm),1))
        f.write("Product of f- :\n")
        f.write("\n")
        write_m(f,Name,Fm)

        f.write("Diagonalized f- :\n")
        f.write("\n")

        Fmvp,FmV=np.linalg.eigh(Fm)
        write_m(f,Name,FmV)
        f.write("vp ")
        for k in range(len(Fmvp)):
            f.write(" "*(9-len("{:3.3f}".format(Fmvp[k])))+"{:3.3f}".format(Fmvp[k]))

        f.write("\n\n\n\n")

        F0=f0*f0.reshape((len(f0),1))
        f.write("Product of f0 :\n")
        f.write("\n")
        write_m(f,Name,F0)

        f.write("Diagonalized f0 :\n")
        f.write("\n")

        F0vp,F0V=np.linalg.eigh(F0)
        write_m(f,Name,F0V)
        f.write("vp ")
        for k in range(len(F0vp)):
            f.write(" "*(9-len("{:3.3f}".format(F0vp[k])))+"{:3.3f}".format(F0vp[k]))

    F2=f2*f2.reshape((len(f2),1))
    F2vp,F2V=np.linalg.eigh(F2)


    return Fpvp,FpV,Fmvp,FmV,F0vp,F0V,F2vp,F2V


def output_comp(file,L,S0,Sp,Sm,fp,fm,f0,f2,Name):
    """
    Write the product of the fukui functions and their diagonalization in a .txt file
    Outputs the eigenvals and eigenvalue of the functions

    Input : file (str)
            L  (ndarray)    condensed linear response kernel
            S0 (ndarray)    condensed local softness kernel computed with f0
            Sp (ndarray)    condensed local softness kernel computed with f+
            Sm (ndarray)    condensed local softness kernel computed with f-
            fp (ndarray)    f+
            fm (ndarray)    f-
            f0 (ndarray)
            f2 (ndarray)    f(2)
            Name (list of str); name of the atoms

    Outputs :
    """
    def write_comp(f,L,S,ff,num):
        """write"""
        Lvp,LV=np.linalg.eigh(L)
        Svp,SV=np.linalg.eigh(S)
        Svp=Svp[::-1]
        SV=SV[:,::-1]
        E=np.linalg.inv(S)
        A=L+S
        LSvp,LSV=np.linalg.eigh(L+S)
        LSVinv=LSV.transpose()
        F=ff*ff.reshape((len(ff),1))
        Fvp,FV=np.linalg.eigh(F)
        FVinv=FV.transpose()

        # print((L+S)*100//1/100)
        #
        # print(F*100//1/100)
        # print(np.linalg.qr(LSV)[1]*100//1/100)
        # print(np.linalg.qr(FV)[1]*100//1/100)


        f.write("Diagonalization of f{}:\n\n".format(num))
        write_m(f,Name,FV)
        f.write("vp ")
        for k in range(len(Fvp)):
            f.write(" "*(9-len("{:3.2f}".format(Fvp[k])))+"{:3.2f}".format(Fvp[k]))
        f.write("\n\n")

        f.write("Diagonalization of L+S{}:\n\n".format(num))
        write_m(f,Name,LSV)
        f.write("vp ")
        for k in range(len(LSvp)):
            f.write(" "*(9-len("{:3.2f}".format(LSvp[k])))+"{:3.2f}".format(LSvp[k]))
        f.write("\n\n")


        f.write("Difference between the eigenvectors of L+S{} and f{}:\n".format(num,num))
        f.write("\n")
        write_m(f,Name,LSV-FV)
        f.write("\n")

        f.write("Product eigenvectors of L+S{} and inverse eigenvectors of f{}:\n".format(num,num))
        f.write("\n")
        write_m(f,Name,LSV.dot(FVinv))
        f.write("\n")

        f.write("Product eigenvectors of f{} and inverse eigenvectors of L+S{}:\n".format(num,num))
        f.write("\n")
        write_m(f,Name,LSVinv.dot(FV))
        f.write("\n")

        f.write("f{} in the basis of L+S{}:\n".format(num,num))
        f.write("\n")
        FbV=LSVinv.dot(F.dot(LSV))
        write_m(f,Name,FbV)
        f.write("\n")

        f.write("L+S{} in the basis of f{}:\n".format(num,num))
        f.write("\n")
        SbV=FVinv.dot((S+L).dot(FV))
        write_m(f,Name,SbV)
        f.write("\n")

    def write_decomp(f,L,S,ff,num):
        Lvp,LV=np.linalg.eigh(L)
        Svp,SV=np.linalg.eigh(S)
        E=np.linalg.inv(S)
        if np.linalg.det(L)!=0:
            Lm1=np.linalg.inv(L)
        else: Lm1=np.array
        f.write("\n\n\n")

        f.write("Projection of f{} on the eigenvector of the linear response :\n\n".format(num))
        Pvp=LV.dot(ff)
        f.write("vp ")

        for k in range(len(Pvp)):
            f.write(" "*(9-len("{:3.3f}".format(Pvp[k])))+"{:3.3f}".format(Pvp[k]))

        f.write("\n\n\n")
        f.write("f{} :\n\n".format(num))
        f.write("vp ")
        for k in range(len(ff)):
            f.write(" "*(9-len("{:3.3f}".format(ff[k])))+"{:3.3f}".format(ff[k]))


        f.write("\n\n\n")
        f.write("Product between the linear response and f{} (P{}) :\n\n".format(num,num))
        Pvp=L.dot(ff)
        f.write("vp ")
        for k in range(len(Pvp)):
            f.write(" "*(9-len("{:3.3f}".format(Pvp[k])))+"{:3.3f}".format(Pvp[k]))
        if np.linalg.det(L)!=0:
            f.write("\n\n\n")
            f.write("Product between the inverse of the linear response and P{} (f{}) :\n\n".format(num,num))
            Fvp=Lm1.dot(Pvp)
            f.write("vp ")
            for k in range(len(Fvp)):
                f.write(" "*(9-len("{:3.3f}".format(Fvp[k])))+"{:3.3f}".format(Fvp[k]))


        f.write("\n\n\n")
        f.write("Product between the softness S{} and f{} (P{}) :\n\n".format(num,num,num))
        Pvp=S.dot(ff)
        f.write("vp ")
        for k in range(len(Pvp)):
            f.write(" "*(9-len("{:3.3f}".format(Pvp[k])))+"{:3.3f}".format(Pvp[k]))

        f.write("\n\n\n")
        f.write("Product between the hardness E{} and P{} (f{}) :\n\n".format(num,num,num))
        Fvp=E.dot(Pvp)
        f.write("vp ")
        for k in range(len(Fvp)):
            f.write(" "*(9-len("{:3.3f}".format(Fvp[k])))+"{:3.3f}".format(Fvp[k]))

    with open(file,"w") as f:
        f.write("============================================\n")
        f.write("||Comparison between the condensed kernels||\n")
        f.write("||        and the fukui functions         ||\n")
        f.write("============================================\n")
        f.write("\n")

        write_comp(f,L,S0,f0,"0")
        f.write("\n\n\n")
        write_comp(f,L,Sp,fp,"+")
        f.write("\n\n\n")
        write_comp(f,L,Sm,fm,"-")
        f.write("\n\n\n")
        write_decomp(f,L,S0,f0,'0')
        f.write("\n\n\n")
        write_decomp(f,L,Sp,fp,'+')
        f.write("\n\n\n")
        write_decomp(f,L,Sm,fm,'-')

        Lvp,LV=np.linalg.eigh(L)

        # f.write("\n\n\n")
        # f.write("Product between the eigenvectors of X and X ? :\n\n")
        # Pvp=LV.dot(a)
        # f.write("vp ")
        # for k in range(len(Pvp)):
        #     f.write(" "*(9-len("{:3.3f}".format(Pvp[k])))+"{:3.3f}".format(Pvp[k]))
    return


def write_analogy(file,L,S0,Name):
    """
    Write the alpha coefficients and the mutual capacitance according to the electrical analogy for both the condensed linear response and the softness kernel

    Input : file (str)
            L  (ndarray)    condensed linear response kernel
            Name (list of str); name of the atoms

    Outputs :   None
    """

    def write_alpha(alph,form):
        """write"""

        f.write("The alpha factors being written as X=[{}]_ij :\n".format(form))
        f.write("\n")
        for k in range(len(Name)):
            f.write(" "*(9-len(Name[k][1]+Name[k][0]))+Name[k][1]+Name[k][0])
        f.write("\n")
        write_m(f,Name,alph)


    with open(file,"w") as f:
        f.write("=================================================\n")
        f.write("||     Electrical Analogy for the Condensed    ||\n")
        f.write("||   Linear Response and the softness kernel   ||\n")
        f.write("=================================================\n")
        f.write("\n")
        f.write("Condensed linear response :\n")
        f.write("\n")
        for k in range(len(Name)):
            f.write(" "*(9-len(Name[k][1]+Name[k][0]))+Name[k][1]+Name[k][0])
        f.write("\n")
        write_m(f,Name,L)

        #There must be a quicker way, but I don't want to think too much about it as it is fast enough
        alph1,alph2,alph3=np.zeros((len(L),len(L))),np.zeros((len(L),len(L))),np.zeros((len(L),len(L)))
        for k in range(len(L)):
            for j in range(len(L[0])):
                OverlineX = (L[k][k] * L[j][j])**(1/2)
                alph1[k][j]=L[k][j]/OverlineX
                alph2[k][j]=1-L[k][j]/OverlineX
                alph3[k][j]=(1-L[k][j]/OverlineX)/2
        write_alpha(alph1,"alphaij * (CiiCjj)**(1/2)")
        write_alpha(alph2,"(1-alphaij) * (CiiCjj)**(1/2)")
        write_alpha(alph3,"(1-2alphaij) * (CiiCjj)**(1/2)")

        f.write("Diagonalized linear response kernel :\n")
        f.write("\n")
        Lvp,LV=np.linalg.eigh(L)
        write_m(f,Name,LV)
        f.write("vp ")
        for k in range(len(Lvp)):
            f.write(" "*(9-len("{:3.3f}".format(Lvp[k])))+"{:3.3f}".format(Lvp[k]))
        f.write("\n\n\n")

        f.write("Mutual Capacitance Matrix ((CiiCjj-Cij**2)/(Cii+Cjj-2Cij)) :\n\n")

        C=np.zeros((len(L),(len(L))))
        for k in range(len(L)):
            for j in range(len(L[0])):
                a,b=L[k][k],L[j][j]
                c=L[j][k]
                if a*b==c*c: C[k][j]=a
                else: C[k][j]=(a*b-c*c)/(a+b+2*c)

        for k in range(len(Name)):
            f.write(" "*(9-len(Name[k][1]+Name[k][0]))+Name[k][1]+Name[k][0])
        f.write("\n")
        write_m(f,Name,C)

        f.write("Eigenvalues divided by 2 :\nvp ")
        for k in range(len(Lvp)):
            f.write(" "*(9-len("{:3.3f}".format(Lvp[k]/2)))+"{:3.3f}".format(Lvp[k]/2))

        #Same functions but for the softness kernel

        f.write("\n\n\n\n")
        f.write("Condensed softness kernel:\n")
        f.write("\n")
        for k in range(len(Name)):
            f.write(" "*(9-len(Name[k][1]+Name[k][0]))+Name[k][1]+Name[k][0])
        f.write("\n")
        write_m(f,Name,S0)
        f.write("\n\n\n")

        #There must be a quicker way, but I don't want to think too much about it as it is fast enough
        alph1,alph2,alph3=np.zeros((len(S0),len(S0))),np.zeros((len(S0),len(S0))),np.zeros((len(S0),len(S0)))
        for k in range(len(L)):
            for j in range(len(L[0])):
                OverlineX = (S0[k][k] * S0[j][j])**(1/2)
                alph1[k][j]=S0[k][j]/OverlineX
                alph2[k][j]=1-S0[k][j]/OverlineX
                alph3[k][j]=(1-S0[k][j]/OverlineX)/2
        write_alpha(alph1,"alphaij * (CiiCjj)**(1/2)")
        write_alpha(alph2,"(1-alphaij) * (CiiCjj)**(1/2)")
        write_alpha(alph3,"(1-2alphaij) * (CiiCjj)**(1/2)")

        f.write("Diagonalized softness kernel :\n")
        f.write("\n")
        S0vp,S0V=np.linalg.eigh(S0)
        write_m(f,Name,S0V)
        f.write("vp ")
        for k in range(len(S0vp)):
            f.write(" "*(9-len("{:3.3f}".format(S0vp[k])))+"{:3.3f}".format(S0vp[k]))
        f.write("\n\n\n")
        f.write("Mutual Capacitance Matrix ((CiiCjj-Cij**2)/(Cii+Cjj-2Cij)) :\n\n")

        CS0=np.zeros((len(S0),(len(S0))))
        for k in range(len(S0)):
            for j in range(len(S0[0])):
                a,b=S0[k][k],S0[j][j]
                c=S0[j][k]
                if a*b==c*c: CS0[k][j]=a
                else: CS0[k][j]=(a*b-c*c)/(a+b+2*c)
        for k in range(len(Name)):
            f.write(" "*(9-len(Name[k][1]+Name[k][0]))+Name[k][1]+Name[k][0])
        f.write("\n")
        write_m(f,Name,CS0)
        f.write("Eigenvalues divided by 2 :\nvp ")
        for k in range(len(S0vp)):
            f.write(" "*(9-len("{:3.3f}".format(S0vp[k]/2)))+"{:3.3f}".format(S0vp[k]/2))
    return
