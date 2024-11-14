#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt



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
    ind=int(B[k][0])-1
    ind2=int(B[k][1])-1
    # print(ind,ind2)
    Vec=(Pos[ind2]-Pos[ind])/np.linalg.norm((Pos[ind]-Pos[ind2]))

    #In order to create an orthonormal basis, the strategy is : taking two vectors linked to two adjacent atoms that are not colinear, one can claculate the cross product wihch gives a perpendicular vector. And by taking the cross product with the last vector and one of the two first, the result is an orthonormal basis that is centered on the atom of interest
    for j in B:
        if int(j[0])-1==ind and int(j[1])-1!=ind2:
            num=int(j[1])-1
            Dist=(Pos[ind]-Pos[num])/np.linalg.norm((Pos[ind]-Pos[num]))
            angl=find_angle(Dist,Vec)
            # print(angl)
            if angl<0: Dist=-Dist
            if abs(Dist.dot(Vec))<0.95:
                per=np.cross(Vec,Dist)
                # if ind==0 and ind2==5:
                #     print(Vec,Dist,np.cross(Vec,per),abs(Dist.dot(Vec)))
                return per,np.cross(Vec,per)

        if int(j[1])-1==ind and int(j[0])-1!=ind2:
            num=int(j[0])-1
            Dist=(Pos[ind]-Pos[num])/np.linalg.norm((Pos[ind]-Pos[num]))
            angl=find_angle(Dist,Vec)
            # print(angl)
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
    return per,np.cross(Vec,per)#linear


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


def bonds_plotting(ax,Pos,bonds,Vec,factor=1,mode='1'):
    """
    Plot the bonds
    """
    #initial values for the parameters of the bonds
    Radbond=0.05
    u=np.linspace(0,2*np.pi,30)#base of the cylinder
    v=np.linspace(0,np.pi,2) #height of the cylinder


    for k in range(len(B)):
        one,two=int(bonds[k][0])-1,int(bonds[k][1])-1
        order = bonds[k][2]
        Vect=Pos[one]-Pos[two]

        if mode=="1": #Drawn
            if order==1: ax.plot([Pos[one][0],Pos[two][0]],[Pos[one][1],Pos[two][1]],[Pos[one][2],Pos[two][2]],"gray",linewidth=4)
            elif order==2:
                zpe,pe=orthonormal_basis(Pos,B,k)
                pe=pe/np.linalg.norm(pe)
                pe/=15
                # dX,dY,dZ=Pos[one]-Pos[two]
                ax.plot([Pos[one][0]-pe[0],Pos[two][0]-pe[0]],[Pos[one][1]-pe[1],Pos[two][1]-pe[1]],[Pos[one][2]-pe[2],Pos[two][2]-pe[2]],"gray",linewidth=4)
                ax.plot([Pos[one][0]+pe[0],Pos[two][0]+pe[0]],[Pos[one][1]+pe[1],Pos[two][1]+pe[1]],[Pos[one][2]+pe[2],Pos[two][2]+pe[2]],"gray",linewidth=4)

            else:
                zpe,pe=orthonormal_basis(Pos,B,k)
                pe=pe/np.linalg.norm(pe)/12
                ax.plot([Pos[one][0]-pe[0],Pos[two][0]-pe[0]],[Pos[one][1]-pe[1],Pos[two][1]-pe[1]],[Pos[one][2]-pe[2],Pos[two][2]-pe[2]],"gray",linewidth=4)
                ax.plot([Pos[one][0]+pe[0],Pos[two][0]+pe[0]],[Pos[one][1]+pe[1],Pos[two][1]+pe[1]],[Pos[one][2]+pe[2],Pos[two][2]+pe[2]],"gray",linewidth=4)
                ax.plot([Pos[one][0],Pos[two][0]],[Pos[one][1],Pos[two][1]],[Pos[one][2],Pos[two][2]],"gray",linewidth=4)

        elif mode=="2": #Physical cylinder
            dist=np.linalg.norm(Pos[one]-Pos[two])

            x=Radbond*(np.outer(np.cos(u),np.ones(np.size(v))))
            y=Radbond*(np.outer(np.sin(u),np.ones(np.size(v))))
            z=(np.outer(np.ones(np.size(u)),np.linspace((abs(Vec[one]/factor)-1/20),(dist-abs(Vec[two]/factor)+1/20),np.size(v))))
            x,y,z=rota_bonds(Vect,x,y,z)
            x,y,z=x-Pos[one][0],y-Pos[one][1],z-Pos[one][2]
            if order==1: ax.plot_surface(x,y,z,color="gray")

            elif order==1.5:
                zpe,pe=orthonormal_basis(Pos,B,k)#Get a orthonormal vector in order to distance the two cylinders
                pe=pe/np.linalg.norm(pe)/15
                ax.plot_surface(x-pe[0],y-pe[1],z-pe[2],color="gray")
                ax.plot_surface(x+pe[0],y+pe[1],z+pe[2],color="white")

            elif order==2:
                zpe,pe=orthonormal_basis(Pos,B,k)
                pe=pe/np.linalg.norm(pe)/15
                ax.plot_surface(x-pe[0],y-pe[1],z-pe[2],color="gray")
                ax.plot_surface(x+pe[0],y+pe[1],z+pe[2],color="gray")

            else:
                zpe,pe=orthonormal_basis(Pos,B,k)
                pe=pe/np.linalg.norm(pe)/12
                ax.plot_surface(x-pe[0],y-pe[1],z-pe[2],color="gray")
                ax.plot_surface(x+pe[0],y+pe[1],z+pe[2],color="gray")
                ax.plot_surface(x,y,z,color="gray")




def plot_3d(Pos,Rad,V,vp,num_vec,Name,B,Ord,file,save=False,mode="2"):
    """
    Do the 3d plot of a vector V
    """
    Pos[:,0]-=np.mean(Pos[:,0]) #Center the Pos
    Pos[:,1]-=np.mean(Pos[:,1])
    Pos[:,2]-=np.mean(Pos[:,2])
    if save>=0:
        fig=plt.figure(figsize=(20,10),dpi=300)
    else:
        fig=plt.figure()
    ax=fig.add_subplot(projection="3d")

    Vec=V[:,num_vec]
    if Vec[0]<0:
        Vec=-Vec #Fix the sign of the first atom to be always positive


    #rotates the molecule to be perpendicular to the default projection
    Zmean=np.array([0.,0.,0.])
    Zdefault=np.array([1.,-1.,1.])#The default projection from matplotlib
    for k in range(len(B)):
        Dist,zpe,ype=orthonormal_basis(Pos,B,k)
        Zmean+=zpe
    Zmean=Zmean/np.linalg.norm(Zmean)
    Zdefault=Zdefault/np.linalg.norm(Zdefault)

    Pos,Rota=rota_mol(Pos,Zdefault,Zmean)

    fact=1.3#Factor reduce radius
    bonds_plotting(ax,Pos,B,Ord,Vec,mode,fact,Rota)
    u=np.linspace(0,2*np.pi,20)#Parameters for the spheres
    v=np.linspace(0,np.pi,15)
    for k in range(len(Pos)):#Draw the spheres
        x=Vec[k]/fact*(np.outer(np.cos(u),np.sin(v)))-Pos[k][0]
        y=Vec[k]/fact*(np.outer(np.sin(u),np.sin(v)))-Pos[k][1]
        z=Vec[k]/fact*(np.outer(np.ones(np.size(u)),np.cos(v)))-Pos[k][2]
        ax.plot_surface(x,y,z,color=["r","w"][(Vec[k]>0)*1],label=Name[k][0]+Name[k][1]+":{:3.2f}".format(Vec[k]))

    plt.legend(loc='center left', bbox_to_anchor=(1.07, 0.5))
    plt.title('Eigenvector n°{}, '.format(num_vec+1)+r"$\mathrm{\lambda}$"+" = {:3.2f}".format(vp[num_vec]))
    ax.set_xlim(np.min(Pos),np.max(Pos))
    ax.set_ylim(np.min(Pos),np.max(Pos))
    ax.set_zlim(np.min(Pos),np.max(Pos))
    ax.set_aspect('equal')
    if save==0:
        plt.savefig(file+".png")
    elif save==1:
        plt.savefig(file+".svg")
    else: plt.show()
    plt.close()




def plot_all_3d(Pos,Rad,V,vp,Name,B,Ord,file,save,mode="2"):
    """
    Do the 3d plot of a vector V
    """
    fact=1.3#Factor reduce radius 1.3 seems to be a good value
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['svg.fonttype'] = 'none'
    Pos[:,0]-=np.mean(Pos[:,0]) #Center the Pos
    Pos[:,1]-=np.mean(Pos[:,1])
    Pos[:,2]-=np.mean(Pos[:,2])
    minV,maxV=np.min(V), np.max(V)
    if save>=0:
        fig=plt.figure(figsize=(20,10),dpi=300)
    else:
        fig=plt.figure()
    u=np.linspace(0,2*np.pi,20)
    v=np.linspace(0,np.pi,15)
    nrow=int(round(len(V)**(1/2)))
    ncol=int(len(V)/nrow+((len(V)%nrow)!=0))

    #rotates the molecule to be perpendicual to the default projection
    Zmean=np.array([0.,0.,0.])
    Zdefault=np.array([1.,-1.,1.])#The default projection from matplotlib
    if len(Ord)!=0: #If there are bonds, rotate the molecule
        # Mord=np.max(Ord)
        for k in range(len(B)):
            # if Ord[k]==Mord:
            zpe,ype=orthonormal_basis(Pos,B,k)
            Zmean+=zpe
        Zmean=Zmean/np.linalg.norm(Zmean)
        Zdefault=Zdefault/np.linalg.norm(Zdefault)

        Pos,Rota=rota_mol(Pos,Zdefault,Zmean)

    for num_vec in range(len(V)):
        Vec=V[:,num_vec]
        if Vec[0]<0:
            Vec=-Vec #Fix the sign of the first atom to be always positive

        ax=fig.add_subplot(nrow,ncol,num_vec+1,projection="3d")


        bonds_plotting(ax,Pos,B,Ord,Vec,mode,fact)
        u=np.linspace(0,2*np.pi,20)
        v=np.linspace(0,np.pi,15)
        for k in range(len(Pos)):
            x=Vec[k]/fact*(np.outer(np.cos(u),np.sin(v)))-Pos[k][0]
            y=Vec[k]/fact*(np.outer(np.sin(u),np.sin(v)))-Pos[k][1]
            z=Vec[k]/fact*(np.outer(np.ones(np.size(u)),np.cos(v)))-Pos[k][2]
            ax.plot_surface(x,y,z,color=["r","w"][(Vec[k]>0)*1],label=Name[k][0]+Name[k][1]+":{:3.2f}".format(Vec[k]))

        # ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5))
        ax.set_xlim(np.min(Pos),np.max(Pos))
        ax.set_ylim(np.min(Pos),np.max(Pos))
        ax.set_zlim(np.min(Pos),np.max(Pos))
        ax.set_aspect('equal')
        ax.set_title('Eigenvector n°{}, '.format(num_vec+1)+r"$\mathrm{\lambda}$"+" = {:3.2f}".format(vp[num_vec]))
    if save==0:
        plt.savefig(file+".png")
    elif save==1:
        plt.savefig(file+".svg")
    else: plt.show()
    plt.close()



def plot_line_3d(Pos,Rad,V,vp,Name,B,Ord,file,nrow,ncol,row,fig,modevec,Rota_arb=0,Scale=[],List_name=[],mode="2"):
    """
    Do the 3d plot of a vector V on a line does not save
    This function is primarely used in the file mult_extrac_adf
    """
    if modevec=="1":
        fact=1.3#Factor reduce radius 1.3 seems to be a good value
    if modevec=="3":
        fact=4*1.3
        V=V*V*abs(vp)
    Pos[:,0]-=np.mean(Pos[:,0]) #Center the Pos
    Pos[:,1]-=np.mean(Pos[:,1])
    Pos[:,2]-=np.mean(Pos[:,2])

    #rotates the molecule to be perpendicular to the default projection
    Zmean=np.array([0.,0.,0.])
    Zdefault=np.array([0.,1.,0.])#The default projection axis from matplotlib
    if len(Ord)!=0: #If there are bonds, rotate the molecule
        # Mord=np.max(Ord)
        # Mord=2
        for k in range(len(B)):
            # if Ord[k]==Mord:
            zpe,ype=orthonormal_basis(Pos,B,k)
            Zmean+=zpe
        Zmean=Zmean/np.linalg.norm(Zmean)
        Zdefault=Zdefault/np.linalg.norm(Zdefault)
        for k in range(len(B)):
            zpe,ype=orthonormal_basis(Pos,B,k)

        Pos,Rota=rota_mol(Pos,Zdefault,Zmean)
        for k in range(len(Pos)):
            Pos[k]=Rota.transpose().dot(Pos[k])#Rotation to xyz

        #Scale the molecule ONLY USE 1 or -1 TO MIRROR
        if Scale!=[]:
            Pos[:,0]=Scale[row][0]*Pos[:,0]
            Pos[:,1]=Scale[row][1]*Pos[:,1]
            Pos[:,2]=Scale[row][2]*Pos[:,2]

        for k in range(len(Pos)):
            Pos[k]=Rota.dot(Rz(Rota_arb).dot(Pos[k])) #Rotation around the z axis and go back on the 1 -1 1 axis
    ncolmem=ncol
    for num_vec in range(len(V[0])):
        Vec=V[:,num_vec]
        if Vec[0]<0:
            Vec=-Vec #Fix the sign of the first atom to always be positive
        # if row==0 and num_vec==4:
        #     Vec=-Vec
        ncol=len(V[0])
        num_vec2=num_vec
        if num_vec==0:
            ncol=ncolmem
        if num_vec==len(V)-1:
            ncol=ncolmem
            num_vec2=ncol-1
        # if modevec!="3":
        ax=fig.add_subplot(nrow,ncol,num_vec2+1+row*ncol,projection="3d")
        # if modevec=="3":
        #     ax=fig.add_subplot(1,nrow,row+1,projection="3d")
        ax.grid(False)
        ax.set_xticks([])#remove ticks
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=0,azim=-90)

        bonds_plotting(ax,Pos,B,Ord,Vec,mode,fact)
        u=np.linspace(0,2*np.pi,60)
        v=np.linspace(0,np.pi,40)
        for k in range(len(Pos)):
            x=Vec[k]/fact*(np.outer(np.cos(u),np.sin(v)))-Pos[k][0]
            y=Vec[k]/fact*(np.outer(np.sin(u),np.sin(v)))-Pos[k][1]
            z=Vec[k]/fact*(np.outer(np.ones(np.size(u)),np.cos(v)))-Pos[k][2]
            ax.plot_surface(x,y,z,color=["r","w"][(Vec[k]>0)*1],label=Name[k][0]+Name[k][1]+":{:3.2f}".format(Vec[k]))

        # ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5))
        ax.set_xlim(np.min(Pos)-np.max(abs(Vec))/fact/2,np.max(Pos)+np.max(abs(Vec))/fact/2)
        ax.set_ylim(np.min(Pos)-np.max(abs(Vec))/fact/2,np.max(Pos)+np.max(abs(Vec))/fact/2)
        ax.set_zlim(np.min(Pos)-np.max(abs(Vec))/fact/2,np.max(Pos)+np.max(abs(Vec))/fact/2)
        ax.set_aspect('equal')
        # ax.set_title('Eigenvector n°{}, '.format(num_vec+1)+r"$\mathrm{\lambda}$"+"$_{}$".format(num_vec+1)+" = {:3.2f}".format(vp[num_vec]),y=1.0,pad=-6)
        ax.set_title(r"$\mathrm{\lambda}$"+"$_{}$".format(num_vec+1)+" = {:3.2f} a.u.".format(vp[num_vec]),y=1.0,pad=-6)
        # if modevec=="3": break
    # if modevec!="3":
    ax=fig.add_subplot(nrow,1,row+1)
    # if modevec=="3":
    #     ax=fig.add_subplot(nrow,1,row+1)
    if List_name==[]:
        title=file.split("//")[-1]
        title=title[0].upper()+title[1:-1]
    else: title=List_name[row]
    ax.set_title("{}".format(title),fontsize=20)
    # ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_color('black') #add box
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set_facecolor("none")




def plot_one_vec_3d(Pos,Rad,V,vp,nvec,Name,B,Ord,file,nrow,ncol,col,fig,modevec,Rota_arb=0,Scale=[],List_name=[],mode="2"):
    """
    Do the 3d plot of a vector V on a line does not save
    This function is primarely used in the file mult_extrac_adf
    """
    Vec=V[:,nvec]
    vp=vp[nvec]
    if modevec=="1":
        fact=1.3#Factor reduce radius 1.3 seems to be a good value
    if modevec=="3":
        fact=5*1.3
        Vec=Vec*vp
    Pos[:,0]-=np.mean(Pos[:,0]) #Center the Pos
    Pos[:,1]-=np.mean(Pos[:,1])
    Pos[:,2]-=np.mean(Pos[:,2])

    #rotates the molecule to be perpendicular to the default projection
    Zmean=np.array([0.,0.,0.])
    Zdefault=np.array([1.,-1.,1.])#The default projection axis from matplotlib
    # Mord=np.max(Ord)
    for k in range(len(B)):
        # if Ord[k]==Mord:
        zpe,ype=orthonormal_basis(Pos,B,k)
        Zmean+=zpe
    Zmean=Zmean/np.linalg.norm(Zmean)
    Zdefault=Zdefault/np.linalg.norm(Zdefault)

    Pos,Rota=rota_mol(Pos,Zdefault,Zmean)

    for k in range(len(Pos)):
        Pos[k]=Rota.dot(Rz(Rota_arb).dot(Rota.transpose().dot(Pos[k])))

    if Vec[0]<0:
        Vec=-Vec #Fix the sign of the first atom to always be positive

    ax=fig.add_subplot(nrow,ncol,col+1,projection="3d")
    ax.grid(False)
    ax.set_xticks([])#remove ticks
    ax.set_yticks([])
    ax.set_zticks([])


    bonds_plotting(ax,Pos,B,Ord,Vec,mode,fact)
    u=np.linspace(0,2*np.pi,60)
    v=np.linspace(0,np.pi,40)
    for k in range(len(Pos)):
        x=Vec[k]/fact*(np.outer(np.cos(u),np.sin(v)))-Pos[k][0]
        y=Vec[k]/fact*(np.outer(np.sin(u),np.sin(v)))-Pos[k][1]
        z=Vec[k]/fact*(np.outer(np.ones(np.size(u)),np.cos(v)))-Pos[k][2]
        ax.plot_surface(x,y,z,color=["r","w"][(Vec[k]>0)*1],label=Name[k][0]+Name[k][1]+":{:3.2f}".format(Vec[k]))

    # ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5))
    ax.set_xlim(np.min(Pos)-np.max(abs(Vec))/fact/2,np.max(Pos)+np.max(abs(Vec))/fact/2)
    ax.set_ylim(np.min(Pos)-np.max(abs(Vec))/fact/2,np.max(Pos)+np.max(abs(Vec))/fact/2)
    ax.set_zlim(np.min(Pos)-np.max(abs(Vec))/fact/2,np.max(Pos)+np.max(abs(Vec))/fact/2)
    ax.set_aspect('equal')
    if List_name==[]:
        title=file.split("//")[-1]
        title=title[0].upper()+title[1:-1]
    else: title=List_name[col]
    ax.set_title('{}, '.format(title)+r"$\mathrm{\lambda}$"+" = {:3.2f}".format(vp),y=1.0,pad=-6)

def plot_atom(ax,atom,plotted_property="radius",transparency=1,factor=1):
    """plot atom as a sphere"""

    Norm = atom.property[plotted_property]
    Pos = atom.pos

    u=np.linspace(0,2*np.pi,20)
    v=np.linspace(0,np.pi,15)

    x=Norm*factor*(np.outer(np.cos(u),np.sin(v)))-Pos[0]
    y=Norm*factor*(np.outer(np.sin(u),np.sin(v)))-Pos[1]
    z=Norm*factor*(np.outer(np.ones(np.size(u)),np.cos(v)))-Pos[2]
    ax.plot_surface(x,y,z,color=atom.color,alpha=transparency)


