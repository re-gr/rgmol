from mpl_toolkits.mplot3d import Axes3D
import codecs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import argparse
import os
import extrac_adf as eadf
Rota_arb=[]
List_out=["H2","H3"]

q=(1/2)**(1/2)

H1=np.array([[0.,0.,0.]])

H2 = np.array([[0.,0,0],[1,0,0]])
H22 = np.array([[0.,0,0],[1,0,0]])
H3 = np.array([[0.,0,0],[1,0,0],[1/2,0,(3)**(1/2)/2]])

CH3=np.array([[0.,0,0],[1,0,0],[1/2,0,(3)**(1/2)/2],[1/2,1/3,(3)**(1/2)/4]])
CH32=np.array([[0.,0,(3)**(1/2)/2],[1,0,(3)**(1/2)/2],[1/2,0,0],[1/2,-1/3,(3)**(1/2)/4]])

CH3=CH3.dot(eadf.Rz(-1.4))
CH32=CH32.dot(eadf.Rz(-1.4))
d=1/1.5

C2H6 = np.array([[0.,0,0],[1,0,0],[1/2,0,(3)**(1/2)/2],[1/2,1/3,(3)**(1/2)/4],[0.,d+2/3,(3)**(1/2)/2],[1,d+2/3,(3)**(1/2)/2],[1/2,d+2/3,0],[1/2,1/3+d,(3)**(1/2)/4]])
C2H6=C2H6.dot(eadf.Rz(-1.4))


X1V=np.array([[1/2]])
X1vp=np.array([-1])
X2V = np.array([[1.,1],[-1,1]])
X2V2 = np.array([[1.2,1],[-.8,1]])
X2V/=np.linalg.norm(X2V)
X2V2/=np.linalg.norm(X2V2)
X2vp = np.array([-1.,0])
X3V = np.array([[1/2,1,1],[1/2,-1,1],[-1,0,1]])
X3V/=np.linalg.norm(X3V)
X3vp = np.array([-1,-1.,0])

X4V = np.array([[1/3,1/2,1,1],[1/3,1/2,-1,1],[1/3,-1,0,1],[-1,0,0,1]])
X4V/=np.linalg.norm(X4V)

X4vp=np.array([-3,-1,-1,0])
X8V = np.array([[1/3,1/3,1/2,1,1/2,1,1,1],[1/3,1/3,1/2,-1,1/2,-1,1,1],[1/3,1/3,-1,0,-1,0,1,1],[-1,-1,0,0,0,0,1,1],[-1/3,1/3,1/2,-1,-1/2,1,-1,1],[-1/3,1/3,1/2,1,-1/2,-1,-1,1],[-1/3,1/3,-1,0,1,0,-1,1],[1,-1,0,0,0,0,-1,1]])
print(np.linalg.norm(X8V))
X8V/=3
X8vp=np.array([-5,-4,-2,-2,-1,-1,-0.5,0])

BH2=[[1,2]]
OrdH2=[1]

BH3=[]
OrdH3=[]

BCH3=[[1,4],[2,4],[3,4]]
OrdCH3=[1,1,1]

BC2H6=[[1,4],[2,4],[3,4],[4,8],[5,8],[6,8],[7,8]]
OrdC2H6=[1,1,1,1,1,1,1]

ncols,nrows=2,2



# Zdefault=np.array([1.,-1.,1.])
# Zdefault=Zdefault/np.linalg.norm(Zdefault)
#
# H2,R = eadf.rota_mol(H2,Zdefault,np.array([0,0,1]))
# H3,R = eadf.rota_mol(H3,Zdefault,np.array([0,0,1]))


fig=plt.figure(figsize=(3.1*2*ncols,2.8*2*nrows),dpi=200) #x: ~3.1 for 1 vector; y: ~2.8 for one molecule
plt.rcParams.update({'font.size': 13})
plt.rcParams['svg.fonttype'] = 'none'

#remove grid background
plt.rcParams['axes.edgecolor'] = 'none'
plt.rcParams['axes3d.xaxis.panecolor'] = 'none'
plt.rcParams['axes3d.yaxis.panecolor'] = 'none'
plt.rcParams['axes3d.zaxis.panecolor'] = 'none'


file="fragments"
vec="X"

# eadf.plot_line_3d(H1,0,X1V,X1vp,[["1",'H']],BH2,OrdH2,"X",nrows,ncols,0,fig,"1",0,[])
eadf.plot_line_3d(H2,0,X2V,X2vp,[["1",'H'],['2','H']],BH2,OrdH2,"X",nrows,ncols,0,fig,"1",0,[])
eadf.plot_line_3d(H22,0,X2V2,X2vp,[["1",'H'],['2','H']],BH2,OrdH2,"X",nrows,ncols,1,fig,"1",0,[])

# eadf.plot_line_3d(H3,0,X3V,X3vp,[["1",'H'],['2','H'],["3","H"]],BH3,OrdH3,"X",nrows,ncols,2,fig,"1",0,[])
# eadf.plot_line_3d(CH3,0,X4V,X4vp,[["1",'H'],['2','H'],["3","H"],["4","C"]],BCH3,OrdCH3,"X",nrows,ncols,3,fig,"1",0,[])
# eadf.plot_line_3d(CH32,0,X4V,X4vp,[["1",'H'],['2','H'],["3","H"],["4","C"]],BCH3,OrdCH3,"X",nrows,ncols,4,fig,"1",0,[])
# eadf.plot_line_3d(C2H6,0,X8V,X8vp,[["1",'H'],['2','H'],["3","H"],["4","C"],["5",'H'],['6','H'],["7","H"],["8","C"]],BC2H6,OrdC2H6,"X",nrows,ncols,5,fig,"1",0,[])
# plt.show()
plt.savefig(file+vec+".png",transparent=True,bbox_inches="tight", pad_inches=0.1)



