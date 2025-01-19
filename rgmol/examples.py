"""
File to import that gives the path to output files used as examples.
"""

from importlib import resources as impresources
from output_examples.ADF import chloromethane
from output_examples.ADF import methane
from output_examples.Orca import CH3Cl
from output_examples.Orca import chfclbr
from output_examples.Orca import formaldehyde
from output_examples.Orca import formaldehyde_huge
from output_examples import Cube

adf_chloromethane = impresources.files(chloromethane) / "chloromethane.out"
adf_methane = impresources.files(methane) / "methane.out"
adf_CH3Cl = impresources.files(chloromethane) / "chloromethane.out"
adf_CH4 = impresources.files(methane) / "methane.out"

orca_chloromethane = impresources.files(CH3Cl) / "CH3Cl.out"
molden_chloromethane = impresources.files(CH3Cl) / "CH3Cl.molden.input"
orca_CH3Cl = impresources.files(CH3Cl) / "CH3Cl.out"
molden_CH3Cl = impresources.files(CH3Cl) / "CH3Cl.molden.input"


orca_CHFClBr = impresources.files(CH3Cl) / "chfclbr.out"
molden_CHFClBr = impresources.files(CH3Cl) / "chfclbr.molden.input"
orca_bromochlorofluoromethane = impresources.files(CH3Cl) / "chfclbr.out"
molden_bromochlorofluoromethane = impresources.files(CH3Cl) / "chfclbr.molden.input"


orca_formaldehyde = impresources.files(formaldehyde) / "H2CO.out"
molden_formaldehyde = impresources.files(formaldehyde) / "H2CO.molden.input"
orca_H2CO = impresources.files(formaldehyde) / "H2CO.out"
molden_H2CO = impresources.files(formaldehyde) / "H2CO.molden.input"



orca_formaldehyde_qz = impresources.files(formaldehyde_huge) / "H2CO.out"
molden_formaldehyde_qz = impresources.files(formaldehyde_huge) / "H2CO.molden.input"
orca_H2CO_qz = impresources.files(formaldehyde_huge) / "H2CO.out"
molden_H2CO_qz = impresources.files(formaldehyde_huge) / "H2CO.molden.input"

cube_formaldehyde_MO8 = impresources.files(Cube) / "H2CO.mo8a.cube"
cube_formaldehyde_MO9 = impresources.files(Cube) / "H2CO.mo9a.cube"
cube_H2CO_MO8 = impresources.files(Cube) / "H2CO.mo8a.cube"
cube_H2CO_MO9 = impresources.files(Cube) / "H2CO.mo9a.cube"