#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

File to import that gives the path to output files used as examples.
"""

from importlib import resources as impresources
from rgmol.output_examples.ADF import chloromethane
from rgmol.output_examples.ADF import methane
from rgmol.output_examples.Orca import CH3Cl
from rgmol.output_examples.Orca import chfclbr
from rgmol.output_examples.Orca import formaldehyde
from rgmol.output_examples.Orca import formaldehyde_huge
from rgmol.output_examples import Cube

adf_chloromethane = (impresources.files(chloromethane) / "chloromethane.out").__str__()
adf_methane = (impresources.files(methane) / "methane.out").__str__()
adf_CH3Cl = (impresources.files(chloromethane) / "chloromethane.out").__str__()
adf_CH4 = (impresources.files(methane) / "methane.out").__str__()

orca_chloromethane = (impresources.files(CH3Cl) / "CH3Cl.out").__str__()
molden_chloromethane = (impresources.files(CH3Cl) / "CH3Cl.molden.input").__str__()
orca_CH3Cl = (impresources.files(CH3Cl) / "CH3Cl.out").__str__()
molden_CH3Cl = (impresources.files(CH3Cl) / "CH3Cl.molden.input").__str__()


orca_CHFClBr = (impresources.files(CH3Cl) / "chfclbr.out").__str__()
molden_CHFClBr = (impresources.files(CH3Cl) / "chfclbr.molden.input").__str__()
orca_bromochlorofluoromethane = (impresources.files(CH3Cl) / "chfclbr.out").__str__()
molden_bromochlorofluoromethane = (impresources.files(CH3Cl) / "chfclbr.molden.input").__str__()


orca_formaldehyde = (impresources.files(formaldehyde) / "H2CO.out").__str__()
molden_formaldehyde = (impresources.files(formaldehyde) / "H2CO.molden.input").__str__()
orca_H2CO = (impresources.files(formaldehyde) / "H2CO.out").__str__()
molden_H2CO = (impresources.files(formaldehyde) / "H2CO.molden.input").__str__()



orca_formaldehyde_qz = (impresources.files(formaldehyde_huge) / "H2CO.out").__str__()
molden_formaldehyde_qz = (impresources.files(formaldehyde_huge) / "H2CO.molden.input").__str__()
orca_H2CO_qz = (impresources.files(formaldehyde_huge) / "H2CO.out").__str__()
molden_H2CO_qz = (impresources.files(formaldehyde_huge) / "H2CO.molden.input").__str__()

cube_formaldehyde_MO8 = (impresources.files(Cube) / "H2CO.mo8a.cube").__str__()
cube_formaldehyde_MO59 = (impresources.files(Cube) / "H2CO.mo59a.cube").__str__()
cube_H2CO_MO8 = (impresources.files(Cube) / "H2CO.mo8a.cube").__str__()
cube_H2CO_MO59 = (impresources.files(Cube) / "H2CO.mo59a.cube").__str__()