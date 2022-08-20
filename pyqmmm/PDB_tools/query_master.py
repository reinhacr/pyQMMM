import collections
import logging
import pathlib
import time
import warnings
import datetime

import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import redo
import requests_cache
import pypdb
import biotite.database.rcsb as rcsb
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG=True

# Code requires defined paths, and a list of PDB IDs in a file. One per line, no extra characters currently, aka 
"1B4U
1BOU
...
..."

# This code was adapted from https://projects.volkamerlab.org/teachopencadd/talktorials/T008_query_pdb.html

# Our goal here is to pass a list of PDBs I have curated from the PDB and
# collect useful information for them. Further geometric analysis will be done by
# molSimplify. We may also grab useful things from the validation reports.
#Please see the readme for required packages

# Current issues- test ligand drawing/ligand parsing. A little buggy for me. CRR

# Cache requests -- this will speed up repeated queries to PDB
requests_cache.install_cache("rcsb_pdb", backend="memory")

# Define paths
(pathlib.Path().absolute())
HERE = pathlib.Path().absolute()
DATA=Path(HERE, 'data')

# Either define an array of PDBs or pass a file. Be very careful of whitespace/
# nonstandard encoding. I had PDB IDs in an excel spreadsheet and exported that
# to a txt or csv, and it was buggy until i removed the spaces at the end and
# got rid of nonstandard formatting


#pdb_ids = []
with open("trunc2.csv") as file_name:
    pdb_ids = np.loadtxt(file_name, dtype="str", delimiter=" ")

## PDB metadata dump
@redo.retriable(attempts=10, sleeptime=5)
def describe_one_pdb_id(pdb_id):
    """Fetch meta information from PDB."""
    described = pypdb.describe_pdb(pdb_id)
    if described is None:
        print(f"! Error while fetching {pdb_id}, retrying ...")
        raise ValueError(f"Could not fetch PDB id {pdb_id}")
    return described

# Collect the data. Use a progress bar for sanity
pdbs_data = [describe_one_pdb_id(pdb_id) for pdb_id in tqdm(pdb_ids)]
print(pdbs_data[0]["struct"])
print(pdbs_data[0]["refine"])
# Pull out properties we are interested in from the metadata PyPdb collected for us
# Collecting structure keywords, text associated with structure, resolution,
#title of citation and disulfide bond info currently, along with some more features

# To Do List: get DOIs, get ligands, organism, etc
# Additionally, should write data to one single file and add plotting functionality using matplotlib just to 
# limit amount of external processing user has to do
# Will turn off excessive printing of data to terminal, this is mainly for testing currently. 

# This is a keyword, i.e "DIOXYGENASE" or "OXIDOREDUCTASE"
PDB_keywords = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["struct_keywords"]["pdbx_keywords"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "pdbx_keywords"],
)
display(PDB_keywords)
PDB_keywords.to_csv(DATA / "PDB_keywords.csv", header=True, index=False)

# This is text associated with the entry, i.e "EXTRADIOL TYPE DIOXYGENASE, PROTOCATECHUATE"
struct_keywords_text = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["struct_keywords"]["text"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "text"],
)
display(struct_keywords_text)
struct_keywords_text.to_csv(DATA / "struct_keywords_text.csv", header=True, index=False)

# this is the descriptor associated with the molecule, i.e "PROTOCATECHUATE 4,5-DIOXYGENASE, 3,4-DIHYDROXYBENZOIC ACID"
PDBx_descriptor = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["struct"]["pdbx_descriptor"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "pdbx_descriptor"],
)
display(PDBx_descriptor)
PDBx_descriptor.to_csv(DATA / "pdbx_descriptor.csv", header=True, index=False)

# this is the citation title. i.e "Crystal structure of an aromatic ring opening dioxygenase LigAB, a protocatechuate 4,5-dioxygenase, under aerobic conditions."
# i.e ""
citation_title = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_primary_citation"]["title"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "title"],
)
display(citation_title)
citation_title.to_csv(DATA / "citation_titles.csv", header=True, index=False)

# S-S bonds. Collecting for modeling purposes. Example output "O" or "1".."N" etc
ss_bonds = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["disulfide_bond_count"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "disulfide_bond_count"],
)
display(ss_bonds)
ss_bonds.to_csv(DATA / "ss_bond_count.csv", header=True, index=False)

# Resolution reported on main PDB page
resolution = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["resolution_combined"][0]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "resolution"],
)
display(resolution)
resolution.to_csv(DATA / "PDB_resolution.csv", header=True, index=False)

# inter_mol_metalic_bond_count.

metal_mol_bonds = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["inter_mol_metalic_bond_count"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "inter_mol_metalic_bond_count"],
)
display(metal_mol_bonds)
metal_mol_bonds.to_csv(DATA / "PDB_metal_mol_bonds.csv", header=True, index=False)

# Protein size information of largest chain. here is MW. i.e "33.32" (units are kilodaltons)
# To get whole complex MW will need to query something else (need to check) or do the math

protein_mw_max = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["polymer_molecular_weight_maximum"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "polymer_molecular_weight_maximum"],
)
display(protein_mw_max)
protein_mw_max.to_csv(DATA / "protein_mw_max.csv", header=True, index=False)

# Protein size information of smallest chain. here is MW. i.e "15.57" (units are kilodaltons)
# To get whole complex MW will need to query something else (need to check) or do the math

protein_mw_min = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["polymer_molecular_weight_minimum"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "polymer_molecular_weight_minimum"],
)
display(protein_mw_min)
protein_mw_min.to_csv(DATA / "protein_mw_min.csv", header=True, index=False)

# Fetch fasta sequence.
for pdb_id in pdb_ids:
    rcsb.fetch([pdb_id], "fasta", './data/fasta/')

## Ligand information isn't working yet
