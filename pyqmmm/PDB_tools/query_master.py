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

# Pull out properties we are interested in from the metadata PyPdb collected for us
# Collecting structure keywords, text associated with structure, resolution,
#title of citation and disulfide bond info currently
# To Do List: Break down citation information further, get ligands, organism, etc

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

# this is the descriptor associated with the molecule, i.e "PROTOCATECHUATE 4,5-DIOXYGENASE, ..."
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
# turn off print statements later, mainly for testing purposes currently
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

# Polymer entity count. Will provide us the number of unique protein chains.
# Example 1BOU returns "2" because there is both an alpha and beta chain
# This information is useful as there may be mutliple metal sites and the
# metal site could only be in one protein. For example, 1BOU has two metal sites,
# both in beta chain but not alpha

polymer_count = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["polymer_entity_count"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "polymer_entity_count"],
)
display(polymer_count)
polymer_count.to_csv(DATA / "polymer_entity_count.csv", header=True, index=False)

# Nonpolymer entity count, i.e number of unique ligands or non-protein components.
# Example "2" for 1B4U, ligand plus Fe

nonpolymer_count = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["nonpolymer_entity_count"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "nonpolymer_entity_count"],
)
display(nonpolymer_count)
nonpolymer_count.to_csv(DATA / "nonpolymer_count.csv", header=True, index=False)

# deposited_unmodeled_polymer_monomer_count. Reports back number of missing residues
# Example "1B4U                22" meainng 22 AAs were not in PDB structures

number_of_missing_AAs = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["deposited_unmodeled_polymer_monomer_count"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "deposited_unmodeled_polymer_monomer_count"],
)
display(number_of_missing_AAs)
number_of_missing_AAs.to_csv(DATA / "number_of_missing_AAs.csv", header=True, index=False)

# Polymer composition. Is it a monomer? Dimer? Heterodimer, etc.
# Example output: "1B4U  heteromeric protein"

polymer_composition = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["polymer_composition"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "polymer_composition"],
)
display(polymer_composition)
polymer_composition.to_csv(DATA / "polymer_composition.csv", header=True, index=False)

# Atom count of deposited PDB.
atom_count = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["deposited_atom_count"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "deposited_atom_count"],
)
display(atom_count)
atom_count.to_csv(DATA / "atom_count.csv", header=True, index=False)

# Atom count of deposited PDB.
pubmed_id = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_container_identifiers"]["pubmed_id"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "pubmed_id"],
)
display(pubmed_id)
pubmed_id.to_csv(DATA / "pubmed_id.csv", header=True, index=False)

# ligand names obtained by getting res types no checked for bond angle geomtry.
#This is typically non standard residues, so will include iron, etc
ligand_names = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["pdbx_vrpt_summary"]["restypes_notchecked_for_bond_angle_geometry"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "restypes_notchecked_for_bond_angle_geometry"],
)
display(ligand_names)
ligand_names.to_csv(DATA / "ligand_names.csv", header=True, index=False)

# Fetch fasta sequence.
for pdb_id in pdb_ids:
    rcsb.fetch([pdb_id], "fasta", './data/fasta/')

## Ligand information isn't working yet
