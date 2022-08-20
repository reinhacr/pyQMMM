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
import nglview
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
with open("trunc.csv") as file_name:
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
# To Do List: Break down citatin information further, get ligands, organism, etc

PDB_keywords = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["struct_keywords"]["pdbx_keywords"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "pdbx_keywords"],
)
display(PDB_keywords)
PDB_keywords.to_csv(DATA / "PDB_keywords.csv", header=True, index=False)

struct_keywords_text = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["struct_keywords"]["text"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "text"],
)
display(struct_keywords_text)
struct_keywords_text.to_csv(DATA / "struct_keywords_text.csv", header=True, index=False)

citation_title = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_primary_citation"]["title"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "title"],
)
display(citation_title)
citation_title.to_csv(DATA / "citation_titles.csv", header=True, index=False)


# Resolution
resolution = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["resolution_combined"][0]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "resolution"],
)
display(resolution)
resolution.to_csv(DATA / "PDB_resolution.csv", header=True, index=False)

# Fetch fasta sequence.


## Ligand information isn't working yet
