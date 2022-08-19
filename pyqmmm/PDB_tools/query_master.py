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

#from opencadd.structure.superposition.api import align, METHODS
#from opencadd.structure.core import Structure

# This code was adapted from https://projects.volkamerlab.org/teachopencadd/talktorials/T008_query_pdb.html
# Our goal here is to pass a list of PDBs i have curated from the PDB and
#collect useful information for them. Further geometric analysis will be done by
# molSimplify. We may also grab useful things from the validation reports


# Disable some unneeded warnings
logger = logging.getLogger("opencadd")
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Cache requests -- this will speed up repeated queries to PDB
requests_cache.install_cache("rcsb_pdb", backend="memory")

# Define paths
(pathlib.Path().absolute())
HERE = pathlib.Path().absolute()
print(HERE)
DATA=Path(HERE, 'data')
print(DATA)

# Establish a list of PDBs for testing. Later pass it a file to do automaticaly
#pdb_ids = []
with open("trunc.csv") as file_name:
    pdb_ids = np.loadtxt(file_name, dtype="str", delimiter=" ")

#with open('trunc.txt', 'rb') as file:
#    contents = file.read()
#    for l in contents:
#        pdb_ids.append(l.strip())

#pdb_ids = ["6W4X", "1OS7"]
#print(pdb_ids)
## PDB metadata dump
@redo.retriable(attempts=10, sleeptime=5)
def describe_one_pdb_id(pdb_id):
    """Fetch meta information from PDB."""
    described = pypdb.describe_pdb(pdb_id)
    if described is None:
        print(f"! Error while fetching {pdb_id}, retrying ...")
        raise ValueError(f"Could not fetch PDB id {pdb_id}")
    return described

pdbs_data = [describe_one_pdb_id(pdb_id) for pdb_id in tqdm(pdb_ids)]
#print(pdbs_data)
print("\n".join(pdbs_data[0].keys()))
print(pdbs_data[0]["rcsb_primary_citation"])
print(pdbs_data[0]["exptl"])
print(pdbs_data[0]["citation"])
#print(pdbs_data[0][rcsb_primary_citation[0]].keys())
struct_keywords1 = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["struct_keywords"]["pdbx_keywords"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "pdbx_keywords"],
)
display(struct_keywords1)
struct_keywords1.to_csv(DATA / "PDB_keywords.csv", header=True, index=False)

struct_keywords2 = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["struct_keywords"]["text"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "text"],
)
display(struct_keywords2)
struct_keywords2.to_csv(DATA / "PDB_keywords_2.csv", header=True, index=False)

struct_keywords3 = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_primary_citation"]["title"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "title"],
)
display(struct_keywords3)
struct_keywords3.to_csv(DATA / "PDB_keywords_3.csv", header=True, index=False)

struct_keywords4 = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["citation"]["year"]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "year"],
)
display(struct_keywords4)
struct_keywords4.to_csv(DATA / "PDB_keywords_4.csv", header=True, index=False)

#rcsb_primary_citation
resolution = pd.DataFrame(
    [
        [pdb_data["entry"]["id"], pdb_data["rcsb_entry_info"]["resolution_combined"][0]]
        for pdb_data in pdbs_data
    ],
    columns=["pdb_id", "resolution"],
)
display(resolution)
resolution.to_csv(DATA / "PDB_resolution.csv", header=True, index=False)
top_num = len(pdb_ids)
## Ligand information
#def get_ligands(pdb_id):
#    """
#    RCSB has not provided a new endpoint for ligand information yet. As a
#    workaround we are obtaining extra information from ligand-expo.rcsb.org,
#    using HTML parsing. Check Talktorial T011 for more info on this technique!
    #"""
#    pdb_info = _fetch_pdb_nonpolymer_info(pdb_id)
#    ligand_expo_ids = [
#        nonpolymer_entities["pdbx_entity_nonpoly"]["comp_id"]
#        for nonpolymer_entities in pdb_info["data"]["entry"]["nonpolymer_entities"]
#    ]

#    ligands = {}
#    for ligand_expo_id in ligand_expo_ids:
#        ligand_expo_info = _fetch_ligand_expo_info(ligand_expo_id)
#        ligands[ligand_expo_id] = ligand_expo_info

#    return ligands


#def _fetch_pdb_nonpolymer_info(pdb_id):
#    """
#    Fetch nonpolymer data from rcsb.org.
#    Thanks @BJWiley233 and Rachel Green for this GraphQL solution.
#    """
#    query = (
#        """{
 #         entry(entry_id: "%s") {
  #          nonpolymer_entities {
   #           pdbx_entity_nonpoly {
    #            comp_id
     #           name
      #          rcsb_prd_id
     #         }
    #        }
   #       }
  #      }"""
 #       % pdb_id
#    )

#    query_url = f"https://data.rcsb.org/graphql?query={query}"
#    response = requests.get(query_url)
#    response.raise_for_status()
#    info = response.json()
#    return info


#def _fetch_ligand_expo_info(ligand_expo_id):
 #   """
 #   Fetch ligand data from ligand-expo.rcsb.org.
 #   """
 #   r = requests.get(f"http://ligand-expo.rcsb.org/reports/{ligand_expo_id[0]}/{ligand_expo_id}/")
 #   r.raise_for_status()
 #   html = BeautifulSoup(r.text)
 #   info = {}
 #   for table in html.find_all("table"):
 #       for row in table.find_all("tr"):
 #           cells = row.find_all("td")
 #           if len(cells) != 2:
 #               continue
 #           key, value = cells
 #           if key.string and key.string.strip():
 #               info[key.string.strip()] = "".join(value.find_all(string=True))

    # Postprocess some known values
#    info["Molecular weight"] = float(info["Molecular weight"].split()[0])
#    info["Formal charge"] = int(info["Formal charge"])
#    info["Atom count"] = int(info["Atom count"])
#    info["Chiral atom count"] = int(info["Chiral atom count"])
#    info["Bond count"] = int(info["Bond count"])
#    info["Aromatic bond count"] = int(info["Aromatic bond count"])
#    return info

#columns = [
#    "@structureId",
#    "@chemicalID",
#    "@type",
#    "@molecularWeight",
#    "chemicalName",
#    "formula",
#    "InChI",
#    "InChIKey",
#    "smiles",
#]
#rows = []
#for pdb_id in pdb_ids:
  #  ligands = get_ligands(pdb_id)
#    # If several ligands contained, take largest (first in results) # going to fix this later
 #   ligand_id, properties = max(ligands.items(), key=lambda kv: kv[1]["Molecular weight"])
 #       #this_ligand = len(ligands)
 #   rows.append(
 ##   [
 #   pdb_id,
 #   ligand_id,
 #   properties["Component type"],
 #   properties["Molecular weight"],
 #   properties["Name"],
 #   properties["Formula"],
 #   properties["InChI descriptor"],
 #   properties["InChIKey descriptor"],
  #  properties["Stereo SMILES (OpenEye)"],
 #   ]
 #       )
# NBVAL_CHECK_OUTPUT
# Change the format to DataFrame
#ligands = pd.DataFrame(rows, columns=columns)
#ligands
#display(ligands)
#ligands.to_csv(DATA / "PDB_top_ligands.csv", header=True, index=False)

#PandasTools.AddMoleculeColumnToFrame(ligands, "smiles")
#Draw.MolsToGridImage(
#    mols = list(ligands.ROMol),
#    legends = list(ligands["@chemicalID"] + ", " + ligands["@structureId"]),
#    molsPerRow = top_num,
#)

    # NBVAL_CHECK_OUTPUT
#pairs = collections.OrderedDict(zip(ligands["@structureId"], ligands["@chemicalID"]))
#pairs
