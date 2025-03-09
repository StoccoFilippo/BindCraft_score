import os
import math
import random
import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import json
import tempfile
import glob
from glob import glob
import csv
from collections import defaultdict

from functions import *

import pdbfixer
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import io



# ---------------------------
# Hyperparameters and Config
# ---------------------------
CONFIG = {
    "beta": 0.1,
    "seed": 1998,
    "learning_rate": 1e-6,
    "batch_size": 20,
    "num_epochs": 10,
    "split_percent": 0.2,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,
    "adam_decay": 0.1,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Utility Functions
# ---------------------------
def seed_everything(seed=2003):
    """
    Sets random seed for reproducibility across libraries.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def append_to_csv(
    name,
    pMPNN,
    d_rmsd,
    pAE_bt,
    shape_complimentary,
    uns_hydrogens,
    hydrophobicity,
    binder_score,
    interface_dSASA,
    sequence,
    plddt,
    i_pTM,
    pAE_b,
    ipsae,
    helicity,
    lenght,
    pae,
    ptm,
    has_clash,
    pAE_t,
    pae_2,
    contact_probs,
    output_file
):
    file_exists = os.path.exists(output_file) and os.stat(output_file).st_size > 0
    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "name",
            "pMPNN",
            "d_rmsd",
            "pAE_bt",
            "shape_complimentary",
            "uns_hydrogens",
            "hydrophobicity",
            "binder_score",
            "interface_dSASA",
            "sequence",
            "plddt",
            "i_pTM",
            "pAE_b",
            "ipsae",
            "helicity",
            "lenght",
            "ptm",
            "has_clash",
            "pAE_t",
            "pae_2",
            "contact_probs",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "name": name,
            "pMPNN": pMPNN,
            "d_rmsd": d_rmsd,
            "pAE_bt": pAE_bt,
            "shape_complimentary": shape_complimentary,
            "uns_hydrogens": uns_hydrogens,
            "hydrophobicity": hydrophobicity,
            "binder_score": binder_score,
            "interface_dSASA": interface_dSASA,
            "sequence": sequence,
            "plddt": plddt,
            "i_pTM": i_pTM,
            "pAE_b": pAE_b,
            "ipsae": ipsae,
            "helicity": helicity,
            "lenght": lenght,
            "ptm": ptm,
            "has_clash": has_clash,
            "pAE_t": pAE_t,
            "pae_2": pae_2,
            "contact_probs": contact_probs
        })




def af_metrics(name, path):
    name = name.lower()
    metrics_file = f"{name}_summary_confidences.json"
   
    with open(os.path.join(path, metrics_file), "r") as f:
        metrics_summary = json.load(f)

    with open(os.path.join(path, metrics_file.replace("summary_","")), "r") as f:
        metrics = json.load(f)

    pae = np.array(metrics["pae"])
    ptm = metrics_summary['ptm']
    iptm = metrics_summary['iptm']
    has_clash = metrics_summary['has_clash']
    
    #ipsae = get_ipsae(path, arg1=10, arg2=10)
    ipsae = 1
    
    chain_ids = metrics["token_chain_ids"]  
    atom_ids = metrics["atom_chain_ids"]
    plddt = metrics['atom_plddts']

    chain_ids_binder = [x for x in chain_ids if x == "B"]
    atom_ids_binder = [x for x in atom_ids if x == "B"]
    
    plddt = np.array(plddt[:len(atom_ids_binder)]).mean()

    pae = np.array(metrics["pae"])
    b_pae = pae[len(chain_ids_binder):, :len(chain_ids_binder)].mean()
    t_pae = pae[:len(chain_ids_binder), len(chain_ids_binder):].mean()

    pae_2 = (b_pae.mean() + t_pae.mean()) / 2

    return iptm, pae , plddt, ptm, has_clash, ipsae, b_pae, t_pae, pae_2

def get_chain_indices(chain_ids):

    chain_map = defaultdict(list)
    for i, c in enumerate(chain_ids):
        chain_map[c].append(i)
    return dict(chain_map)


def compute_inter_chain_contacts(contact_probs, chain_map, chain1='A', chain2='B'):

    idx_chain1 = chain_map.get(chain1, [])
    idx_chain2 = chain_map.get(chain2, [])
    
    if not idx_chain1 or not idx_chain2:
        raise ValueError(f"No residues found for chain {chain1} or chain {chain2}!")
    
    # Convert to numpy arrays (just for safety)
    idx_chain1 = np.array(idx_chain1)
    idx_chain2 = np.array(idx_chain2)
    
    # Extract the submatrix of probabilities for chain1 vs chain2
    sub_probs = contact_probs[np.ix_(idx_chain1, idx_chain2)]

    return sub_probs


def load_alphafold_data(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_contact_points(sub_probs, chain_ids):
    contact = []
    for i in range(len(chain_ids)):
        contact.append(sum(sub_probs[:,i]))
        
    return np.array(contact)

def get_contact_probs(sequences_rep, pdb_folder):
    # the contact probabilites for the hotspot residues of the target (A) must be the same 
    for name in sequences_rep:
        name = name.lower()
        json_path = f"./{pdb_folder}/{name}/{name}_confidences.json"
        data = load_alphafold_data(json_path)
        atom_ids = data["atom_chain_ids"] # multiple atoms maps to chain ID
        chain_ids = data["token_chain_ids"] # number of residues 
        chain_ids_target = [x for x in chain_ids if x == "A"]
        contact_probs = np.array(data["pae"])  # shape (N, N)

        chain_map = get_chain_indices(chain_ids)
        sub_probs = compute_inter_chain_contacts(contact_probs, chain_map, chain1='B', chain2='A')

        contact_DNA = compute_contact_points(sub_probs, chain_ids_target)
        hotspot_residues = [18,43,44,46,49,50,53,61,62,64,65,66,68,69,70,72,73,75,76,77,80]

        # Sum elements at the specified indices
        sum_targets = np.sum(contact_DNA[hotspot_residues])
        sum_off_target = np.sum(contact_DNA) - sum_targets
        sequences_rep[name]["contact_probs"] = sum_targets - sum_off_target

    return sequences_rep



def get_pMPNN(pdb_file):
    

    with tempfile.TemporaryDirectory() as output_dir:
       
            command_line_arguments = [
                "python",
                "./functions/protein_mpnn_run.py",
                "--pdb_path", pdb_file,
                "--pdb_path_chains", "B",
                "--score_only", "1",
                "--save_score", "1",
                "--out_folder", output_dir,
                "--batch_size", "1"
            ]

            proc = subprocess.run(command_line_arguments, stdout=subprocess.PIPE, check=True)
            output = proc.stdout.decode('utf-8')
            for x in output.split('\n'):
                if x.startswith('Score for'):
                                name = x.split(',')[0][10:-9]
                                mean =x.split(',')[1].split(':')[1]
    return float(mean)

def convert_cif_to_pdb(cif_file):
    """
    Converts a CIF file to PDB format and returns the PDB string.
    """
    fixer = PDBFixer(cif_file)

    # Handle missing atoms/residues if needed
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    # Store PDB data in a string buffer
    pdb_file = cif_file.replace("cif","pdb")
    with open(pdb_file, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

def get_ipsae(json_path, arg1=10, arg2=10):
    # run ipsae.py
    command = ["python", "./functions/ipsae.py", json_path, str(arg1), str(arg2)]
    subprocess.run(command)
    output_path=""
    with open(output_path, "r") as f:
        ipsae_data = f.read()
    ipsae = 1 
    #Process data

    return ipsae

def calc_ss_percentage(pdb_file):
    # Parse the structure

    chain_id="B"
    atom_distance_cutoff=4.0

    with open("./functions/default_4stage_multimer.json", "r") as f:
        advanced_settings = json.load(f)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Consider only the first model in the structure

    # Calculate DSSP for the model
    dssp = DSSP(model, pdb_file, dssp=advanced_settings["dssp_path"])

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues = set(hotspot_residues(pdb_file, chain_id, atom_distance_cutoff).keys())

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_type = 'loop'
            if ss in ['H', 'G', 'I']:
                ss_type = 'helix'
            elif ss == 'E':
                ss_type = 'sheet'

            ss_counts[ss_type] += 1

            if ss_type != 'loop':
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(total_residues, ss_counts['helix'], ss_counts['sheet'])
    interface_percentages = calculate_percentages(total_interface_residues, ss_interface_counts['helix'], ss_interface_counts['sheet'])

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)

def py_ros_score_interface(pdb_file):
    
    with open("./functions/default_4stage_multimer.json", "r") as f:
        advanced_settings = json.load(f)

    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball "./functions/DAlphaBall.gcc" -corrections::beta_nov16 true -relax:default_repeats 1')
    
    print(f"Scoring interface of {pdb_file}")
    interface_scores, interface_AA, interface_residues_pdb_ids_str = score_interface(pdb_file, binder_chain="B")

    return interface_scores, interface_AA, interface_residues_pdb_ids_str
 

# ---------------------------
# Dataset Generation
# ---------------------------
def score_sequences(fasta_file, pdb_folder, target_pdb):
    
    with open(fasta_file, "r") as f:
        rep_seq = f.readlines()

    sequences_rep = {}
    for line in rep_seq:
        if ">" in line:
            name = line.split("\t")[0].replace(">", "").strip()
        else:
            sequences_rep[name] = {"sequence": line.strip()}
    
    sequences_rep = get_contact_probs(sequences_rep, pdb_folder)
    

    for entry in sequences_rep:
        try:
            name = entry
            sequence = sequences_rep[str(name)]['sequence']
            lenght = len(sequence)
            
            path = f"./{pdb_folder}/{name.lower()}"
            i_pTM, pae , plddt, ptm, has_clash, ipsae, pAE_b, pAE_t, pae_2 = af_metrics(name.lower(), path)
            
            pAE_bt = (pAE_b + pAE_t)/2

            cif_file = path + f"/{name.lower()}_model.cif"
            convert_cif_to_pdb(cif_file)
            pdb_file = path + f"/{name.lower()}_model.pdb"
            
            interface_scores, interface_AA, interface_residues_pdb_ids_str = py_ros_score_interface(pdb_file)

            helicity, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, i_plddt, trajectory_ss_plddt = calc_ss_percentage(pdb_file)
            
            pMPNN = get_pMPNN(pdb_file)
            contact_probs = sequences_rep[str(name)]["contact_probs"]
            shape_complimentary = interface_scores["interface_sc"]
            uns_hydrogens = interface_scores["interface_delta_unsat_hbonds"]
            hydrophobicity = interface_scores["surface_hydrophobicity"]
            binder_score = interface_scores["binder_score"]
            interface_dSASA = interface_scores["interface_dSASA"]
            d_rmsd = target_pdb_rmsd(pdb_file, target_pdb, "A")
            
            append_to_csv(name, -pMPNN, d_rmsd, pAE_bt, shape_complimentary, uns_hydrogens, hydrophobicity,
                            binder_score, interface_dSASA, sequence, plddt, i_pTM,
                            pAE_b, ipsae, helicity, lenght, pae, ptm, has_clash, pAE_t, pae_2,contact_probs, "logs_output.cvs")
            
           
        except Exception as e:
            print(f"Error processing sequence {name}")
            print(e)
   


# ---------------------------
#     MAIN
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file", type=str, required=True)
    parser.add_argument("--pdb_folder", type=str, required=True)
    parser.add_argument("--target_pdb", type=str, required=True)

    args = parser.parse_args()

    score_sequences(args.fasta_file, args.pdb_folder, args.target_pdb)
    
