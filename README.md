
# BindCraft Scoring

This script evaluates the parameters used for loss and filtering in the [BindCraft paper](https://github.com/martinpacesa/BindCraft). 

## Usage

1. Clone **ProteinMPNN** inside the `functions` folder:
   ```bash
   git clone https://github.com/dauparas/ProteinMPNN functions/ProteinMPNN
   ```
2. Set up the **BindCraft** Conda environment as described in the `bindcraft` folder.
3. Provide the following inputs:\
    a. FASTA file\
    b. AlphaFold3 (AF3) results in a directory (see example inputs). ⚠️ The binder chain_ID should be "B" ⚠️\
    c. PDB file of the target without any bind\
See the ´input_example´ to see what are the inputs and the outputs :) 
### Output:
The script generates a **CSV file** containing all scoring parameters.

🚧 *Work in progress!* 🚧
