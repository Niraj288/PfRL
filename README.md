# PfRL

Protein folding via Reinforcement Learning (PfRL)

## Requirements

1. Python 3
2. AmberTools 16

### Installation of AmberTools

conda install -c ambermd ambertools
source ~/amber/bin/amber.sh


## Protein class description

When called, the class performs some priliminary functions to get started from pdb. The coordinates from pdb is takes out and a straight chain for the protein is made. The straight chain is minimized for VanDerWallas interactions etc. The force field ff14fb is applied for making topology files and energy calculations.

### coordi

gives the initial coordinates of the straight chain of protein atoms
shape = (-1,3)

### atoms

gives the atomic number of each of the atoms
shape = (-1)

### API : getPE

args = coordinates of shape (-1,3)
returns Potential energy of the coordinates provided
