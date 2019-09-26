# PfRL

Protein folding via Reinforcement Learning (PfRL) This code is a naive implementation of PfRL. The actual code is private.

# PfRL with Improved dqn

Example to train:

Download protein databse from 'https://smu.box.com/s/0395yjihdwm1p0he09cqm5mahg8z9ah9'

Extract data and put the proteins folder in your working directory

Make directory models, errors

make input file (An example is given as inp)

Train : python path_to_PfRL/idqn.py path_to_inp

Test : python path_to_PfRL/test_model.py path_to_inp path_to_pdb_file

## Requirements

1. Python 3
2. AmberTools 16

### Installation of AmberTools

conda install -c ambermd ambertools

source ~/amber/bin/amber.sh

## Just run for test (steps)

Train the model for protein 1k43
1. copy 1k43.pdb in some folder
2. From that folder run
3. 'python dqn.py'


## Protein class description

When called, the class performs some priliminary functions to get started from pdb. The sequence from pdb file is taken out and a straight chain for the protein is made. The straight chain is minimized for VanDerWallas interactions etc. The force field ff14fb is applied for making topology files and energy calculations.

### Initiate class by the name of protein file

E.g. p = protein('1mzi.pdb')

### icoord

gives the initial coordinates of the straight chain of protein atoms,
shape = (-1,3)

### atoms

gives the atomic number of each of the atoms,
shape = (-1)

### API : getPE

args = coordinates of shape (-1,3)],
returns Potential energy of the coordinates provided

## Environment

The class environ inherits from protein class

### reset

Resets the coordinates to initial coordinates

### state

Give the information about the environment. Currently return pairwise distance for each atom in the protein

### step

performs the action and return new_state, reward, is_done

### smaple_action_space

Sample a random action

### save_xyz

Save xyz information for render purpose
