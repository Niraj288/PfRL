#!/bin/python3

import math
import os
import random
import re
import sys
import numpy as np

FCOUNTS = [10,20,50,100]
TRACK = [-1,20,50]
BCOUNT = [-1,20,50]

#FCOUNTS = [10,20,30,40,50,80,100,150]
#TRACK = [-1,10,20,30,50,80,100]
#BCOUNT = [-1,10,20,30,50,80,100]

def make_inp(st1,st2,i,j,k):
	name = 'F'+str(i)+'B'+str(j)+'T'+str(k)+'sta'
	os.system('mkdir '+name)
	os.system('cp -r proteins ' + name)
	os.system('cp -r errors ' + name)
	os.system('cp -r models ' + name)
	f = open('inp','w')
	f.write(st1)
	f.close()
	os.system('mv inp '+ name)
	f2 = open('run.sbatch','w')
	f2.write(st2)
	f2.close()
	os.system('mv run.sbatch ' +name)
	os.chdir(name)
	#os.system('chdir ' + name)
	print (os.getcwd())
	os.system('sbatch run.sbatch')
	os.chdir('../')

for i in FCOUNTS:
	for j in TRACK:
		for k in BCOUNT:
			st1 = """DEFAULT_ENV_NAME PfRL\n\ndoSimulation 0\n\nRENDER 0\n\nMEAN_REWARD_BOUND -3.0 \n\nGAMMA  0.99\n\nBATCH_SIZE  32\n\nREPLAY_SIZE  10000\n\nLEARNING_RATE  1e-4\n\nSYNC_TARGET_FRAMES  1000\n\nREPLAY_START_SIZE  10000\n\nEPSILON_DECAY_LAST_FRAME  10**7\n\nMAX_ITER 10**9\n\nEPSILON_START  1.0\n\nEPSILON_FINAL  0.05\n\ndevice cpu\n\nHIDDEN_SIZE 256\n\nFCOUNTS """ + str(i) + '\n\n' + 'BCOUNT ' + str(j) + '\n\n' + 'TRACK ' + str(k) + '\n'
			st2 ="""#!/bin/bash\n#SBATCH -J inp\n#SBATCH -o log_%j.out
#SBATCH -p standard-mem-m,standard-mem-l,high-mem-1,high-mem-2 
#SBATCH -c 1\n#SBATCH --mem=124G\n\nmodule load python/3\n\npython -u /users/mmakos/scratch/Protein_folding/PfRL/idqn.py inp > log.txt\n\n"""
			
			make_inp(st1,st2,i,j,k)
 


