#!/bin/python3

import math
import os
import random
import re
import sys
import numpy as np

FCOUNTS = [10,20]
TRACK = [-1,5]
BCOUNT = [-1,5]

def make_inp(st1,st2,i,j,k):
	name = 'F'+str(i)+'B'+str(j)+'T'+str(k)
	os.system('mkdir '+name)
	#os.system('chdir ' + name)
	f = open('inp','w')
	f.write(st1)
	f.close()
	os.system('mv inp '+ name)
	f2 = open('run.sbatch','w')
	f2.write(st2)
	f2.close()
	os.system('mv run.sbatch ' +name)
	os.system('cd ' + name)
	#os.system('chdir ' + name)
	os.system('sbatch run.sbatch')
	os.system('cd ../')

for i in FCOUNTS:
	for j in TRACK:
		for k in BCOUNT:
			st1 = """MEAN_REWARD_BOUND -3.0 \n\nGAMMA  0.99\n\nBATCH_SIZE  32\n\nREPLAY_SIZE  10000\n\nLEARNING_RATE  1e-4\n\nSYNC_TARGET_FRAMES  1000\n\nREPLAY_START_SIZE  10000\n\nEPSILON_DECAY_LAST_FRAME  10**7\n\nMAX_ITER 10**9\n\nEPSILON_START  1.0\n\nEPSILON_FINAL  0.05\n\ndevice gpu\n\nHIDDEN_SIZE 256\n\nFCOUNTS """ + str(i) + '\n\n' + 'BCOUNT ' + str(j) + '\n\n' + 'TRACK ' + str(k) + '\n'
			st2 ="""#!/bin/bash\n#SBATCH -J inp\n#SBATCH -o log_%j.out
#SBATCH -p gpgpu-1,v100x8
#SBATCH -c 1\n#SBATCH --mem=124G\n\nmodule load python/3\n\npython -u ../../idqn.py inp > log.txt\n\n"""
			
			make_inp(st1,st2,i,j,k)
 


