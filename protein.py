from sklearn.preprocessing import MinMaxScaler
import sys
import os
import numpy as np
from scipy.spatial import distance_matrix
import scipy.spatial as spatial
import atom_data as ad
import math 
from math import log10, floor

class protein:
	def __init__(self, pdb_file, name = 'Environment not set'):
		self.name = name 
		self.pdb_file = pdb_file
		self.initialize()

	def initialize(self):
		self.name = '.'.join(self.pdb_file.split('/')[-1].split('.')[:-1])
		print ('Making pdb file with the first frame (_f1.pdb) ...')
		self.truncate_pdb()

		print ('Generating pdb file with hydrogens in it (_r.pdb) ...')
		st = 'reduce name_f1.pdb > name_r.pdb'
		st = st.replace('name',self.name)
		os.system(st)

		print ('Making topology files for amber with straight chain sequence ...')
		self.make_xleap_input_sequence(self.name+'_r.pdb', self.name)
		os.system('tleap -f xleap_input > xleap.out')

		f = open('xleap.out', 'r')
		lines = f.readlines()
		f.close()

		if 'Failed' in ''.join(lines):
			raise Exception("Failed to perform tleap on "+self.name)

		print ('Initial Minimization ...')
		self.minimization()
		st = 'sander -O -i min1.in -o min1.out -p name.prmtop -c name.xyz -r min1.rst'
		st = st.replace('name',self.name+'_r')
		os.system(st)

		# initial coordinates
		self.icoord = self.get_coord_xyz(self.name+'_r.xyz')

		print ('Saving final pdb (_r_opt.pdb) ...')
		st = 'ambpdb -p name.prmtop -c min1.rst > name_opt.pdb'
		st = st.replace('name',self.name+'_r')
		os.system(st)

		# topology file
		self.namet = self.name+'_r.prmtop'

		# xyz file
		self.namexyz = self.name+'_r.xyz'

		# reduced file
		self.namer = self.name+'_r'

		self.atoms()

		print ('Making single point energy input ...')
		self.sp_input()

		self.getConn()

	def make_xleap_input(self, name):

		st=''
		st+='source oldff/leaprc.ff99SB\n'

		st+=name +' = loadpdb '+name+'.pdb\n'

		st+='saveamberparm '+name+' '+' '+name+'.prmtop'+' '+name+'.xyz\n'
		st+='quit\n'
		g = open('xleap_input','w')
		g.write(st)
		g.close()

		return

	def make_xleap_input_sequence(self, f, name):

		def get_sequence(lines):
			d={}
			for line in lines:
				if "TER" in line.split()[0]:
					break
				if line.split()[0] in ['ATOM','HETATM']:
					#print line
					id,at,rt,_,_0,x,y,z=line.strip().split()[1:9]
					s=line.strip().split()[-1]
					d[int(_0)]=rt
			print (d)
			d[1] = 'N'+d[1]
			d[len(d)] = 'C'+d[len(d)]
			arr = [d[i] for i in range (1,len(d)+1)]
			return ' '.join(arr)

		file = open(f,'r')
		lines= file.readlines()
		file.close()

		#name = '.'.join(f.split('/')[-1].split('.')[:-1])

		st=''
		st+='source oldff/leaprc.ff14SB\n'
		seq = get_sequence(lines)
		st+=name +' = sequence { '+seq+' }\n'

		st+='saveoff '+name+' '+name+'_linear.lib\n'
		st+='saveoff '+name+' '+name+'_linear.pdb\n'

		st+='saveamberparm '+name+' '+' '+name+'_r.prmtop'+' '+name+'_r.xyz\n'
		st+='quit\n'

		g = open('xleap_input','w')
		g.write(st)
		g.close()

		return

	def get_coord_xyz(self, xyz_file):
		f = open(xyz_file, 'r')
		lines = f.readlines()
		f.close()

		k = []
		for line in lines[2:]:
			if len(line.strip().strip()) == 0:
				break

			k += list(map(float,line.strip().split()))

		k = np.array(k).reshape((-1,3))

		return k

	def truncate_pdb(self):
		f = open(self.pdb_file,'r')
		lines = f.readlines()
		f.close()

		new_lines = []
		for line in lines:
			new_lines.append(line)
			if 'ENDMDL' in line:
				break
		g = open(self.name+'_f1.pdb', 'w')
		g.write(''.join(new_lines))
		g.close()
		return 

	def minimization(self):
		text = """Stage 1 - minimisation
&cntrl
 imin=1, maxcyc=100, ncyc=500,
 cut=999., rgbmax=999.,igb=1, ntb=0,
 ntpr=100
/
"""
		g = open('min1.in','w')
		g.write(text)
		g.close()

	def sp_input(self):
		f = open('sp.in','w')
		f.write("""single point at 0 K
&cntrl
 imin=0, irest=0, ntx=1,
 nstlim=1, dt=0.0005,
 ntc=2, ntf=2,
 ntt=1, tautp=1.0,
 tempi=0.0, temp0=0.0,
 ntpr=50, ntwx=50,
 ntb=0, igb=1,
 cut=999.,rgbmax=999.
/
""")
		f.close()

	def atoms(self):
		f = open(self.namet, 'r')
		lines = f.readlines()
		f.close()
		ref = 0
		li = []
		for line in lines:
			if ref == 2 and '%FLAG' in line:
				break
			if ref == 2:
				li += list(map(int, line.strip().split()))
			if ref == 1:
				ref = 2
			if '%FLAG ATOMIC_NUMBER' in line:
				ref = 1
		self.atoms = li 

	def round_sig(self, x, sig=2):
		return round(x, sig-int(floor(log10(abs(x))))-1)

	# get potential energy
	def getPE(self, coord):
		a,b = coord.shape
		coord = coord.reshape((-1))

		st_coord = 'Temporary file\n'+str(a)+'\n'
		for i in range (0, a*b, 6):
			li = []
			for j in range (i,min(i+6, a*b)):
				if len(str(coord[j])) > 10:
					coord[j] = self.round_sig(coord[j],9)
			if min(i+6, a*b) == a*b:#coord[j] = float(str(coord[j])[:12])
				st_coord += "{:>12}{:>12}{:>12}".format(*coord[i:a*b])+'\n'
			else:
				st_coord += "{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}".format(*coord[i:min(i+6, a*b)])+'\n'

		f = open(self.namer+'_temp.xyz', 'w')
		f.write(st_coord)
		f.close()

		st = 'sander -O -i sp.in -o sp.out -p namet -c name_temp.xyz'
		st = st.replace('namet',self.namet)
		st = st.replace('name',self.namer)
		os.system(st)

		g = open('sp.out', 'r')
		lines = g.readlines()
		g.close()

		PE = 0.0

		for line in lines:
			if 'Etot   = ' in line:
				if '*' in line:
					PE = 9999999999
				else:
					PE = float(line.strip().split()[-1])
				#print (PE)
				break
		# Harmonic potential
		coord = coord.reshape((-1,3))
		HE = 0.0
		for t in self.conn:
			a,b = t
			dis = self.distance(coord[a], coord[b])
			diff = dis - self.conn[t]
			if diff > 1.0:
				HE += 10000*diff**2
		if PE == 0.0:
			raise Exception('Something wrong while evaluating PE output')
		#print (HE)
		return PE+HE

	def distance(self,a,b):
	    a = list(map(float,a))
	    b = list(map(float,b))
	    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

	def getConn(self):
		a = self.atoms
		d = {}
		point_tree = spatial.cKDTree(self.icoord)
		for i in range (self.icoord.shape[0]):
			li=(point_tree.query_ball_point(self.icoord[i], 4.0))
			for j in li:
				if i == j:
					continue
				at = [self.atoms[i], self.atoms[j]]
				at.sort()
				dis = self.distance(self.icoord[j], self.icoord[i])
				if at[-1] > 10:
					if dis > 3.5:
						continue
				elif dis > 1.5:
					continue
				l2 = [i,j]
				l2.sort()		
				d[tuple(l2)] = self.distance(self.icoord[j], self.icoord[i])
		self.conn = d
		return 

	def __str__(self):
		return self.name 

class environ(protein):
	def __init__(self, pdb, name):
		self.name = name
		self.SYNC_TARGET_FRAMES = 100
		protein.__init__(self, pdb)

		self.natoms = len(self.atoms)

		# add random noise to  initial coordinates
		noise = np.random.normal(0,0.5,self.natoms*3).reshape((self.natoms, 3))
		self.dcoord = np.copy(self.icoord + noise)
		
		#self.dcoord = np.copy(self.icoord)
		
		# indexes for upper triangle
		self.iu = np.triu_indices(self.natoms)
		
		self.directions = np.array([[1,0,0],
                                            [-1,0,0],
                                            [0,1,0],
                                            [0,-1,0],
                                            [0,0,1],
                                            [0,0,-1]])
		self.reset()

	def reset(self):
	        #print('reset called')
	        # set dynamic coordinate to initial coordinate
	        ind = 1#np.random.choice([1,0])
	        if ind:
	                self.dcoord = np.copy(self.icoord)
	                #print('actual reset')
	        self.nframes = 1

	        state = self.state()

	        # specific to pairwise state
	        l = state.shape[0]
	        self.obs_size = l

	        self.n_actions = self.natoms*6
	        return state

	def state(self):
		#print (self.dcoord.shape, 'dcoord')
		M = distance_matrix(self.dcoord, self.dcoord)

		# take upper triangle
		M = M[self.iu]
		
		M = M.flatten()
		
		'''
		# for adding previous 5 frames
		if M.shape[0] < 5*self.iu.shape[0]:
			for j in range (
		'''
		return M

		# Made M upper triangle pairwise distances

	def step(self, action):
		ac = np.argmax(action)
		#print (self.nframes,'frame')
		# action space is N*6
		atom_index, direcn = divmod(ac,6)
		atom_index = atom_index
		#print (int(atom_index), direcn)
		new_coord = self.dcoord[int(atom_index)]+0.05*self.directions[direcn] # move 0.05 Angstron
		#print (self.dcoord[int(atom_index)])
		self.dcoord[int(atom_index)] = new_coord 
		#print (self.dcoord[int(atom_index)])

		new_state = self.state()

		reward = -self.getPE(self.dcoord) # -ve is to maximize energy
		#print ('Reward:',reward)
		is_done = False

		if self.nframes >= self.SYNC_TARGET_FRAMES:
			#print ('done')
			is_done = True
		self.nframes += 1

		return new_state, reward, is_done



	def sample_action_space(self, index = None):
		s = np.zeros(self.natoms*6) # N coordinates * 6 direcn
		if index:
			s[index] = 1.0
			return s
		i = np.random.randint(self.natoms*6)
		s[i] = 1.0
		return s

	def save_xyz(self, reward = 0):
		d = ad.data(sys.argv[0])
		a = self.atoms
		sd = [d[i]['symbol'] for i in a]

		c = self.dcoord
		print ('Reward:',reward)
		f = open('render.xyz','a')
		f.write(str(len(a))+'\nReward : '+str(reward)+'\n')
		for j in range (len(a)):
			st = sd[j]+' '+' '.join(list(map(str,c[j])))+'\n'
			f.write(st)
		f.close()


class environ_coord(environ):
        def state(self):
                a = np.array(self.atoms).reshape((-1,1))
                c = np.copy(self.dcoord)
                min_max_scaler = MinMaxScaler()
                c = min_max_scaler.fit_transform(c)
                M1 = np.concatenate((a,c), axis = 1).flatten()
                M2 = distance_matrix(self.dcoord, self.dcoord)

                # take upper triangle
                M2 = M2[self.iu]

                M2 = M2.flatten()
                M = np.concatenate((M1, M2), axis = None)
                return M.flatten()

class environ_grid(environ):
        def initialize(self):
                self.name = '.'.join(self.pdb_file.split('/')[-1].split('.')[:-1])
                print ('Making pdb file with the first frame (_f1.pdb) ...')
                self.truncate_pdb()

                self.res_arr = self.make_xleap_input_sequence(self.name+'_r.pdb', self.name)

                self.icoord = self.make_and_assign_3Dgrid()
                # initial grid

        def make_and_assign_3Dgrid(self):
                l = arr.shape[0]+2
                grid = np.zeros((l,l,l))
                for i in range (len(self.res_arr)):
                        grid[i+1, i+1, i+1] = self.res_arr[i]
                return grid


        def make_xleap_input_sequence(self, f, name):

                def get_sequence(lines):
                        d={}
                        for line in lines:
                                if "TER" in line.split()[0]:
                                        break
                                if line.split()[0] in ['ATOM','HETATM']:
                                        #print line
                                        id,at,rt,_,_0,x,y,z=line.strip().split()[1:9]
                                        s=line.strip().split()[-1]
                                        d[int(_0)]=rt
                        print (d)
                        arr = [d[i] for i in range (1,len(d)+1)]
                        return arr

                file = open(f,'r')
                lines= file.readlines()
                file.close()

                seq = get_sequence(lines)
                d = {}
                for i in range (len(seq)):
                        if seq[i] in d:
                                seq[i] = d[seq[i]]
                        else:
                                d[seq[i]] = len(d)
                                seq[i] = d[seq[i]]

                return seq


if __name__ == '__main__':

	# pdb file
	pfile = sys.argv[1]

	p = protein(pfile)

	# initial coord
	ic = p.icoord

	# calculate this coordinates energy
	e = p.getPE(ic)
	print (e)

	print (p.atoms)

	#print (p.getConn())





















