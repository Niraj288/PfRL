import sys
import os
import numpy as np
from scipy.spatial import distance_matrix

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

	# get potential energy
	def getPE(self, coord):
		a,b = coord.shape
		coord = coord.reshape((-1))

		st_coord = 'Temporary file\n'+str(a)+'\n'
		for i in range (0, len(coord), 6):

			st_coord += "{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}".format(*coord[i:i+6])+'\n'

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

		for line in lines:
			if 'Etot   = ' in line:
				return float(line.strip().split()[-1])

		raise Exception('Something wrong while evaluating PE output')

	def __str__(self):
		return self.name 

class environ(protein):
	def __init__(self, pdb, name):
		self.name = name
		protein.__init__(self, pdb)
		#super(environ, self, pdb).__init__()
		self.reset()

	def reset(self):
		# set dynamic coordinate to initial coordinate
		self.dcoord = self.icoord
		self.max_energy = 150
		self.natoms = len(self.atoms)
		self.directions = np.array([[1,0,0],
							[-1,0,0],
							[0,1,0],
							[0,-1,0],
							[0,0,1],
							[0,0,-1]])

		state = self.state()

		# specific to pairwise state
		l = state.shape[0]
		self.obs_size = l*l

		self.n_actions = 3*self.natoms*6
		return state

	def state(self):
		#print (self.dcoord.shape, 'dcoord')
		M = distance_matrix(self.dcoord, self.dcoord)

		return M

		# Make M upper triangle pairwise distances

	def step(self, action):
		ac = np.argmax(action)
		#print (ac)
		# action space is 3N*6
		atom_index, direcn = divmod(ac,6)
		atom_index = atom_index/3
		print (atom_index, direcn)
		current_coord = self.dcoord[int(atom_index)]
		self.dcoord[int(atom_index)] = current_coord+0.5*self.directions[direcn] # move 0.5 Angstron


		new_state = self.state()

		reward = -self.getPE(self.dcoord) # -ve is to maximize energy

		is_done = False

		if reward > self.max_energy:
			is_done = True

		return new_state, reward, is_done



	def sample_action_space(self):
		s = np.zeros(self.natoms*3*6) # 3N coordinates * 6 direcn
		i = np.random.randint(self.natoms*3*6)
		s[i] = 1.0
		return s



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



















