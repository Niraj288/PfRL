import os
import numpy as np


class Work:
	def __init__(self, seq, grid, name):
		self.seq = seq
		self.grid = grid
		self.name = name

	def gen_CA_pdb(self):
		st = ''
		
		for i in range (len(self.seq)):
			l = ['ATOM', str(i+1), 'CA', self.seq[i], 'A', str(i+1), self.grid[i][0], self.grid[i][1], self.grid[i][2], '1.00', '0.00', 'C']
			print ("{:>4}{:>7}{:>5}{:>4}{:>2}{:>4}{:>12}{:>8}{:>8}{:>6}{:>6}{:>12}".format(*l))
			st += "{:>4}{:>7}{:>5}{:>4}{:>2}{:>4}{:>12}{:>8}{:>8}{:>6}{:>6}{:>12}".format(*l)+'\n'

		st += 'TER\nEND'

		g = open(self.name+'_CA.pdb','w')
		g.write(st)
		g.close()



if __name__ == '__main__':
	name = '1k43'
	seq = ['ARG', 'GLY', 'LYS', 'TRP', 'THR', 'TYR', 'ASN', 'GLY', 'ILE', 'THR', 'TYR', 'GLU', 'GLY', 'ARG']
	grid = np.array([[0.0, 0.0, 0.0],
		[3.78, 0.0, 0.0],
		[7.56, 0.0, 0.0],
		[11.35, 0.0, 0.0],
		[15.14, 0.0, 0.0],
		[18.92, 0.0, 0.0],
		[22.71, 0.0, 0.0],
		[26.49, 0.0, 0.0],
		[30.28, 0.0, 0.0],
		[34.06, 0.0, 0.0],
		[37.84, 0.0, 0.0],
		[41.63, 0.0, 0.0],
		[45.42, 0.0, 0.0],
		[49.20, 0.0, 0.0]])

	w = Work(seq, grid, name)

	w.gen_CA_pdb()

# for making pdb from CA
# java apps.BBQ -bbq.fix_r12=T -ip=1k43_CA.pdb
# add side chains via SCWRL