# for maya

import maya.cmds as cmds
import numpy as np


def make_chain(icoord, gen = 0, key = 0, t = 0):
	cmds.select(all = True)
	if key:
		cmds.setKeyFrame(t = t)
	for i in range (len(icoord)):
		if gen:
			cmds.sphere(n = 'chain'+str(i), r = 0.8)
		cmds.move('chain'+str(i), icoord[i][0], icoord[i][1], icoord[i][2])


def job():
	d = np.load('temp_grid.npy').item()

	icoord = d[0]

	make_chain(icoord, 1)

	for i in range (1, len(d)):
		make_chain(d[i], 0, 1, i)

if __name__ == '__main__':
	job()





