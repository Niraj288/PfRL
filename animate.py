import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class render:
	def __init__(self, rang):
		plt.ion()
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		ax.set_zlim3d(-rang, rang)                    
		ax.set_ylim3d(-rang, rang)                    
		ax.set_xlim3d(-rang, rang) 

		self.lines1 = None
		self.ax = ax

	def update(self, lis, ini = 0):
		l = np.array(lis)
		x, y, z = l[:,0],l[:,1],l[:,2] 

		if self.lines1:
			self.lines1.remove()
			l = self.lines2.pop(0)
			l.remove()
			del l#self.lines2
			#self.lines2.remove()
		self.lines1 = self.ax.scatter(x, y, z, c = 'r', s = 100)
		self.lines2 = self.ax.plot(x, y, z, c = 'r')
		plt.draw()
		if ini:
			plt.pause(5)
		else:
			plt.pause(0.001)