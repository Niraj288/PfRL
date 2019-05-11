import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class render:
	def __init__(self, rang):
		plt.ion()
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		ax.set_zlim3d(0, rang)                    
		ax.set_ylim3d(0, rang)                    
		ax.set_xlim3d(0, rang) 

		self.lines = None
		self.ax = ax

	def update(self, lis):
		l = np.array(lis)
		x, y, z = l[:,0],l[:,1],l[:,2] 

		if self.lines:
			self.lines.remove()
		self.lines = self.ax.scatter(x, y, z, c = 'r')
		plt.draw()
		plt.pause(0.01)