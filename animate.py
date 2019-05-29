import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import align3D as align

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
                        pass
                        #plt.pause(5)
                else:
                        plt.pause(0.05)

        def plot_final(self, lis1, lis2): # lis1 is test, lis2 is actual protein
                if self.lines1:
                        self.lines1.remove()
                        l = self.lines2.pop(0)
                        l.remove()
                        del l#self.lines2
                        #self.lines2.remove()

                l1 = np.array(lis1)
                l2 = np.array(lis2)

                l1, l2, rmsd = align.alin(l1, l2)

                print ('RMSD :', rmsd)

                x1, y1, z1 = l1[:,0],l1[:,1],l1[:,2]
                x2, y2, z2 = l2[:,0],l2[:,1],l2[:,2]

                self.lines1 = self.ax.scatter(x1, y1, z1, c = 'r', s = 200)
                self.lines2 = self.ax.plot(x1, y1, z1, c = 'r')

                self.lines3 = self.ax.scatter(x2, y2, z2, c = 'g', s = 200)
                self.lines4 = self.ax.plot(x2, y2, z2, c = 'g')

                plt.draw()
                plt.show(block = True)

if __name__ == '__main__':
        b = np.array([[0,0,0], [1,1,1], [2,2,2]], dtype = 'float') # original molecule
        a = np.array([[0,0,0], [-1.5,-1,-1], [-2,-2,-2]], dtype = 'float') # fake molecule	
        r = render(5)
        r.plot_final(a,b)










