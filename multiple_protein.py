import sys
import os
import numpy as np
from scipy.spatial import distance_matrix
import scipy.spatial as spatial
import atom_data as ad
import math 
from math import log10, floor
import animate


class environ_grid:
        def __init__(self, pdb, name, RENDER = 0):
                self.name = name
                self.RENDER = RENDER
                self.SYNC_TARGET_FRAMES = 100
                if 'proteins' not in os.listdir('.'):
                    raise Exception('No folder named proteins found !')
                self.pdb_files = [fi for fi in os.listdir('proteins/')]
                #protein.__init__(self, pdb)
                self.current_index = 0
                self.initialize()
                if self.RENDER:
                        self.anim = animate.render(max(self.nres)+2)
                self.init_args()

        def initialize(self):
                self.names = ['.'.join(pdb.split('/')[-1].split('.')[:-1]) for pdb in self.pdb_files]
                #print ('Making pdb file with the first frame (_f1.pdb) ...')
                #self.truncate_pdb()

                self.direcn = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0],
                                 [0,0,1], [0,0,-1], [1,1,1], [1,1,-1],
                                 [1,-1,1], [-1,1,1], [-1,-1,-1], [-1,-1,1],
                                 [-1,1,-1], [1,-1,-1]])

                self.res_d = {}

                self.res_arrs = [self.make_xleap_input_sequence(self.pdb_files[i], self.names[i]) for i in range (len(self.pdb_files))]

                #print (self.res_arr)

                self.nres = [len(i) for i in self.res_arrs]

                self.igrid = self.make_and_assign_3Dgrid()
                #print ('i', self.igrid)
                # initial grid

        def make_xleap_input_sequence(self, f, name):

                def get_sequence(lines):
                        d,rid={},1
                        for line in lines:
                                if "TER" in line.split()[0]:
                                        break
                                if line.split()[0] in ['ATOM','HETATM']:
                                        #print line
                                        id,at,rt,_,_0,x,y,z=line.strip().split()[1:9]
                                        s=line.strip().split()[-1]
                                        d[int(_0)]=rt
                                        rid+=1
                        print (name, d)
                        arr = [d[i] for i in range (1,len(d)+1)]
                        return arr

                file = open('proteins/'+f,'r')
                lines= file.readlines()
                file.close()

                seq = get_sequence(lines)
                
                for i in range (len(seq)):
                        if seq[i] in self.res_d:
                                seq[i] = self.res_d[seq[i]]
                        else:
                                self.res_d[seq[i]] = len(self.res_d)+1
                                seq[i] = self.res_d[seq[i]]

                return seq

        def make_and_assign_3Dgrid(self):
                l = max(self.nres)+2
                grid = np.zeros((l,l,l))
                res_grid_pos = {}
                for i in range (self.nres[self.current_index]):
                        grid[i+1, i+1, i+1] = self.res_arrs[self.current_index][i]
                        res_grid_pos[i] = [i+1, i+1, i+1]

                self.res_grid_pos = res_grid_pos
                return grid

        def init_args(self):
                self.dgrid = np.copy(self.igrid)
                state = self.reset()
                l = state.shape[0]
                self.obs_size = l

                self.n_actions = max(self.nres)*14

                # Make final conformation grids
                self.fgrids = self.make_fgrid()

                if self.RENDER:
                		lis = [self.trace_r[self.current_index][t] for t in self.trace_r[self.current_index]]
                		self.save_xyz(0.0, lis)
                		self.anim.update(lis)

                #print (self.fgrid)
                #np.save('grid.npy', {'grid':self.fgrid})

                # compute pairwise distances of final proteins
                coords = [[self.trace_r[ind][i] for i in self.trace_r[ind]] for ind in range (len(self.pdb_files))]
                #print (coord)
                self.M_fgrids = [distance_matrix(coord, coord) for coord in coords]

        def make_fgrid(self):

                def is_backbone(r,st):
                            l=len(r)
                            l_=st[l:]
                            if len(l_)==0 or l_.isdigit() or l_ == 'A':
                                return True
                            return False

                def data_extraction(lines):
                            
                            d={}
                            l = {}
                            for line in lines:
                                        if "ENDMDL" in line.split()[0]:
                                            break
                                        if line.split()[0] in ['ATOM']:
                                            #print line
                                            id,at,rt,_,_0,x,y,z=line.strip().split()[1:9]
                                            s=line.strip().split()[-1]
                                            x, y, z = list(map(float, [x, y, z]))
                                            if is_backbone(s, at):
                                                            if rt+_0 not in d:
                                                                        l[len(d)] = rt+_0
                                                                        d[rt+_0]=[[x, y, z]]
                                                            else:
                                                                        d[rt+_0].append([x, y, z])
                            return d, l

                def get_quad(vec, cur):


                        temp = self.direcn[np.argmin(np.sum(np.abs(np.array(vec)-self.direcn), axis=1))]
                        return list(cur+temp)

						


                def get_grid(pdb):
                        f = open('proteins/'+pdb, 'r')
                        lines = f.readlines()
                        f.close()

                        d, l = data_extraction(lines)
                        lis = []

                        #print (d, l)

                        r_coord = []
                        for i in range (len(l)):
                                # centre point of residue
                                tarr = np.array(d[l[i]])
                                #print tarr
                                cp = [tarr[:,0].mean(), tarr[:,1].mean(), tarr[:,2].mean()]
                                #print (cp)
                                r_coord.append(np.array(cp))

                        lt = max(self.nres)+2
                        grid = np.zeros((lt, lt, lt))

                        # first residue at 0,0,0
                        cur = [0,0,0]

                        trace_r = {1:cur} # track where residues are placed

                        for j in range (1, len(r_coord)):
                                vec = r_coord[j] - r_coord[j-1]
                                #print (vec)
                                cur = get_quad(vec, cur) # get current grid place from relative vector
                                #print(cur)
                                trace_r[j+1] = cur 

                        pos_arr = np.array([trace_r[j] for j in range (1, len(trace_r)+1)])
                        min_x = np.min(pos_arr[:,0])-1
                        min_y = np.min(pos_arr[:,1])-1
                        min_z = np.min(pos_arr[:,2])-1

                        # transform the avg pos to nres/2
                        cen = np.array([min_x, min_y, min_z])
                        for i in range (len(trace_r)):
                                trace_r[i+1] = np.array(trace_r[i+1]) - cen 

                                # move all to grid centre lt/2, lt/2, lt/2
                                #trace_r[i+1] = trace_r[i+1] + list(map(int,[lt/2, lt/2, lt/2]))

                        #print (trace_r)


                        for t in trace_r:
                                grid[tuple(map(int,trace_r[t]))] = t  

                        
                        self.trace_r.append(trace_r)

                        return grid 

                self.trace_r = []
                lis = [get_grid(pdb) for pdb in self.pdb_files]

                return lis

        def reset(self):
		        #print('reset called')
		        # set dynamic coordinate to initial coordinate
		        self.current_index = np.random.choice(range(len(self.pdb_files)))

		        self.dgrid = self.make_and_assign_3Dgrid()

		                #print('actual reset')
		        self.nframes = 1

		        state = self.state()

		        return state

        def state(self):

        		return self.dgrid.flatten()

        def get_reward(self):

        	# right now based on the pairwise distance
        	coord = [self.res_grid_pos[i] for i in self.res_grid_pos]
        	M_dgrid = distance_matrix(coord, coord)

        	return -1*np.sum((self.M_fgrids[self.current_index] - M_dgrid)**2)

        def distance(self,a,b):
        		a = list(map(float,a))
        		b = list(map(float,b))
        		return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)



        def step(self, action):
                ac = np.argmax(action)
                #print (self.nframes,'frame')
                # action space is N*6
                res_index, dirn = divmod(ac,14)

                # current place of residue in grid
                cu_r = self.res_grid_pos[res_index]

                # make that place zero and assign a new value
                self.dgrid[tuple(cu_r)] = 0.0

                new_place = cu_r+self.direcn[dirn]

                penalty = 0.0

                def check_chain(ind):
                		lim = 3.0
                		if ind > 1 and ind < max(self.nres) - 1:
                				r1 = self.distance(new_place, self.res_grid_pos[ind - 1])
                				r2 = self.distance(new_place, self.res_grid_pos[ind + 1])
                				if r1 < lim and r2 < lim:
                						return True 
                				return False
                		elif ind == 0:
                				r2 = self.distance(new_place, self.res_grid_pos[ind + 1])
                				if r2 < lim:
                						return True 
                		elif ind == max(self.nres) - 1:
                				r1 = self.distance(new_place, self.res_grid_pos[ind - 1])
                				if r1 < lim:
                						return True 
                		return False

                # No overlap between residues
                if list(new_place) in [list(self.res_grid_pos[pos]) for pos in self.res_grid_pos]:
                        new_place = cu_r

                        penalty = -100

                # dont move if at grid corner
                elif max(new_place) >= max(self.nres)+2 or min(new_place) < 0:
                        new_place = cu_r

                        penalty = -100

                elif not check_chain(res_index):
                		new_place = cu_r

                		penalty = -100

                self.dgrid[tuple(new_place)] = self.res_arrs[self.current_index][res_index]

                self.res_grid_pos[res_index] = new_place

                if self.RENDER:
                		lis = [self.res_grid_pos[t] for t in self.res_grid_pos]
                		self.anim.update(lis)

                new_state = self.state()

                reward = self.get_reward()+penalty

                if penalty != 0.0:
                		reward = None

                is_done = False

                if self.nframes >= self.SYNC_TARGET_FRAMES:
                        #print ('done')
                        is_done = True
                self.nframes += 1

                return new_state, reward, is_done


        def sample_action_space(self, index = None):
                s = np.zeros(max(self.nres)*14) # N residues * 14 direcn
                if index:
                        s[index] = 1.0
                        return s
                i = np.random.randint(max(self.nres)*14)
                s[i] = 1.0
                return s

        def save_xyz(self, reward = 0, lis = None):

                if 'temp_grid.npy' not in os.listdir('.'):
                        d = {}
                else:
                        d = np.load('temp_grid.npy').item()

                c = self.dgrid
                print ('Reward:',reward)
                #print (lis)

                if lis is None:
                		lis = [self.res_grid_pos[t] for t in self.res_grid_pos]

                d[len(d)] = lis#np.copy(c)

                np.save('temp_grid.npy', d)

        def __str__(self):
                return self.name




if __name__ == '__main__':

	env = environ_grid_multiple_pdb(sys.argv[1], 'test')
