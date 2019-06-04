import sys
import os
import numpy as np
#from scipy.spatial import distance_matrix
#import scipy.spatial as spatial
#import atom_data as ad
import math 
from math import log10, floor
import animate


class environ_grid:
        def __init__(self, pdb, 
                           name, 
                           RENDER = 0, 
                           test = 0, 
                           track = 5,
                           fcounts = 6,
                           bcount = -1):
                self.name = name
                self.test = test
                self.RENDER = RENDER
                self.SYNC_TARGET_FRAMES = 100

                # how much to look in future
                self.fcounts = fcounts

                # bcount is how much to go backward for reward
                self.bcount = bcount #5

                self.res_track = track # how much residue coordinates be included from generated sequence in the state
                if 'proteins' not in os.listdir('.'):
                    raise Exception('No folder named proteins found !')
                
                #protein.__init__(self, pdb)
                self.current_index = 0
                if test:
                        self.pdb_files = [pdb]
                        #for i in range (len(self.pdb_files)):
                        #        if pdb == self.pdb_files[i]:
                        #                self.current_index = i
                        #                break
                        print ('Protein folding starting on', self.pdb_files[self.current_index])
                else:
                        self.pdb_files = [fi for fi in os.listdir('proteins/') if fi[-4:] == '.pdb']

                self.initialize()
                if self.RENDER:
                        self.anim = animate.render(self.nres/2+1)
                self.init_args()

        def initialize(self):
                self.names = ['.'.join(pdb.split('/')[-1].split('.')[:-1]) for pdb in self.pdb_files]
                #print ('Making pdb file with the first frame (_f1.pdb) ...')
                #self.truncate_pdb()

                self.direcn = np.array([[1.0,0.0,0.0], [-1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,-1.0,0.0],
                                 [0.0,0.0,1.0], [0.0,0.0,-1.0], [1.0,1.0,1.0], [1.0,1.0,-1.0],
                                 [1.0,-1.0,1.0], [-1.0,1.0,1.0], [-1.0,-1.0,-1.0], [-1.0,-1.0,1.0],
                                 [-1.0,1.0,-1.0], [1.0,-1.0,-1.0]])

                if self.test:
                    self.res_d = np.load('models/res_d.npy').item()
                    chk = len(self.res_d)
                    self.res_arrs = [self.make_input_sequence(self.pdb_files[i], self.names[i]) for i in range (len(self.pdb_files))]
                    if chk != len(self.res_d):
                        raise Exception('Unknown residue detected !!')
                else:
                    self.res_d = {}

                    self.res_arrs = [self.make_input_sequence(self.pdb_files[i], self.names[i]) for i in range (len(self.pdb_files))]
                self.nres = len(self.res_d)#max([max(i) for i in self.res_arrs])
                #print (self.nres)
                self.ohe = self.make_ohe()

                self.current_status = [[0.0,0.0,0.0]]

                # save res_dictionary in the models
                if not self.test:
                    np.save('models/res_d.npy', self.res_d)

                
                #print ('i', self.igrid)
                # initial grid
                print ('\nUnique residues :',self.res_d)

        def make_ohe(self):
                l = self.nres
                ar = np.zeros(l)
                ohe = {}
                for i in range (l):
                        ohe[i] = np.copy(ar)
                        ohe[i][i] = 1.0
                return ohe

        def make_input_sequence(self, f, name):

                def get_sequence(lines):
                        d,rid={},{}
                        for line in lines:
                                if "TER" in line.split()[0]:
                                        break
                                if line.split()[0] in ['ATOM']:
                                        #print (line)
                                        id,at,rt,_,_0,x,y,z=line.strip().split()[1:9]
                                        s=line.strip().split()[-1]
                                        #d[len(rid)+1]=rt+_0
                                        if rt+_0 not in rid:
                                            d[len(rid)+1]=rt
                                            rid[rt+_0] = 1
                        #print (name, d)
                        arr = [d[i] for i in range (1,len(d)+1)]
                        return arr

                if self.test:
                    file = open(f,'r')
                else:
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

        def init_args(self):

		# Make final conformation grids
                self.fcords = self.make_fcords()	
                for i in range (len(self.pdb_files)):
                        if len(self.fcords[i]) != len(self.res_arrs[i]):
                                raise Exception('Multiple chains detected in protein : '+self.pdb_files[i])	
                if self.res_track == -1:
                    self.res_track = max([len(i) for i in self.res_arrs])
                state = self.reset()
                l = state.shape[0]
                self.obs_size = l

                self.n_actions = len(self.direcn)

                if self.RENDER:
                                #lis = [self.trace_r[self.current_index][t] for t in self.trace_r[self.current_index]]
                                self.save_xyz(0.0, self.fcords[self.current_index])
                                self.anim.update(self.fcords[self.current_index], 1)

        def make_fcords(self):

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
                        if self.test:
                            f = open(pdb, 'r')
                        else:
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

                        if self.test:
                            self.ref_coord = np.copy(r_coord)

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
                        

                        return pos_arr 

                lis = [get_grid(pdb) for pdb in self.pdb_files]

                return lis

        def reset(self):
                        #print(self.res_grid_pos)
                        # set dynamic coordinate to initial coordinate
                        if not self.test:
                                self.current_index = np.random.choice(range(len(self.pdb_files)))

                        self.current_status = [[0.0, 0.0, 0.0]]

                        self.nframes = 1

                        state = self.state()

                        return state

        def state(self):
                cur_res = len(self.current_status)-1
                lis = self.ohe[self.res_arrs[self.current_index][cur_res] - 1]
                # Places where next residue cannot move
                # overlap in 14 directions
                t = np.zeros(len(self.direcn))
                k = []
                for i in range (len(self.direcn)):
                    lt = self.current_status[-1]+self.direcn[i]
                    lt1 = list(lt)
                    lt2 = list(self.current_status)
                    #print (lt1, lt2)
                    if lt1 in lt2:
                        k.append(i)
                for i in k:
                    t[i] = 1.0

                # make ohe for all future res and stack
                for i in range (cur_res + 1 ,cur_res + 1 + self.fcounts):
                    if i >= len(self.fcords[self.current_index]) or i < 0:
                        lis = np.concatenate((lis, np.zeros(self.nres)))
                    else:
                        #print (len(self.fcords[self.current_index]), len(self.res_arrs[self.current_index]), i, self.pdb_files[self.current_index])#print (self.res_arrs[self.current_index])
                        lis = np.concatenate((lis, self.ohe[self.res_arrs[self.current_index][i]-1]))
                #print (np.array(np.concatenate((lis,t)), dtype = 'float').shape)
                # current vector
                l_temp = np.zeros(3)
                if len(self.current_status) > 1:
                        l_temp = np.array(self.current_status[-1]) - np.array(self.current_status[-2])
                lis = np.concatenate((np.array(l_temp), lis))

                # last n coordinates of residues
                n = self.res_track
                n_tem = np.array([])
                
                for i in range (n):
                        if cur_res-i-1 < 0:
                                l = np.zeros(self.nres)
                        else:
                                #l = [self.res_arrs[self.current_index][cur_res-i-1]]+list(self.current_status[-1 -i-1])
                                l = self.ohe[self.res_arrs[self.current_index][cur_res-i-1]-1]
                        n_tem = np.concatenate((n_tem, l))
                lis = np.concatenate((lis,n_tem))
                return np.array(np.concatenate((lis,t)), dtype = 'float').flatten()


        def get_reward(self):

            # distance from current amino acid to last two amino acids
            # distance from actual amino acid sequence
            # Mean squred error
            # bcount is how much to go backward
            #print (self.current_status)

            if self.bcount != -1 and len(self.current_status) < self.bcount + 1:
                return -0.001#None

            track = 1
            res = 0.0
            bref = self.bcount
            gamma = 0.2
            if self.bcount == -1:
                bref = len(self.current_status) - 1
            for i in range (bref):
                d1 = self.distance(self.current_status[-1], self.current_status[-1-track])
                d2 = self.distance(self.fcords[self.current_index][len(self.current_status) - 1], self.fcords[self.current_index][len(self.current_status) - 1 - track])
                #print (d1, d2, i)
                res += gamma**2*(d1-d2)**2
                track += 1
            '''   
            if res < 1.0:
                res = 1.0
            else:
                res = -1.0
            '''

            return -res

        def distance(self,a,b):
        		a = list(map(float,a))
        		b = list(map(float,b))
        		return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)



        def step(self, action):
                #ac = np.argmax(action)
                ac = action
                # last place of residue in grid
                cu_r = self.current_status[-1]
                #print (ac, cu_r )
                new_place = cu_r+self.direcn[ac]
                #print (list(new_place) == self.current_status[0])
                penalty = 0.0

                if list(new_place) in self.current_status:
                        #print ('penalty')
                        penalty = -100.0

                if penalty == 0.0:
                    self.current_status.append(list(new_place))

                if self.RENDER:
                    
                    self.anim.update(self.current_status)

                new_state = self.state()

                reward = self.get_reward()

                if reward is not None:
                    reward = reward+penalty
                    

                is_done = False

                if len(self.current_status) == len(self.fcords[self.current_index]):
                        #print ('done')
                        is_done = True
                        if self.test and self.RENDER:
                            #pass
                            self.anim.plot_final(self.current_status, self.fcords[self.current_index])
                            self.map_pdb()
                self.nframes += 1

                return new_state, reward, is_done, self.nframes


        def sample_action_space(self, index = None):
                s = np.zeros(len(self.direcn)) # N residues * 14 direcn
                if index:
                        s[index] = 1.0
                        return s
                i = np.random.randint(len(self.direcn))
                s[i] = 1.0
                return s

        def save_xyz(self, reward = 0, lis = None):

                if 'temp_cord.npy' not in os.listdir('.'):
                        d = {}
                else:
                        d = np.load('temp_cord.npy').item()

                print ('Reward:',reward)
                #print (lis)

                if lis is None:
                		lis = self.current_status

                d[len(d)] = lis#np.copy(c)

                np.save('temp_grid.npy', d)

        def map_pdb(self):

                dis = []
                for i in range (1, len(self.current_status)):
                    dis.append(self.distance(self.ref_coord[i], self.ref_coord[i-1]))

                vec = []
                for i in range (1, len(self.current_status)):
                    vec.append(np.array(self.current_status[i]) - np.array(self.current_status[i-1]))

                # place first one at 0,0,0
                cords = [[0.0,0.0,0.0]]
                for i in range (len(vec)):
                    # normalize vector
                    v = vec[i]/np.linalg.norm(vec[i])
                    new_cord = dis[i]*v + cords[i]
                    cords.append(new_cord)

                r = animate.render(np.amax(cords)/2)

                r.plot_final(cords, self.ref_coord)


        def __str__(self):
                return self.name
            
        def render(self):
                self.anim.update(self.current_status)
                



