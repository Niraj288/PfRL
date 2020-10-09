import time
import numpy as np
import collections
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from denovo import environ_grid

class DQN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(DQN, self).__init__()
        def init_weights(m):
            if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform(m.weight)
                    m.bias.data.fill_(0.0)
            
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            #nn.Linear(int(hidden_size/2), int(hidden_size/4)),
	        #   nn.ReLU(),
            nn.Linear(int(hidden_size/2), n_actions)
        )
        #self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class DQN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(DQN, self).__init__()
        def init_weights(m):
            if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform(m.weight)
                    m.bias.data.fill_(0.0)

        self.net = nn.Sequential(
            nn.Linear(obs_size, n_actions)
            #nn.ReLU(),
            #nn.Linear(hidden_size, int(hidden_size/2)),
            #nn.ReLU(),
            #nn.Linear(int(hidden_size/2), int(hidden_size/4)),
                #   nn.ReLU(),
            #nn.Linear(int(hidden_size/2), n_actions)
        )
        #self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

def read_inp():
    inp = 'inp'
    if len(sys.argv) > 1:
        inp = sys.argv[1]
    f=open(inp,'r')
    lines=f.readlines()
    f.close()
    dic={}
    for line in lines:
        if '#' in line or len(line.strip().split())==0:
            continue
        a,b=line.strip().split()
        dic[a]=b 
    return dic

params = read_inp()


DEFAULT_ENV_NAME = '1k43'
pdb = '1k43.pdb'
if len(sys.argv) > 2:
	pdb = sys.argv[2]

RENDER = 1
test = 1

if int(params['doSimulation']):
    print ('No rendering will be done as simulation is on')
    RENDER = 0

DEFAULT_ENV_NAME = params['DEFAULT_ENV_NAME']
FCOUNTS = eval(params['FCOUNTS'])#10
BCOUNT = eval(params['BCOUNT'])#-1
TRACK = eval(params['TRACK'])#5

HIDDEN_SIZE = eval(params['HIDDEN_SIZE'])

env = environ_grid(pdb, DEFAULT_ENV_NAME, RENDER, test, TRACK, FCOUNTS, BCOUNT)

state = env.reset()
total_reward = 0.0
c = collections.Counter()
RENDER = 1
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

test_net = DQN(env.obs_size, HIDDEN_SIZE, env.n_actions)
print (test_net)
test_net.load_state_dict(torch.load("models/" +DEFAULT_ENV_NAME + "-best.dat", map_location=lambda storage, loc: storage))


while True:
    start_ts = time.time()
    state_v = torch.tensor(np.array([state], copy=False), dtype = torch.float)
    q_vals = test_net(state_v).data.numpy()[0]
    action = np.argmax(q_vals)
    c[action] += 1
    state, reward, done, _ = env.step(action)
    print (reward)
    total_reward += reward
    if done:
        break
print("Total reward: %.2f" % total_reward)
print("Action counts:", c)

if int(params['doSimulation']):
    env.map_pdb()


