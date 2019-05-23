import sys
import time
import numpy as np
import torch.nn as nn
import torch
#from multiple_protein import environ_grid
#from protein import environ, environ_coord, environ_grid
from denovo import environ_grid
import collections
import os

RENDER = 1

test = 1

if RENDER:
        os.system('rm -rf temp_grid.npy')

DEFAULT_ENV_NAME = "Protein folding"
device = "cpu"

if len(sys.argv) > 1:
        pdb = sys.argv[1]
else:
        pdb = '1k43.pdb'

env = environ_grid(pdb,DEFAULT_ENV_NAME,RENDER, test)

print (env)

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            #nn.ReLU(),
            #nn.Linear(int(hidden_size/2), int(hidden_size/4)),
	    nn.ReLU(),
            nn.Linear(int(hidden_size/2), n_actions)
        )

    def forward(self, x):
        return self.net(x)

HIDDEN_SIZE = 500

obs_size = env.obs_size
n_actions = env.n_actions

test_net = Net(obs_size, HIDDEN_SIZE, n_actions)
test_net.load_state_dict(torch.load("models/model-best.dat", map_location=lambda storage, loc: storage))


state = env.reset()
final_reward = -99999999
c = collections.Counter()

while True:
    start_ts = time.time()
    state_v = torch.tensor(np.array([state], copy=False), dtype = torch.float).to(device)
    q_vals = test_net(state_v)
    _, act_v = torch.max(q_vals, dim=1)
    action = env.sample_action_space(int(act_v.item()))
    #action = np.argmax(q_vals)
    c[np.argmax(action)] += 1
    state, reward, done = env.step(action)
    if reward and final_reward < reward: # += reward
        final_reward = reward
    #print (reward)
    if 1 or RENDER:
        env.save_xyz(reward)
    if done:
        break
print("Max reward: %.2f" % final_reward)
print("Action counts:", c)






















