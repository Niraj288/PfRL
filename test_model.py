import time
import numpy as np
import collections
import sys
import torch
import torch.nn as nn
import torch.optim as optim

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

from denovo import environ_grid
DEFAULT_ENV_NAME = '1k43'
pdb = '1k43.pdb'
if len(sys.argv) > 1:
	pdb = sys.argv[1]
env = environ_grid(pdb,'test', 1, 1)

state = env.reset()
total_reward = 0.0
c = collections.Counter()
RENDER = 1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

test_net = DQN(env.obs_size, 256, env.n_actions)
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

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

l1 = np.array(env.current_status)
l2 = np.array(env.fcords[env.current_index])

x1, y1, z1 = l1[:,0],l1[:,1],l1[:,2]
x2, y2, z2 = l2[:,0],l2[:,1],l2[:,2]

lines1 = ax.scatter(x1, y1, z1, c = 'r', s = 100)
lines2 = ax.plot(x1, y1, z1, c = 'r')

lines3 = ax.scatter(x2, y2, z2, c = 'g', s = 100)
lines4 = ax.plot(x2, y2, z2, c = 'g')

plt.show(block = True)
'''
