import time
import numpy as np
import torch.nn as nn
import torch
from protein import environ
import collections
import os

DEFAULT_ENV_NAME = "Protein folding"
device = "cpu"
env = environ('1k43.pdb',DEFAULT_ENV_NAME)

print (env)

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), n_actions)
        )

    def forward(self, x):
        return self.net(x)

HIDDEN_SIZE = 256

obs_size = env.obs_size
n_actions = env.n_actions

test_net = Net(obs_size, HIDDEN_SIZE, n_actions)
test_net.load_state_dict(torch.load("models/model-best.dat", map_location=lambda storage, loc: storage))

RENDER = 1

if RENDER:
	os.system('remove -rf render.xyz')

state = env.reset()
total_reward = 0.0
c = collections.Counter()

while True:
    start_ts = time.time()
    if RENDER:
        env.save_xyz(total_reward)
    state_v = torch.tensor(np.array([state], copy=False), dtype = torch.float).to(device)
    q_vals = test_net(state_v)
    _, act_v = torch.max(q_vals, dim=1)
    action = env.sample_action_space(int(act_v.item()))
    #action = np.argmax(q_vals)
    c[np.argmax(action)] += 1
    state, reward, done = env.step(action)
    total_reward += reward
    if done:
        break
    if RENDER:# too fast without FPS limiter
        delta = 1/30 - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
print("Total reward: %.2f" % total_reward)
print("Action counts:", c)






















