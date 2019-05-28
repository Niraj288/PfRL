import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from denovo import environ_grid
import sys

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

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # use epsilon greedy approach for explore/exploit
        if np.random.random() < epsilon:
            # use rand policy
            action = np.argmax(env.sample_action_space())
        else:
            # use Net policy
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a, dtype = torch.float).to(device)
            q_vals_v = net(state_v)
            # get idx of best action
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        #new_state = new_state

        # add to replay buffer
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    # Two networks are passed in, one we are updateing
    #  and another that is a previous version
    #  we use the previous network to observe Q(s,a)
    
    # send the observed states to Net 
    states_v = torch.tensor(states, dtype = torch.float).to(device)
    next_states_v = torch.tensor(next_states, dtype = torch.float).to(device)
    actions_v = torch.tensor(actions, dtype = torch.long).to(device)
    rewards_v = torch.tensor(rewards, dtype = torch.float).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    # get the Network actions for given states
    #  but only for states that did not end in a 'done' state
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0 # ensures these are only rewards
    
    # detach the calculation we just made from computation graph
    #  we don't want to back-propagate through this calculation
    #  because it is just observations that we want to be true
    #  That is, we want to change the expected values output from 
    #  the net, not the observations calculation
    next_state_values = next_state_values.detach()

    # calc the Q function behavior we want
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    
    # compare what we have to what we want
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

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
for i in params:
    print (i, params[i])

DEFAULT_ENV_NAME = params['DEFAULT_ENV_NAME']#"1k43"
MEAN_REWARD_BOUND = eval(params['MEAN_REWARD_BOUND'])#-3.0
RENDER = eval(params['RENDER'])#0

FCOUNTS = eval(params['FCOUNTS'])#10
BCOUNT = eval(params['BCOUNT'])#-1
TRACK = eval(params['TRACK'])#5 # how much residue coordinates be included from generated sequence

env = environ_grid('1k43.pdb', DEFAULT_ENV_NAME, RENDER, 0, TRACK, FCOUNTS, BCOUNT)

GAMMA = eval(params['GAMMA'])#0.99
BATCH_SIZE = eval(params['BATCH_SIZE'])#32
REPLAY_SIZE = eval(params['REPLAY_SIZE'])#10000
LEARNING_RATE = eval(params['LEARNING_RATE'])#1e-4
SYNC_TARGET_FRAMES = eval(params['SYNC_TARGET_FRAMES'])#1000
REPLAY_START_SIZE = eval(params['REPLAY_START_SIZE'])#10000

EPSILON_DECAY_LAST_FRAME = eval(params['EPSILON_DECAY_LAST_FRAME'])#10**6
EPSILON_START = eval(params['EPSILON_START'])#1.0
EPSILON_FINAL = eval(params['EPSILON_FINAL'])#0.05

MAX_ITER = eval(params['MAX_ITER'])#10**9


Experience = collections.namedtuple('Experience', 
                                    field_names=['state', 'action', 'reward', 'done', 'new_state'])


device = params['device']#'cpu'

HIDDEN_SIZE = eval(params['HIDDEN_SIZE'])#256

#env = make_env(DEFAULT_ENV_NAME)

net = DQN(env.obs_size, HIDDEN_SIZE, env.n_actions).to(device)
#net.load_state_dict(torch.load("models/" +DEFAULT_ENV_NAME + "-best.dat", map_location=lambda storage, loc: storage))
tgt_net = DQN(env.obs_size, HIDDEN_SIZE, env.n_actions).to(device)
print(net)

buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent(env, buffer)
epsilon = EPSILON_START

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
total_rewards = []
frame_idx = 0
ts_frame = 0
ts = time.time()
best_mean_reward = None




while True:
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

    reward = agent.play_step(net, epsilon, device=device)
    if reward is not None:
        total_rewards.append(reward)
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame = frame_idx
        ts = time.time()
        mean_reward = np.mean(total_rewards[-100:])
        if len(total_rewards)%100 == 0:
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
            frame_idx, len(total_rewards), mean_reward, epsilon,
            speed
        ))
        
        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(),"models/" + DEFAULT_ENV_NAME + "-best.dat")
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
            best_mean_reward = mean_reward
        if mean_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break

    if frame_idx >= MAX_ITER:
        print ('Maximum iteration reached')
        break

    if len(buffer) < REPLAY_START_SIZE:
        continue

    if frame_idx % SYNC_TARGET_FRAMES == 0:
        tgt_net.load_state_dict(net.state_dict())

    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss_t = calc_loss(batch, net, tgt_net, device=device)
    loss_t.backward()
    optimizer.step()
