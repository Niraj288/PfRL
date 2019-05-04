import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
#import gym
from protein import environ

GAMMA = 0.9


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


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
            action = self.env.sample_action_space()
        else:
            # use Net policy
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a, dtype = torch.float).to(device)
            q_vals_v = net(state_v)
            # get idx of best action
            _, act_v = torch.max(q_vals_v, dim=1)
            action = self.env.sample_action_space(int(act_v.item()))
            #print (action, 'action') 
	# do step in the environment
        new_state, reward, is_done = self.env.step(action)
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
    # print (net(states_v).view(-1,4248,1).shape, actions_v.unsqueeze(-1).shape)
    state_action_values = net(states_v).view(-1, actions_v.shape[1], 1).gather(1, actions_v.unsqueeze(-1)).squeeze(-1).max(1)[0]
    next_state_values = tgt_net(next_states_v).max(1)[0]
    #print ('Working')
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


DEFAULT_ENV_NAME = "Protein folding"

env = environ('1k43.pdb',DEFAULT_ENV_NAME)

print (env)

obs_size = env.obs_size
n_actions = env.n_actions

print(obs_size,n_actions)

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

net = Net(obs_size, HIDDEN_SIZE, n_actions)
tgt_net = Net(obs_size, HIDDEN_SIZE, n_actions)
print(net)


print(obs_size,n_actions)

device = "cpu"

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.0

MEAN_REWARD_BOUND = 150
SYNC_TARGET_FRAMES = 50
BATCH_SIZE = 16
REPLAY_SIZE = 500
REPLAY_START_SIZE = 500
LEARNING_RATE = 1e-4

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
    # track epsilon and cool it down
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

    # play step and add to experience buffer
    reward = agent.play_step(net, epsilon, device=device)
    if reward is not None:
        total_rewards.append(reward)
        ts_frame = frame_idx
        
        # calculate progress of rewards
        mean_reward = np.mean(total_rewards[-100:])
        if frame_idx % 100==0:
            print("%d: done %d iterations, mean reward %.3f, eps %.2f" % (
                frame_idx, len(total_rewards), mean_reward, epsilon
            ))
        # save best model thus far
        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(), "models/model-best.dat")
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
            best_mean_reward = mean_reward
            
        # quit if we have solved the problem
        if mean_reward > 0.8:
            print("Solved in %d frames!" % frame_idx)
            break

    # check to see if Agent has played enough rounds
    if len(buffer) < REPLAY_START_SIZE:
        continue

    # sync the networks evry so often
    if frame_idx % SYNC_TARGET_FRAMES == 0:
        tgt_net.load_state_dict(net.state_dict())

    # use experience buffer and two networks to get loss 
    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE) # grab some examples from buffer
    loss_t = calc_loss(batch, net, tgt_net, device=device)
    loss_t.backward()
    optimizer.step()




