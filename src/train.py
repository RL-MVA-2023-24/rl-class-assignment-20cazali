from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from torch.distributions.categorical import Categorical

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from gymnasium.wrappers import TimeLimit
import random
import joblib
from sklearn.ensemble import ExtraTreesRegressor

from gymnasium.wrappers import TimeLimit
from torch.distributions.categorical import Categorical


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

# %load solutions/dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from replay_buffer import ReplayBuffer
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, state_dim, n_action, nb_neurons):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, nb_neurons)
        self.bn1 = nn.BatchNorm1d(nb_neurons,track_running_stats=False)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(nb_neurons, nb_neurons)
        self.bn2 = nn.BatchNorm1d(nb_neurons,track_running_stats=False)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(nb_neurons, nb_neurons)
        self.bn3 = nn.BatchNorm1d(nb_neurons,track_running_stats=False)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc4 = nn.Linear(nb_neurons, n_action)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x=self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x=self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x=self.dropout(x)
        
        x = self.fc4(x)
        return x
class RNDUncertainty:
    """ This class uses Random Network Distillation to estimate the uncertainty/novelty of states. """
    def __init__(self, scale, env, hidden_dim=1024, embed_dim=256, **kwargs):
        self.scale = scale
        self.criterion = torch.nn.MSELoss(reduction='none')
        # YOUR CODE HERE
        self.target_net = torch.nn.Sequential(torch.nn.Linear(env.observation_space.shape[0], hidden_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_dim, embed_dim))
        self.predict_net = torch.nn.Sequential(torch.nn.Linear(env.observation_space.shape[0], hidden_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, embed_dim))
        self.optimizer = torch.optim.Adam(self.predict_net.parameters())
    
    def error(self, state):
        """ Computes torche error between torche prediction and target network. """
        if not isinstance(state, torch.Tensor): state = torch.tensor(state, dtype=torch.float32)
        if len(state.shape) == 1: state.unsqueeze(dim=0)
        # YOUR CODE HERE: return torche RND error
        return self.criterion(self.predict_net(state), self.target_net(state))
    
    def observe(self, state, **kwargs):
        """ Observes state(s) and 'remembers' torchem using Random Network Distillation"""
        # YOUR CODE HERE
        self.optimizer.zero_grad()
        self.error(state).mean().backward()
        self.optimizer.step()
    
    def __call__(self, state, **kwargs):
        """ Returns the estimated uncertainty for observing a (minibatch of) state(s) as Tensor. """
        # YOUR CODE HERE
        return self.scale * self.error(state).mean(dim=-1)
    
def greedy_action(Q,s,nb_actions):
    Qsa = predict_state_action(Q, s, nb_actions)
    return np.argmax(Qsa)

def predict_state_action(Q, state, nb_actions):
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(state,a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return Qsa

class ProjectAgentFQ:
    def __init__(self, env) -> None:
        self.nb_actions = env.action_space.n
    def load(self, paths):
        self.qfunctions = []
        for path in paths:
            self.qfunctions.append(joblib.load(path))
    
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            actions = 0
            for model in self.qfunctions:
                action = predict_state_action(model, observation, nb_actions=self.nb_actions)
                action = (action - np.min(action))/(np.max(action)-np.min(action)) 
                actions += action
            return np.argmax(actions)
        
    def save(self, path):
        joblib.dump(self.model, path)

class ProjectAgent:
    def __init__(self, config, uncertainty = None):
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = self.mlp(config['state_dim'], self.nb_actions, config['nb_neurons']) 
        self.device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        self.memory = ReplayBuffer(buffer_size,self.device)
        self.target_model = deepcopy(self.model).to(self.device) if config['target_copy'] else self.build_network(config['state_dim'], self.nb_actions, config['nb_neurons']).to(self.device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer_target = None if config['target_copy'] else torch.optim.Adam(self.target_model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.uncertainty = uncertainty

    def mlp(self, state_dim, n_action, nb_neurons):
        model = torch.nn.Sequential(torch.nn.Linear(state_dim, nb_neurons),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(nb_neurons, nb_neurons),
                                    torch.nn.ReLU(), 
                                    torch.nn.Linear(nb_neurons, nb_neurons),
                                    torch.nn.ReLU(), 
                                    torch.nn.Linear(nb_neurons, n_action))
        return model


    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = self.greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(self.device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    def gradient_step_double(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            next_action = self.model(Y).argmax(1).unsqueeze(1)
            if self.uncertainty is not None:
                R += self.uncertainty(Y).detach()
            QYmax = self.target_model(Y).gather(1, next_action).detach()
            update = torch.addcmul(R.unsqueeze(1), (1-D).unsqueeze(1), QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def train(self, env, max_episode):
        best_episode_return = 0
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            next_states = []
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            next_states.append(next_state)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step_double()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
              if self.uncertainty is not None:
                self.uncertainty.observe(next_states)
              if episode_cum_reward > best_episode_return:
                best_episode_return = episode_cum_reward
                self.save("best_model.pt")
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.q_function = []
        for path_ in path:
            self.model.load_state_dict(torch.load(path_))
            self.q_function.append(deepcopy(self.model))

    def act(self, observation, use_random=False):

        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            with torch.no_grad():
                return self.act_ensembling(observation)
            
    def act_ensembling(self, observation, use_random=False):
        actions = 0
        for q_function in self.q_function:
            if use_random:
                return np.random.randint(self.nb_actions)
            else:
                with torch.no_grad():
                    q_function.eval()
                    action_q = q_function(torch.Tensor(observation).unsqueeze(0))
                    action_q_normalized = (action_q - action_q.mean())/action_q.std()
                    actions +=  action_q_normalized
        if np.random.random()<-1:
            return np.random.randint(self.nb_actions)
        else:
            return actions.argmax().item()

                
    def greedy_action(self, model, observation):
        with torch.no_grad():
            return model(torch.Tensor(observation).unsqueeze(0)).argmax().item()