from collections import deque
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit

def greedy_action(Q,s,nb_actions):
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s,a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return np.argmax(Qsa)

class ProjectAgentTrainingFQI:
    def __init__(self, env, horizon, max_length):
        self.horizon = horizon
        self.max_length = max_length
        self.env = env
        self.Qfunctions = []
        self.best_episode_reward = 0

    def collect_samples(self, max_length, horizon):
        s, _ = self.env.reset()
        self.S = deque(maxlen=max_length)
        self.A = deque(maxlen=max_length)
        self.R = deque(maxlen=max_length)
        self.S2 = deque(maxlen=max_length)
        self.D = deque(maxlen=max_length)
        for _ in range(horizon):
            a = self.env.action_space.sample()
            s2, r, done, trunc, _ = self.env.step(a)
            self.S.append(s)
            self.A.append(a)
            self.R.append(r)
            self.S2.append(s2)
            self.D.append(done)
            if done or trunc:
                s, _ = self.env.reset()
            else:
                s = s2

    def fit_q(self, n_iter,start = False, gamma=0.99):
        nb_samples = len(self.S)
        nb_actions = self.env.action_space.n
        S = np.array(self.S)
        A = np.array(self.A).reshape(-1,1)
        SA = np.append(S, A, axis=1)
        for _ in range(n_iter):
            if start == True:
                value=self.R.copy()
            else:
                Q2 = np.zeros((nb_samples, nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2 * np.ones((len(self.S), 1))
                    S2 = np.array(self.S2)
                    S2A2 = np.append(S2, A2, axis=1)
                    Q2[:, a2] = self.Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = np.array(self.R) + gamma * (1 - np.array(self.D)) * max_Q2
            Q = ExtraTreesRegressor(n_estimators=100, max_depth=30)
            Q.fit(SA, value)
            self.Qfunctions.append(Q)

            cum_reward = self.evaluate(self.env, Q)
            print("reward = ", cum_reward)

        return self.Qfunctions

    def train(self, nb_steps, n_iter, gamma, size_start = 10000, epsilon=0.20):
        self.collect_samples(self.max_length, size_start)
        self.fit_q(n_iter, gamma= gamma, start = True)
        for _ in range(nb_steps):
            self.online_buffer(self.horizon, self.env, epsilon)
            print(len(self.S), self.S[0].shape)
            self.fit_q(n_iter, gamma = gamma, start = False)
            ep_reward = self.evaluate(self.env, self.Qfunctions[-1], 1)[0]

            if ep_reward > self.best_episode_reward:
                self.best_episode_reward = ep_reward
                joblib.dump(self.Qfunctions[-1], 'best_et_model.joblib')

    def resume_train(self,nb_steps, n_iter, gamma, epsilon=0.20):
        for _ in range(nb_steps):
            self.online_buffer(self.horizon, self.env, epsilon)
            self.fit_q(n_iter, gamma = gamma, start = False)
            ep_reward = self.evaluate(self.env, self.Qfunctions[-1])

            if ep_reward > self.best_episode_reward:
                self.best_episode_reward = ep_reward
                joblib.dump(self.Qfunctions[-1], 'best_et_model.joblib')
        return self.Qfunctions
    
    def online_buffer(self, horizon, env, epsilon=0.2):
        state, _ = env.reset()
        for _ in range(horizon):
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = greedy_action(self.Qfunctions[-1], state, env.action_space.n)

            next_state, reward, done, trunc, _ = env.step(a)
            self.S.append(state)
            self.A.append(a)
            self.R.append(reward)
            self.S2.append(next_state)
            self.D.append(done)
            if done or trunc:
                state, _ = env.reset()
            else:
                state = next_state

    def evaluate(self, env, Qfunction):
        s, _ = env.reset()
        cum_reward = 0
        for _ in range(200):
            a = greedy_action(Qfunction, s, env.action_space.n)
            s2, r, d, trunc, _ = env.step(a)
            s = s2
            cum_reward += r
            if d or trunc:
                break
        return cum_reward
    

if __name__ == "__main__":
    gamma = 0.98
    nb_iter = 1
    n_steps = 100
    horizon = 2000
    max_length = 40000
    env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
    agent= ProjectAgentTrainingFQI(env, horizon, max_length)
    agent.train(n_steps, nb_iter, gamma, size_start = 5000, epsilon=0.10)