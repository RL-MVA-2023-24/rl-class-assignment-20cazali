from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit
# DQN config
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


CONFIG = {"nb_actions": env.action_space.n,
          "state_dim":env.observation_space.shape[0],
          "learning_rate": 0.001,
          "gamma": 0.95,
          "buffer_size": 10000,
          "epsilon_min": 0.05,
          "epsilon_max": 1.,
          "epsilon_decay_period": 1000,
          "epsilon_delay_decay": 20,
          "batch_size": 200,
          "target_copy" : True,
          "nb_neurons": 512,
          "update_target_freq" : 100,
          "update_target_strategy":"replace",
          "gradient_steps":5}