import random
import os
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from evaluate import evaluate_HIV, evaluate_HIV_population
from train import ProjectAgent, ProjectAgentFQ  # Replace DummyAgent with your agent implementation
from config import CONFIG
from env_hiv import HIVPatient
from train import RNDUncertainty
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    # Declare network

    env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200)
    #uncertainty = RNDUncertainty(400, env)
    #agent = ProjectAgent(CONFIG)
    # agent.train(env, 10)
    agent = ProjectAgentFQ(env)
    agent.load(["model_compressed.joblib"])
    #agent.load(["best_model_norandom_35md.pt"])#"best_model_batchnorm_25md.pt"])#,)"best_modelbis_25.pt","best_model_norandom_19md.pt","best_model_pretrain_no_random_34md.pt"]) #"best_model_prerandom_1_7.pt"]) #best_modelbis_25.pt", 

    # Keep the following lines to evaluate your agent unchanged.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")
