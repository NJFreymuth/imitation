"""Trains BC, GAIL and AIRL models on saved CartPole-v1 demonstrations."""

import pickle

import seals  # noqa: F401
import stable_baselines3 as sb3

from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util

# Load pickled test demonstrations.
with open("tests/testdata/expert_models/cartpole_0/rollouts/final.pkl", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

# Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
# This is a more general dataclass containing unordered
# (observation, actions, next_observation) transitions.
transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env("seals/CartPole-v0", n_envs=1)

algorithm = "bc"

algorithm_logger = logger.configure(f"recordings/{algorithm}")
if algorithm == "bc":
    # todo start with this!
    # Train BC on expert data.
    # BC also accepts as `demonstrations` any PyTorch-style DataLoader that iterates over
    # dictionaries containing observations and actions.
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        custom_logger=algorithm_logger,
    )
    bc_trainer.train(n_epochs=10)
    #  bc_trainer.policy is the trained policy. Can be evaluated on different environments
    # we probably want to implement
    # just calling this repeatedly seems to work, but not intended
    # todo look at this more closely bc_trainer.train(n_epochs=10)

elif algorithm == "gail":
    # Train GAIL on expert data.
    # GAIL, and AIRL also accept as `demonstrations` any Pytorch-style DataLoader that
    # iterates over dictionaries containing observations, actions, and next_observations.
    gail_trainer = gail.GAIL(
        venv=venv,  # gives the environment, but overwrites the reward part of the step() to a learned reward
        demonstrations=transitions,  # expert demonstrations (todo format etc.!)
        demo_batch_size=32,  # number of expert samples per batch
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
        # policy to create the data, todo: use SAC instead? read corresponding paper
        # algorithm that is used to generate the learner data
        custom_logger=algorithm_logger,  # todo look at this
        # can take a reward_net as a parameter, which may be any pytorch module of states and actions that outputs
        # a scalar
    )
    gail_trainer.train(total_timesteps=2048)

elif algorithm.lower() == "airl":
    # Train AIRL on expert data.
    airl_trainer = airl.AIRL(
        venv=venv,
        demonstrations=transitions,
        demo_batch_size=32,
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
        custom_logger=algorithm_logger,
    )
    airl_trainer.train(total_timesteps=2048)

else:
    raise ValueError(f"Unknown algorithm '{algorithm}'")
