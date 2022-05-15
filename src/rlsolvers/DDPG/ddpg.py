# Import the DDPG algorithm from RLlib
import ray
from ray import tune
import ray.rllib.agents.ddpg as ddpg
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.tune.logger import pretty_print

ray.init()

tune.run(
    "DDPG",

    # Stopping condition
    stop={"episode_reward_mean":200},

    # Config
    # The default DDPG specific config is used with required 
    # Options for the config are in the default DDPG config: 
    # https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ddpg
    config={
        "env": "Pendulum-v1",
        "framework": "torch",
        "num_gpus":0,
        "num_workers":1,
        "lr":tune.grid_search([0.01, 0.001, 0.0001]),
    },
)

# OTHER CODE IDK IF IT'S USEFUL
#
# config = ddpg.DEFAULT_CONFIG.copy()
# config["num_gpus"] = 0
# config["num_workers"] = 1

# trainer = DDPGTrainer(config=config, env="Pendulum-v1")

# # Number of training iterations
# n = 3
# for i in range(n):
#     # Train the trainer and print the results of training
#     result = trainer.train()
#     print(pretty_print(result))

#     # Save model checkpoints if needed
#     if i%n == 0:
#         checkpoint = trainer.save()
#         print("Checkpoint saved at iteration ", i)
