from src.td3_code.networks.dp_critics import LSTMCritic, MLPCritic
from src.td3_code.networks.dp_actors import LSTMActor, MLPActor
from src.td3_code.util import parse_sample_dict
import torch.nn.functional as F
import torch
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('USING DEVICE %s' % DEVICE)


class TD3Agent(object):

    def __init__(self, observation_dim, action_dim, max_action_val, lstm_dims_list, hidden_dims_list,
                 actor_lr=3e-4, critic_lr=3e-4,  discount=0.99, tau=0.005,
                 policy_noise=0.2, min_policy_noise=0.02, noise_clip=0.5, noise_decay=0.9995, policy_update_freq=2):
        self.on_policy = False
        self._create_networks(observation_dim, action_dim, max_action_val, lstm_dims_list,
                              hidden_dims_list, actor_lr, critic_lr)
        # Save parameters
        self.critic_updates = 0
        self.max_action_val = max_action_val
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.min_policy_noise = min_policy_noise
        self.noise_clip = noise_clip
        self.noise_decay = noise_decay
        self.policy_update_freq = policy_update_freq

    def _create_networks(self, observation_dim, action_dim, max_action_val,
                         lstm_dims_list, hidden_dims_list, actor_lr, critic_lr, ):
        # Create actor
        if len(lstm_dims_list):
            self.actor = LSTMActor(observation_dim, action_dim, max_action_val,
                                   lstm_dims_list, hidden_dims_list).to(DEVICE)
        else:
            self.actor = MLPActor(observation_dim, action_dim, max_action_val, hidden_dims_list).to(DEVICE)
            # self.actor = Actor(observation_dim, action_dim, max_action_val, hidden_dims_list).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters_list), lr=actor_lr)
        # Create critic
        if len(lstm_dims_list):
            self.critic1 = LSTMCritic(observation_dim, action_dim, lstm_dims_list, hidden_dims_list).to(DEVICE)
            self.critic2 = LSTMCritic(observation_dim, action_dim, lstm_dims_list, hidden_dims_list).to(DEVICE)
        else:
            self.critic1 = MLPCritic(observation_dim, action_dim, hidden_dims_list).to(DEVICE)
            # self.critic1 = Critic(observation_dim, action_dim, hidden_dims_list).to(DEVICE)
            self.critic2 = MLPCritic(observation_dim, action_dim, hidden_dims_list).to(DEVICE)
            # self.critic2 = Critic(observation_dim, action_dim, hidden_dims_list).to(DEVICE)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critics_optimizer = torch.optim.Adam(self.critic1.parameters_list + self.critic2.parameters_list,
                                                  lr=critic_lr)


    def select_action(self, observation, hidden_state=None, use_target=False, noisy=True):
        actor_to_use = self.actor_target if use_target else self.actor
        hidden_state = actor_to_use.get_initial_state(observation.shape[0]) if hidden_state is None else hidden_state
        action, hidden_state = actor_to_use(observation, hidden_state)
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) if noisy else 0
        action = action + noise
        return action.clamp(-self.max_action_val, self.max_action_val), hidden_state

    def evaluate_q1(self, observation, action, hidden_state=None, use_target=True):
        critic_to_use = self.critic1_target if use_target else self.critic1
        hidden_state = critic_to_use.get_initial_state(observation.shape[0]) if hidden_state is None else hidden_state
        q, hidden_state = critic_to_use(observation, action, hidden_state)
        return q, hidden_state

    def evaluate_q2(self, observation, action, hidden_state=None, use_target=True):
        critic_to_use = self.critic2_target if use_target else self.critic2
        hidden_state = critic_to_use.get_initial_state(observation.shape[0]) if hidden_state is None else hidden_state
        q, hidden_state = critic_to_use(observation, action, hidden_state)
        return q, hidden_state

    def train(self, observations_extended, actions, rewards, dones, reset_mask_extended):
        # Data should be of the shape (batch, time, vshape)
        # 'extended' variables are one timestep longer
        initial_hidden_state = self.actor.get_initial_state(observations_extended.shape[0])
        assert len(observations_extended.shape) == 3, 'Train must receive 3 dimensional data (batch, time, vshape)'
        # Reduce the noise scale with noise decay
        if self.policy_noise > self.min_policy_noise:
            self.policy_noise = self.policy_noise * self.noise_decay
            self.noise_clip = self.noise_clip * self.noise_decay
        # Compute target q values for the observations
        with torch.no_grad():
            target_actions_extended, _ = self.select_action(observations_extended, initial_hidden_state,
                                                            noisy=True, use_target=True)
            q1_values_extended, _ = self.evaluate_q1(observations_extended, target_actions_extended,
                                                     initial_hidden_state, use_target=True)
            q2_values_extended, _ = self.evaluate_q2(observations_extended, target_actions_extended,
                                                     initial_hidden_state, use_target=True)
            q_values_next = torch.min(q1_values_extended, q2_values_extended)[:, 1:]
            target_q_values = rewards + (1 - dones) * self.discount * q_values_next
        # Optimize critics
        q1_values, _ = self.evaluate_q1(observations_extended[:, :-1], actions, initial_hidden_state, use_target=False)
        q2_values, _ = self.evaluate_q2(observations_extended[:, :-1], actions, initial_hidden_state, use_target=False)
        critics_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(q2_values, target_q_values)
        self.critics_optimizer.zero_grad()
        critics_loss.backward()
        self.critics_optimizer.step()
        self.critic_updates += 1
        # Optimize actor and update target networks if needed
        if self.critic_updates % self.policy_update_freq == 0:
            best_actions, _ = self.select_action(observations_extended[:, :-1], initial_hidden_state,
                                                 noisy=False, use_target=False)
            q1_values, _ = self.evaluate_q1(observations_extended[:, :-1], best_actions,
                                            initial_hidden_state, use_target=False)
            actor_loss = -q1_values.mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # Update the target networks
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()
