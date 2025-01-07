import gymnasium as gym
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
import random

# see https://gymnasium.farama.org/environments/classic_control/cart_pole/
# understand environment, state, action and other definitions first before your dive in.

ENV_NAME = 'CartPole-v1'

# Hyper Parameters
# Following params work well if your implement Policy Gradient correctly.
# You can also change these params.
EPISODE = 3000  # total training episodes
STEP = 5000  # step limitation in an episode
EVAL_EVERY = 10  # evaluation interval
TEST_NUM = 5  # number of tests every evaluation
GAMMA = 0.95  # discount factor
LEARNING_RATE = 3e-3  # learning rate for mlp and ac


# A simple mlp implemented by PyTorch #
# it receives (N, D_in) shaped torch arrays, where N: the batch size, D_in: input state dimension
# and outputs the possibility distribution for each action and each sample, shaped (N, D_out)
# e.g. 
# state = torch.randn(10, 4)
# outputs = mlp(state)  #  output shape is (10, 2) in CartPole-v0 Game
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class AC(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, output_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        return x

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


class REINFORCE:
    def __init__(self, env):
        # init parameters
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        self.last_state = None
        self.net = MLP(input_dim=self.state_dim, output_dim=self.action_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def predict(self, observation, deterministic=False):
        observation = torch.FloatTensor(observation).unsqueeze(0)
        action_score = self.net(observation)
        probs = F.softmax(action_score, dim=1)
        m = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = m.sample()
        return action, probs

    def store_transition(self, s, a, p, r):
        self.states.append(s)
        self.actions.append(a)
        self.action_probs.append(p)
        self.rewards.append(r)

    def learn(self):
        # Please make sure all variables used to calculate loss are of type torch.Tensor, or autograd may not work properly.
        # You need to calculate the loss of each step of the episode and store them in '''loss'''.
        # The variables you should use are: self.rewards, self.action_probs, self.actions.
        # self.rewards=[R_1, R_2, ...,R_T], self.actions=[A_0, A_1, ...,A_(T-1)]
        # self.action_probs corresponds to the probability of different actions of each timestep, see predict() for details

        loss = []
        # -------------------------------
        # Your code goes here
        # TODO Calculate the loss of each step of the episode and store them in '''loss'''

        # -------------------------------
        # 计算回报
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(self.rewards):
            cumulative_reward = r + GAMMA * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # 标准化回报
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        # 计算每个步骤的损失
        loss = []
        for t in range(len(self.states)):
            # 获取当前状态下采取的动作概率
            action_prob = self.action_probs[t]
            action = self.actions[t]

            # 对数概率
            log_prob = torch.log(action_prob.squeeze(0)[action])

            # 计算损失: -log(P(a_t|s_t)) * R_t
            loss.append(-log_prob * discounted_rewards[t])
         
        
        # code for autograd and back propagation
        self.optim.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optim.step()

        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        return loss.item()


class TDActorCritic(REINFORCE):
    def __init__(self, env):
        super().__init__(env)
        self.ac = AC(input_dim=self.state_dim, output_dim=self.action_dim)
        # override
        self.net = self.ac.pi
        self.done = None
        self.optim = torch.optim.Adam(self.ac.parameters(), lr=LEARNING_RATE)

    def make_batch(self):
        done_lst = [1.0 if i != len(self.states) - 1 else 0.0 for i in range(len(self.states))]

        self.last_state = torch.tensor(self.last_state, dtype=torch.float).reshape(1, -1)
        self.states = torch.tensor(np.array(self.states), dtype=torch.float)
        self.done = torch.tensor(done_lst, dtype=torch.float).reshape(-1, 1)
        self.actions = torch.tensor(self.actions, dtype=torch.int64).reshape(-1, 1)
        self.action_probs = torch.cat(self.action_probs)
        self.states_prime = torch.cat((self.states[1:], self.last_state))
        self.rewards = torch.tensor(self.rewards, dtype=torch.float).reshape(-1, 1) / 100.0

    def learn(self):
        # Please make sure all variables are of type torch.Tensor, or autograd may not work properly.
        # You only need to calculate the policy loss.
        # The variables you should use are: self.rewards, self.action_probs, self.actions, self.states_prime, self.states.
        # self.states=[S_0, S_1, ...,S_(T-1)], self.states_prime=[S_1, S_2, ...,S_T], self.done=[1, 1, ..., 1, 0]
        # Invoking self.ac.v(self.states) gives you [v(S_0), v(S_1), ..., v(S_(T-1))]
        # For the final timestep T, delta_T = R_T - v(S_(T-1)), v(S_T) = 0
        # You need to use .detach() to stop delta's gradient in calculating policy_loss, see value_loss for an example

        policy_loss = None
        td_target = None
        delta = None
        self.make_batch()
        # -------------------------------
        # Your code goes here
        # TODO Calculate policy_loss


        # Calculate td_target and delta
        with torch.no_grad():
            # v(S_t+1) for all time steps except the last one
            next_state_values = self.ac.v(self.states_prime)  # v(S_t+1)
            td_target = self.rewards + GAMMA * next_state_values * (1 - self.done)  # R_t+1 + γv(S_t+1)

        # Compute the advantage delta_t = td_target - v(S_t)
        state_values = self.ac.v(self.states)  # v(S_t)
        delta = td_target - state_values  # Advantage function

        # Calculate the policy loss: -log(π(a_t | s_t)) * delta_t
        # This requires action probabilities for each state-action pair
        log_probs = torch.log(self.action_probs.gather(1, self.actions))  # log(π(a_t | s_t))
        policy_loss = -(log_probs * delta).mean()  # Policy gradient loss


        
        # -------------------------------

        # compute value loss and total loss
        # td_target is used as a scalar here, and is detached to stop gradient
        value_loss = F.smooth_l1_loss(self.ac.v(self.states), td_target.detach())
        loss = policy_loss + value_loss

        # code for autograd and back propagation
        self.optim.zero_grad()
        loss = loss.mean()
        loss.backward()
        self.optim.step()

        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        return loss.item()

def plot_training_curve(losses, rewards, model_name, seeds):
    """
    Function to plot the training loss and rewards for different seeds and models
    """
    plt.figure(figsize=(12, 5))

    # Plot loss curve
    plt.subplot(1, 2, 1)
    for loss in losses:
        plt.plot(loss, label=f'{model_name} - Loss - seed {seeds[losses.index(loss)]}')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss Curve')
    plt.legend()

    # Plot reward curve
    plt.subplot(1, 2, 2)
    for reward in rewards:
        plt.plot(reward, label=f'{model_name} - Reward - seed {seeds[rewards.index(reward)]}')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'{model_name} Reward Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # initialize OpenAI Gym env and PG agent
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    train_env = gym.make(ENV_NAME)
    render_env = gym.make(ENV_NAME, render_mode="human")
    RENDER = False
    
    # You may uncomment the line below to enable rendering for visualization.
    # RENDER = True
    
    # uncomment to switch between methods
    # agent = REINFORCE(train_env)
    agent = TDActorCritic(train_env)

    for episode in range(EPISODE):
        # initialize task
        env = train_env
        state, _ = env.reset(seed=random.randint(0,int(1e10)))
        agent.last_state = state
        # Train
        for step in range(STEP):
            action, probs = agent.predict(state)
            next_state, reward, terminated, turcated, _ = env.step(action.item())
            done = terminated or turcated
            agent.store_transition(state, action, probs, reward)
            state = next_state
            if done:
                loss = agent.learn()
                break

        # Test
        env = render_env if RENDER else train_env
        if episode % EVAL_EVERY == 0:
            total_reward = 0
            for i in range(TEST_NUM):
                state, _ = env.reset(seed=random.randint(0,int(1e10)))
                for j in range(STEP):
                    action, _ = agent.predict(state, deterministic=True)
                    next_state, reward, terminated, turcated, _ = env.step(action.item())
                    done = terminated or turcated
                    total_reward += reward
                    state = next_state
                    if done:
                        break
            avg_reward = total_reward / TEST_NUM

            # Your avg_reward should reach 200(cartpole-v0)/500(cartpole-v1) after a number of episodes.
            print('episode: ', episode, 'Evaluation Average Reward:', avg_reward)

# 这里的main函数是用来使用三个随机数来进行绘制loss和reward曲线的。

# def main():
#     seeds = [11, 5, 3]  # Three different seeds
#     losses_reinforce = []
#     rewards_reinforce = []
#     losses_td_ac = []
#     rewards_td_ac = []

#     for seed in seeds:
#         # Initialize OpenAI Gym env and REINFORCE agent
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         random.seed(seed)

#         train_env = gym.make(ENV_NAME)
#         render_env = gym.make(ENV_NAME, render_mode="human")
#         agent = REINFORCE(train_env)

#         losses = []
#         rewards = []

#         print(f"Training REINFORCE with seed {seed}...")

#         for episode in range(EPISODE):
#             state, _ = train_env.reset(seed=random.randint(0, int(1e10)))
#             agent.last_state = state
#             episode_reward = 0
#             for step in range(STEP):
#                 action, probs = agent.predict(state)
#                 next_state, reward, terminated, truncated, _ = train_env.step(action.item())
#                 done = terminated or truncated
#                 agent.store_transition(state, action, probs, reward)
#                 state = next_state
#                 episode_reward += reward
#                 if done:
#                     loss = agent.learn()
#                     losses.append(loss)
#                     rewards.append(episode_reward)
#                     break

#             # Print progress every 100 episodes
#             if episode % 100 == 0:
#                 print(f"Seed {seed} - Episode {episode}/{EPISODE} - Loss: {losses[-1]:.3f} - Reward: {episode_reward:.2f}")

#         losses_reinforce.append(losses)
#         rewards_reinforce.append(rewards)

#         # Initialize TD Actor-Critic agent
#         agent = TDActorCritic(train_env)

#         losses = []
#         rewards = []

#         print(f"Training TDActorCritic with seed {seed}...")

#         for episode in range(EPISODE):
#             state, _ = train_env.reset(seed=random.randint(0, int(1e10)))
#             agent.last_state = state
#             episode_reward = 0
#             for step in range(STEP):
#                 action, probs = agent.predict(state)
#                 next_state, reward, terminated, truncated, _ = train_env.step(action.item())
#                 done = terminated or truncated
#                 agent.store_transition(state, action, probs, reward)
#                 state = next_state
#                 episode_reward += reward
#                 if done:
#                     loss = agent.learn()
#                     losses.append(loss)
#                     rewards.append(episode_reward)
#                     break

#             # Print progress every 100 episodes
#             if episode % 100 == 0:
#                 print(f"Seed {seed} - Episode {episode}/{EPISODE} - Loss: {losses[-1]:.3f} - Reward: {episode_reward:.2f}")

#         losses_td_ac.append(losses)
#         rewards_td_ac.append(rewards)

#     # Plot training results for both models
#     plot_training_curve(losses_reinforce, rewards_reinforce, 'REINFORCE', seeds)
#     plot_training_curve(losses_td_ac, rewards_td_ac, 'TDActorCritic', seeds)


if __name__ == '__main__':
    main()
