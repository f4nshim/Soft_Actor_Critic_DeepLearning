import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import pybullet_envs
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model', default="v1", choices=['v1', 'v2'],  help='The version of the SAC model to use. Either \'v1\' or \'v2\'.')
parser.add_argument('--env-name', default="HalfCheetahBulletEnv-v0", help='environment name')
parser.add_argument('--eval', type=bool, default=True, help='evaluates policy')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount for reward')
parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='target smoothing coefficient')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G', help='learning rate')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G', help='determines the importance of the entropy term')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G', help='parameter alpha adjusted')
parser.add_argument('--seed', type=int, default=456, metavar='N', help='random seed')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N', help='maximum of steps')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', help='model updates per simulator step')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N', help='steps sampling')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='value target update')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N', help='capacity')
parser.add_argument('--cuda', action="store_true", help='CUDA')

args = parser.parse_args()

# Seed
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
if args.model == 'v1':
    from sac_v1 import SAC
else:
    from sac_v2 import SAC
agent = SAC(env.observation_space.shape[0], env.action_space, args)
writer = SummaryWriter('runs/{}_SAC_V_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name))

# ReplayMemory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Setting
sum_num_steps = 0
updates = 0
episode = 1

while True:
    ep_reward = 0
    ep_steps = 0
    done = False
    state = env.reset()

    while not done:

        action = env.action_space.sample() if args.start_steps > sum_num_steps else agent.select_action(state)

        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):
                # Update parameters
                value_loss, critic1_loss, critic2_loss, policy_loss = agent.learn(memory, args.batch_size, updates)
                # write log
                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/critic_1', critic1_loss, updates)
                writer.add_scalar('loss/critic_2', critic2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                updates += 1
        next_state, reward, done, _ = env.step(action)
        ep_steps += 1
        sum_num_steps += 1
        ep_reward += reward

        if ep_steps == env._max_episode_steps:
            mask = 1
        else:
            mask = float(not done)

        memory.remember(state, action, reward, next_state, mask)
        state = next_state

    if sum_num_steps > args.num_steps:
        break

    writer.add_scalar('reward/train', ep_reward, episode)
    print(f"episode: {episode},  num steps sum: {sum_num_steps}, episode steps: {ep_steps}, reward: {round(ep_reward, 2)}")

    if episode % 10 == 0 and args.eval == True:
        avg_reward = 0.0
        ep = 10
        for _ in range(ep):
            state = env.reset()
            ep_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                state = next_state
            avg_reward += ep_reward
        avg_reward /= ep

        writer.add_scalar('reward/test', avg_reward, episode)
        print(f"Test Episodes: {ep}, Average Reward: {round(avg_reward, 2)}")

    episode += 1
env.close()
