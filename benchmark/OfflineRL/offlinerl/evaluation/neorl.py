import multiprocessing
import gym
import ray
from copy import deepcopy
import numpy as np
from collections import OrderedDict

from offlinerl.utils.env import get_env


# @ray.remote(num_gpus=1)
@ray.remote
def test_one_trail(env, policy):
    # env = deepcopy(env)
    # policy = deepcopy(policy)

    state, done = env.reset(), False
    rewards = 0
    lengths = 0
    while not done:
        state = state[np.newaxis]
        action = policy.get_action(state).reshape(-1)
        state, reward, done, _ = env.step(action)
        rewards += reward
        lengths += 1

    return (rewards, lengths)


def test_one_trail_local(env, policy):
    # env = deepcopy(env)
    # policy = deepcopy(policy)

    state, done = env.reset(), False
    rewards = 0
    lengths = 0
    while not done:
        state = state[np.newaxis]
        action = policy.get_action(state).reshape(-1)
        if "thickener" in env.get_name():
            state, cost, done, _ = env.step(action)
            reward = -cost
        else:
            state, reward, done, _ = env.step(action)
        rewards += reward
        lengths += 1

    return (rewards, lengths)



def test_one_trail_local_plot(env, policy):
    state, done = env.reset(), False
    # act_dim = env.action_space.shape[0]
    state_list = []
    action_list = []
    while not done:
        state = state[np.newaxis]
        action = policy.get_action(state).reshape(-1)
        state_list.append(state)
        action_list.append(action)
        state, _, done, _ = env.step(action)
    traj_dict = {
        "state": state_list,
        "action": action_list
    }


    return traj_dict


def test_one_trail_sp_local(env, policy):
    # env = deepcopy(env)
    # policy = deepcopy(policy)

    state, done = env.reset(), False
    rewards = 0
    lengths = 0
    act_dim = env.action_space.shape[0]
    
    while not done:
        state = state
        action = policy.get_action(state).reshape(-1, act_dim)
        # print("actions: ", action[0:3,])
        state, reward, done, _ = env.step(action)
        rewards += reward
        lengths += 1

    return (rewards, lengths)


def test_on_real_env(policy, env, number_of_runs=100):
    # rewards = []
    # episode_lengths = []
    policy = deepcopy(policy)
    policy.eval()           # 切换pytorch的eval()模式，停止模型梯度更新、BN层权重更新和Dropout，主要针对于OPE相关的网络，没有BN层和Dropout的作用和no_grad()相似
    
    if "sales" in env.get_name():
        results = [test_one_trail_sp_local(env, policy) for _ in range(number_of_runs)]
    elif "thickener" in env.get_name():
        results = [test_one_trail_local(env, policy) for _ in range(number_of_runs)]     # TODO: 后续可以并行化提速
        traj_dict = test_one_trail_local_plot(env, policy)         # 取一条单独eval轨迹绘图
    else:
        # results = ray.get([test_one_trail.remote(env, policy) for _ in range(number_of_runs)])
        results = [test_one_trail_local(env, policy) for _ in range(number_of_runs)]
    # results = ray.get([test_one_trail.remote(env, policy) for _ in range(number_of_runs)])
    policy.train()
    
    rewards = [result[0] for result in results]
    episode_lengths = [result[1] for result in results]
    
    rew_mean = np.mean(rewards)
    len_mean = np.mean(episode_lengths)

    res = OrderedDict()
    res["Reward_Mean_Env"] = rew_mean
    res["Length_Mean_Env"] = len_mean
    if "thickener" in env.get_name():
        res["Traj_dict"] = traj_dict

    return res
