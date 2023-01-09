#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import sys
import pandas as pd

from neorl.neorl_envs.thickener.control.base_control import ControlBase
from neorl.neorl_envs.thickener.exp.one_round_exp import OneRoundExp as BaseExp

from neorl.neorl_envs.thickener.thickener import Thickener


class Test_Control(ControlBase):
    def __init__(self, u_bounds, act_u=None):
        super(Test_Control, self).__init__(u_bounds)
        self.act_u = act_u
        self.u_bounds = u_bounds

    def _act(self, state):
        return self.act_u(state)

    def train(self, s, u, ns, r, done):
        pass


def test_controller_construct(act_list, Env, init_para=None, act_name=None, test_step=200):

    exp_list = []
    if act_name is None:
        act_name = [str(i+1) for i in range(len(act_list))]
    for id, act in enumerate(act_list):
        env = Env(**init_para)
        controller = Test_Control(np.copy(env.u_bounds), act_u=act)

        exp = BaseExp(env=env, controller=controller,
                      exp_name=act_name[id], max_step=test_step)
        exp_list.append(exp)


    return exp_list


def simulation_test(Env=None, mode="uniform", init_para=None,
                    seprate_num=5, const_u=None,
                    test_step=200):

    save_path = "./tmp_data"

    if Env is None:
        raise ValueError("No env to simulation")
    if init_para is None:
        init_para = {}
    tmp_env = Env(**init_para)
    u_bounds = tmp_env.external_u_bounds

    low = u_bounds[:, 0]
    high = u_bounds[:, 1]
    act_name = []
    if mode == "uniform":
        act_u_list = []
        for id in np.arange(0, seprate_num, 1):
            action = np.copy((high-low)/seprate_num*id) + low
            act_u = np.copy(action)
            act_u_list.append(lambda x, act=act_u: act)
            act_name.append(str(action))
        act_u_list.append(lambda x, act=high: act)
        act_name.append(str(high))

        exp_list = test_controller_construct(act_u_list, Env, init_para=init_para, act_name=act_name, test_step=test_step)


    elif mode == "random":
        act_u = lambda x, al=low, ah=high: np.random.uniform(al, ah)
        act_name = ["random action"]
        exp_list = test_controller_construct([act_u], Env, init_para=init_para, act_name=act_name, test_step=test_step)

    elif mode == "const":
        if const_u is None:
            raise ValueError("const_u could not be None")
        act_u_list = []
        act_name = []
        for id, u in enumerate(const_u):
            act_u = np.array(u)
            act_u_list.append(lambda x, act=act_u: act)
            act_name.append(str(act_u))
        exp_list = test_controller_construct(act_u_list, Env, init_para=init_para, act_name=act_name, test_step=test_step)

    else:
        raise ValueError("mode should be assigned in {uniform, const, random}")

    data_dict = [exp_list[i].run() for i in range(len(exp_list))][0]

    # build_npz
    # np.savez(f"{save_path}/thickener-random-{test_step}-val",
    #          obs=data_dict['obs'],
    #          next_obs=data_dict['next_obs'],
    #          action=data_dict['action'],
    #          reward=data_dict['reward'],
    #          done=data_dict['done'],
    #          index=data_dict['index'],
    #          )


    # bulid_csv
    if len(data_dict['obs'][0]) == 8:
        obs_keys = ["y_h_star", "y_c_star", "y_h", "y_c", "u_fu", "u_ff", "c_fi", "c_ci"]
        df_obs = pd.DataFrame(data_dict['obs'], columns=obs_keys)
        df_obs.to_csv(f"{save_path}/thickener_simulation_6000_test.csv", index=False)


    print("Simulation over")


def debug_const():
    simulation_test(Thickener, mode="const",
                    const_u=[[60, 30], [60, 50], [80, 30], [85, 38], [80, 50], [100, 30], [100, 50]],
                    test_step=120)


def debug_cy_mul():
    init_para = {
        "noise_in": True
    }
    simulation_test(Thickener, mode='const', init_para=init_para, test_step=500,
                    const_u=[[60, 30], [60, 50], [80, 30], [85, 38], [80, 50], [100, 30], [100, 50]], )


def debug_cy():
    init_para = {
        "noise_in": True,
        "noise_p": 0.01
    }
    simulation_test(Thickener, mode='const', init_para=init_para, test_step=1000,
                    const_u=[[85, 38]])


def debug_s():
    init_para = {
        "c_start": [46, 77.61991659],
        "y_start": [1.142443791260041, 759.9451550012649]

    }

    simulation_test(Thickener, mode='const', init_para=init_para, test_step=2000, const_u=[[80, 40]])


def debug_random():
    simulation_test(Thickener, init_para={"noise_in": True, "noise_type": 2, "reset_after_timesteps": 1000000}, mode="random", test_step=6000)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        # 恒定控制，波动输入
        if sys.argv[1] == '-cy':
            debug_cy()
        # 恒定控制，波动输入
        elif sys.argv[1] == '-const':
            debug_const()
        # 指定c y起始值随机策略
        elif sys.argv[1] == '-s':
            debug_s()
        # 多个恒定策略，噪音输入
        elif sys.argv[1] == '-cy_mul':
            debug_cy_mul()
        # 随机策略，噪音输入
        elif sys.argv[1] == '-random':
            debug_random()
