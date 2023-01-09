# -*- coding:utf8 -*-
import pprint

import numpy as np

# from neorl.neorl_envs.thickener.exp.data_package import DataPackage


# 实验类：用于调度env和controller进行交互，并每隔一定的训练轮次(rounds)，对模型进行一次评估
class OneRoundExp:
    def __init__(self, env=None, controller=None,
                 max_step=1000,
                 exp_name=None):
        """

        :param env:
        :param controller:
        :param max_step: 仿真迭代次数
        :param exp_name:
        """

        self.env = env
        self.controller = controller
        # 每个round的迭代次数
        self.max_step = max_step
        # 总迭代次数上限
        self.render_mode = False
        self.log = {}
        if exp_name is None:
            exp_name = "None"
        self.exp_name = exp_name

    def add_log(self, key, value):
        self.log[key] = value

    def render(self):

        print('************Exp**************')
        # print("Step : %i" % self.step)
        pprint.pprint(self.log)
        print('************Exp**************')
        print()

    def run(self):

        state = self.env.reset()
        self.controller.step_reset()
        self.controller.env = self.env
        # 训练eval_cycle个round之后，进行一次模型评估

        state_list = []
        action_list = []
        next_state_list = []
        reward_list = []
        done_list = []
        index_list = np.linspace(0, self.max_step, (100 + 1) if self.max_step >= 10000 else (10 + 1))[1:]

        for step in range(self.max_step):

            # 控制器计算策略
            action = self.controller.act(state)

            # 仿真环境进行反馈
            next_state, cost, done, _ = self.env.step(action)
            r = -cost

            # 记录环境相关状态值
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            reward_list.append(r)     # r==cost
            done_list.append(done)

            # 训练模型
            self.controller.train(state, action, next_state, r, done)
            state = next_state

            # self.log = {}
            # self.add_log("step", step)
            if step % 1000 == 0:
                print("step:", step, " ------------------max_step: ", self.max_step)

            if self.render_mode:
                self.render()
            # if done:
            #     break

        data_dict = {
            'obs': state_list,
            'next_obs': next_state_list,
            'action': action_list,
            'reward': reward_list,
            'done': done_list,
            'index': index_list,
        }

        return data_dict
