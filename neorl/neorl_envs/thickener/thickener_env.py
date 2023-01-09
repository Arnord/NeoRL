import os
import random
from neorl.neorl_envs.thickener.thickener import Thickener


def build_env(sys_mode):

    # 随机噪音
    if sys_mode == 'random':
        init_para = {
            "noise_in": True,
            "noise_type": 2,
            "reset_after_timesteps": 1000,
        }
    # 无噪音，恒定c
    elif sys_mode == 'const':
        init_para = None

    # 有噪音，波动c
    elif sys_mode == 'cy':
        init_para = {
            "noise_in": True,
            "noise_p": 0.01
        }

    # 指定c y起始值
    elif sys_mode == 's':
        init_para = {
            "c_start": [46, 77.61991659],
            "y_start": [1.142443791260041, 759.9451550012649]

        }

    else:
        raise NotImplementedError

    return Thickener(**init_para)


def thickener(sys_mode):

    env = build_env(sys_mode)

    return env
