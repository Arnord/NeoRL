
# -*- coding:utf8 -*-
import copy
import pprint
import gym

import numpy as np
from gym.utils import seeding

from neorl import core
from neorl.neorl_envs.thickener.common.penaltys.demo_penalty import DemoPenalty
from neorl.neorl_envs.thickener import utils
from neorl.utils import get_json, sample_dataset, LOCAL_JSON_FILE_PATH, DATA_PATH


class BaseEnv(core.EnvData):
    """
    Superclass for all simulations.
    """
    def __init__(self, dt=1, penalty_calculator=None,
                 size_yudc=None,
                 y_low=None, u_low=None,
                 d_low=None, c_low=None,
                 y_high=None, u_high=None,
                 d_high=None, c_high=None,
                 normalize=True,
                 render_mode=False,
                 terminate_state = False,
                 time_length=1.0,
                 one_step_length=0.001,
                 y_name=None,
                 u_name=None,
                 c_name=None,
                 d_name=None,
                 reset_after_timesteps=10000,   # 1000
                 ):
        # time step
        self.dt = dt
        self.np_random = None
        self.penalty = 0
        self.done = False
        self.terminate_state = terminate_state
        self.render_mode = render_mode
        self.time_step = -1
        if size_yudc is None:
            raise Exception('No size_yudc!')

        self.size_yudc = size_yudc

        # goal of y
        self.y_star = np.zeros(self.size_yudc[0])
        # y --- core indices
        self.y = np.zeros(self.size_yudc[0])

        # control input
        self.u = np.zeros(self.size_yudc[1])

        # unmeasurable parameters
        self.d = np.zeros(self.size_yudc[2])

        # measurable parameters
        self.c = np.zeros(self.size_yudc[3])



        # 定义参量边界
        self.y_bounds = None
        self.u_bounds = None
        self.d_bounds = None
        self.c_bounds = None


        # set bounds for yudc
        self.set_y(self.size_yudc[0], y_low, y_high)
        self.set_u(self.size_yudc[1], u_low, u_high)
        self.set_d(self.size_yudc[2], d_low, d_high)
        self.set_c(self.size_yudc[3], c_low, c_high)

        # Each env has a penalty calculator, the default calculator is DemoPenalty.
        if penalty_calculator is None:
            penalty_calculator = DemoPenalty()
        self.penalty_calculator = penalty_calculator
        self.penalty_calculator.u_bounds = self.u_bounds
        self.penalty_calculator.y_shape = self.size_yudc[0]
        self.penalty_calculator.u_shape = self.size_yudc[1]
        self.penalty_calculator.create()

        # normalize action in to range(-1,1)
        self.normalize = normalize

        self.external_u_bounds = np.copy(self.u_bounds)
        if self.normalize:
            self.external_u_bounds[:, 0] = -1*np.ones(self.size_yudc[1])
            self.external_u_bounds[:, 1] = np.ones(self.size_yudc[1])

        # store log
        self.log = {}

        self.time_length = time_length
        self.one_step_length = one_step_length


        if y_name is None:
            y_name = [str(i) for i in range(self.size_yudc[0])]
        self.y_name = y_name

        if u_name is None:
            u_name = [str(i) for i in range(self.size_yudc[1])]
        self.u_name = u_name

        if c_name is None:
            c_name = [str(i) for i in range(self.size_yudc[2])]
        self.c_name = c_name

        if d_name is None:
            d_name = [str(i) for i in range(self.size_yudc[3])]
        self.d_name = d_name

        # reset_after_timesteps
        self.env_steps = None
        self.reset_after_timesteps = reset_after_timesteps


    @staticmethod
    def set_bound(size, low, high, kind='Unkown'):
        if type(low) is list:
            low = np.array(low)
        if type(high) is list:
            high = np.array(high)
        if low is None:
            low = np.ones(size)*np.inf*-1
        if high is None:
            high = np.ones(size)*np.inf*1
        if not utils.is_nparray(low) or not utils.is_nparray(high):
            raise TypeError('Bounds should be numpy.ndarray!')
        if low.shape[0] != size or high.shape[0] != size:
            raise ValueError('Shape of bounds is not match to dim %s' % kind )
        return np.concatenate((low.reshape(size, 1), high.reshape(size, 1)), axis=1)

    # 定义目标变量的边界，下同 output
    def set_y(self,size, y_low = None, y_high = None):
        self.y_bounds = self.set_bound(size, y_low, y_high, 'Indices')

    def set_u(self, size, u_low=None, u_high=None):
        self.u_bounds = self.set_bound(size, u_low, u_high, 'Control input')

    def set_d(self,size,d_low = None, d_high=None):
        self.d_bounds = self.set_bound(size, d_low, d_high, 'Unmeasurable parameters')

    def set_c(self, size, c_low=None, c_high=None):
        self.c_bounds = self.set_bound(size, c_low, c_high, 'Measurable parameters')

    def set_y_star(self, y_star):
        if y_star.shape != self.y_bounds[:,0]:
            raise ValueError('Shape of y* is not match to  y')
        self.y_star = y_star

    # clip x in the range of bounds
    def bound_detect(self, x, bounds, item_name='Unknow'):
        if x.shape[0] == 0:
            return 0, 'Normal', x
        low = bounds[:, 0]
        high = bounds[:, 1]
        res = np.clip(x, low, high)
        if (x < low).sum() >= 1:
            self.add_log("Bound %s"%item_name, "Too low")
            return -1, 'Too low', res
        if (x > high).sum() >= 1:
            self.add_log("Bound %s"%item_name, "Too large")
            return 1, 'Too high', res
        return 0, 'Normal', res

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ----------------------------------------------------------
    # Need to be Implement for reset
    def reset_y(self):
        raise NotImplementedError('Please implement reset_y')

    def reset_u(self):
        raise NotImplementedError('Please implement reset_u')

    def reset_y_star(self):
        raise NotImplementedError('Please implement reset_y_star')

    def reset_c(self):
        return np.array([])

    def reset_d(self):
        return np.array([])
    # ----------------------------------------------------------



    # ----------------------------------------------------------
    # Normalize action
    """
    Input control u is in a range between (-1*np.ones(),np.ones())
    The Normalization could scale u between (u_bounds[:,0],u_bounds[:,1])
    """
    def normalize_actions(self, u):
        if (u > 1.0).sum() > 0 or (u < -1.0).sum() > 0:
            raise ValueError("u before normalization exceeds (-1,1)")
        low = self.u_bounds[:, 0]
        high = self.u_bounds[:, 1]

        action = low + ( u + 1.0) * 0.5 * (high-low)

        action = np.clip(action, low, high)

        return action

    # ----------------------------------------------------------

    def _reset_y(self):
        y = self.reset_y()
        bound_res = self.bound_detect(y, self.y_bounds, item_name='reset y')
        y = bound_res[2]
        if bound_res[0] != 0:
            raise ValueError('Reset for y is %s' % bound_res[1])
        if len(y) != self.size_yudc[0]:
            raise ValueError('Shape for y is wrong')

        self.y = y

    def _reset_y_star(self):
        y_star = self.reset_y_star()
        bound_res = self.bound_detect(y_star, self.y_bounds, item_name='reset y*')
        y_star = bound_res[2]
        if bound_res[0] != 0:
            raise ValueError('Reset for y_star is %s' % bound_res[1])
        if len(y_star) != self.size_yudc[0]:
            raise ValueError('Shape for y_star is wrong')
        self.y_star = y_star

    def _reset_u(self):
        u = self.reset_u()

        if len(u) != self.size_yudc[1]:
            raise ValueError('Shape for u is wrong' )

        bound_res = self.bound_detect(u, self.u_bounds, item_name='reset u')
        u= bound_res[2]
        if bound_res[0] != 0:
            raise ValueError('Reset for u is %s' % bound_res[1])
        self.u = u

    def _reset_d(self):
        d = self.reset_d()
        if len(d) != self.size_yudc[2]:
            raise ValueError('Shape for d is wrong')
        bound_res = self.bound_detect(d, self.d_bounds, item_name='reset d')
        d = bound_res[2]
        if bound_res[0] != 0:
            raise ValueError('Reset for d is %s' % bound_res[1])
        self.d = d

    def _reset_c(self):
        c = self.reset_c()
        if len(c) != self.size_yudc[3]:
            raise ValueError('Shape for c is wrong')
        bound_res = self.bound_detect(c, self.c_bounds, item_name='reset c')
        c = bound_res[2]
        if bound_res[0] != 0:
            raise ValueError('Reset for c is %s' % bound_res[1])

        self.c = c

    def _reset(self):
        pass

    def reset(self):


        self._reset()
        self._reset_d()
        self._reset_c()
        self._reset_u()
        self._reset_y()
        self._reset_y_star()
        self.time_step = 0
        # used to set the self.done variable - If larger than self.reset_after_timesteps, the environment resets
        self.env_steps = 0

        # whether or not the trajectory has ended
        self.done = False

        return self.observation()

    def solve_u(self, u):
        bound_res_u = self.bound_detect(u, self.u_bounds,item_name='simulation u')
        if bound_res_u[0] != 0:
            self.done = True
        return bound_res_u

    def observation(self):
        """
        Could be implemented
        :return:
        """
        return np.concatenate((self.y_star, self.y, self.c))

    def render(self):

        print('-----------Env--------------')
        dic = {
            "y_star": self.y_star,
            "y_new": self.y,
            "u": self.u,
            "d": self.d,
            "c": self.c,
            "penalty": self.penalty,
            "done": self.done,
        }
        self.log.update(dic)

        pprint.pprint(self.log)
        print('----------------------------')

        return None

    def step(self, new_u=None):
        if self.done:
            self.reset()

        if self.time_step == -1:
            raise ValueError('Please reset before step')
        if new_u is None:
            new_u = self.u

        new_u = np.array(new_u)
        self.log = {}
        self.add_log("u normal",new_u)
        if self.normalize is True:
            new_u = self.normalize_actions(new_u)
        # clean log
        self.add_log("u action",new_u)
        self.add_log("y old", self.y)
        self.done = False
        self.penalty = 0

        # add time
        self.time_step += 1

        # clip u in u_bounds
        self.u = self.solve_u(new_u)[2]


        # calculate the penalty according to penalty calculator
        self.penalty = self.penalty_calculator.cal(self.y_star, self.y, self.u, self.c, self.d)

        # calculate new y u d c ,and whether the env is terminal
        self.done = self._step()


        if self.render_mode is True:
            self.render()

        return self.observation(), self.penalty, self.done, None

    def _step(self):

        done = False
        for _ in range(self.dt):
            self.y, self.u, self.c, self.d = self.f(self.y, self.u, self.c, self.d)

            # clip y,c,d in their range
            bound_res_y = self.bound_detect(self.y, self.y_bounds,item_name='simulation y')
            self.y = bound_res_y[2]

            bound_res_c = self.bound_detect(self.c, self.c_bounds,item_name='simulation c')
            self.c = bound_res_c[2]

            bound_res_d = self.bound_detect(self.d, self.d_bounds,item_name='simulation d')
            self.d = bound_res_d[2]

            if bound_res_y[0] != 0:
                #print('Indices are %s' % bound_res_y[1])
                #done = True
                if self.terminate_state:
                    done = True
                    break
                else:
                    done = False

        # Stopping condition
        self.env_steps += 1
        if self.env_steps >= self.reset_after_timesteps:
            done = True

        return done

    def observation_size(self):
        return len(self.observation())

    def f(self, y, u, c, d):
        """
        Implement the simulation process in one step
        :return: new y,u,c,d
        """
        return (y,u,c,d)
        #raise NotImplementedError

    def add_log(self, key, value):
        self.log[key] = copy.deepcopy(value)


    def get_dataset(self, task_name_version: str = None, data_type: str = "human", train_num: int = 10000,
                    need_val: bool = True, val_ratio: float = 0.1, path: str = DATA_PATH, use_data_reward: bool = True):
        """
        Get dataset from given env. However, Thickener env only has one training dataset and one validation dataset, where
        the actions are from human or a learned promotion policy model.

        :param task_name_version: The name and version (if applicable) of the task,
            default is the same as `task` while making env
        :param data_type: Which type of policy is used to collect data. It should
            be one of ["high", "medium", "low"], default to `high`
        :param train_num: The num of trajectory of training data. Note that the num
            should be less than 10,000, default to `100`
        :param need_val: Whether needs to download validation data, default to `True`
        :param val_ratio: The ratio of validation data to training data, default to `0.1`
        :param path: The directory of data to load from or download to `./data/`
        :param use_data_reward: Whether uses default data reward. If false, a customized
            reward function should be provided by users while making env

        :return train_samples, val_samples
        """

        if data_type.lower() == "random" or "const" or 'low' or 'medium' or 'high':
            data_type = 'random'
        else:
            raise Exception(f"Please check `data_type`, {data_type} is not supported!")

        # task_name_version = self._name if task_name_version is None else task_name_version
        task_name_version = 'thickener'

        data_json = get_json(LOCAL_JSON_FILE_PATH)

        train_samples = sample_dataset(task_name_version, path, train_num, data_json, data_type, use_data_reward,
                                       self._reward_func, "train")
        train_samples['action'] /=self.action_space.high    # [0]  # to make the action in [-1,1]. It will be better to revise the output of the policy network
        val_samples = None
        if need_val:
            val_samples = sample_dataset(task_name_version, path, int(train_num * val_ratio), data_json, data_type,
                                         use_data_reward, self._reward_func, "val")
            val_samples['action'] /=self.action_space.high  # [0]
        return train_samples, val_samples





