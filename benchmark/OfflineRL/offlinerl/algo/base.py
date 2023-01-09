import os
import uuid
import json
from abc import ABC, abstractmethod

import torch
from collections import OrderedDict
from loguru import logger
from benchmark.OfflineRL.offlinerl.utils.exp import init_exp_logger
from benchmark.OfflineRL.offlinerl.utils.io import create_dir, download_helper, read_json
from benchmark.OfflineRL.offlinerl.utils.logger import log_path

from aim import Run, Image
from common import get_traj_plt
import matplotlib.pyplot as plt



class BaseAlgo(ABC):
    def __init__(self, args):

        logger.info('Init AlgoTrainer')
        if "exp_name" not in args.keys():
            exp_name = str(uuid.uuid1()).replace("-", "")
        else:
            exp_name = args["exp_name"]

        if ("aim_path" in args.keys()) and os.path.exists(args["aim_path"]):
            repo = args["aim_path"]
        else:
            repo = os.path.join(log_path(), "./.aim")
            if not os.path.exists(repo):
                logger.info('{} dir is not exist, create {}', repo, repo)
                os.system(str("cd " + os.path.join(repo, "../") + "&& aim init"))

        self.run = Run(
            experiment=exp_name,
            repo=log_path(),
        )

        self.repo = repo
        # self.exp_logger = init_exp_logger(repo=repo, experiment_name=exp_name)
        # if self.exp_logger.repo is not None:  # a naive fix of aim exp_logger.repo is None
        #     self.index_path = self.exp_logger.repo.index_path
        self.index_path = os.path.join(repo, exp_name)
        if not os.path.exists(self.index_path):
            create_dir(self.index_path)
        # else:
        #     repo = os.path.join(log_path(), "./.aim")
        #     if not os.path.exists(repo):
        #         logger.info('{} dir is not exist, create {}', repo, repo)
        #         os.system(str("cd " + os.path.join(repo, "../") + "&& aim init"))
        #     self.index_path = repo

        self.models_save_dir = os.path.join(self.index_path, "models")
        create_dir(self.models_save_dir)

        self.metric_logs = OrderedDict()
        self.metric_logs_path = os.path.join(self.index_path, "metric_logs.json")

        self.index_path = repo

        # self.exp_logger.set_params(args, name='hparams')
        self.run["hparams"] = args
        self.best_reward = None

    def log_res(self, epoch, result):
        logger.info('Epoch : {}', epoch)

        # Thickener plot
        for k, v in result.items():
            if k == "Traj_dict":
                aim_figure = Image(get_traj_plt(v))
                self.run.track(aim_figure, epoch=epoch, name=f"traj_plot at epoch: {epoch}")
                result["Traj_dict"] = "traj plot over"
                continue
            logger.info('{} : {}', k, v)
            self.run.track(v, name=k.split(" ")[0], epoch=epoch, context={"subset": "train"})


        # In aim 3.14.X Terminal logs are captured by default
        self.metric_logs[str(epoch)] = result
        with open(self.metric_logs_path, "w") as f:
            json.dump(self.metric_logs, f)

        if 'Reward_Mean_Env' in result.keys():
            if self.best_reward is None:
                self.best_reward = result['Reward_Mean_Env'] - 1
            if result['Reward_Mean_Env'] > self.best_reward:
                self.best_reward = result['Reward_Mean_Env']
                logger.info('Model Saved at epoch {}', epoch)   # TODO:暂时不保存模型
                # self.save_model(os.path.join(self.models_save_dir, str(epoch) + ".pt"))

    @abstractmethod
    def train(self,
              history_buffer,
              eval_fn=None, ):
        pass

    def _sync_weight(self, net_target, net, soft_target_tau=5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)

    @abstractmethod
    def get_policy(self, ):
        pass

    # @abstractmethod
    def save_model(self, model_path):
        torch.save(self.get_policy(), model_path)

    # @abstractmethod
    def load_model(self, model_path):
        model = torch.load(model_path)

        return model

    def get_best_reward(self):
        return self.best_reward
