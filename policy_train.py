
import os
import ray
import json
import time
import argparse
import numpy as np
import copy
import pandas as pd

from ray import tune

from benchmark.OfflineRL.offlinerl.algo import algo_select
from benchmark.OfflineRL.offlinerl.data import load_data_from_neorl
from benchmark.OfflineRL.offlinerl.evaluation import OnlineCallBackFunction, PeriodicCallBack

# SEEDS = [7, 42, 210]
SEEDS = [42]

ResultDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp_results'))


def training_function(config):
    ''' run on a seed '''
    config["kwargs"]['seed'] = config['seed']
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(config["kwargs"])
    train_buffer, val_buffer = load_data_from_neorl(algo_config["task"], algo_config["task_data_type"], algo_config["task_train_num"])
    algo_config.update(config)
    algo_config["device"] = "cuda"
    algo_config['dynamics_path'] = os.path.join(config['dynamics_root'],
                                                f'{algo_config["task"]}-{algo_config["task_data_type"]}-{algo_config["task_train_num"]}-{config["seed"]}.pt')
    algo_config['behavior_path'] = os.path.join(config['behavior_root'],
                                                f'{algo_config["task"]}-{algo_config["task_data_type"]}-{algo_config["task_train_num"]}-{config["seed"]}.pt')
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)
    if config['kwargs']['task'] == 'sp' or 'sales' in config['kwargs']['task']:
        # Note the evaluation is slow in sp env, since it interact with 10,000 user models each step, 
        # which can be viewed as run 10,000 trajectories simultaneously and average the rewards. However, we do not need to 
        # too many trials in this env, so we just set the default number of runs to 3.
        callback = PeriodicCallBack(OnlineCallBackFunction(), 10)
        callback.initialize(train_buffer=train_buffer, val_buffer=val_buffer, task=algo_config["task"], number_of_runs=3)
    else:
        callback = PeriodicCallBack(OnlineCallBackFunction(), 2)
        # callback.initialize(train_buffer=train_buffer, val_buffer=val_buffer, task=algo_config["task"], number_of_runs=1000)
        callback.initialize(train_buffer=train_buffer, val_buffer=val_buffer, task=algo_config["task"], number_of_runs=10)
    algo_trainer.train(train_buffer, val_buffer, callback_fn=callback)
    # algo_trainer.exp_logger.flush()
    time.sleep(10)  # sleep ensure the log is flushed even if the disks or cpus are busy

    result = algo_trainer.get_best_reward()
    grid_search_keys = list(algo_config['grid_tune'].keys())
    parameter = {k: algo_config[k] for k in grid_search_keys}

    return {
        'reward': result,
        'parameter': parameter,
        'seed': config['seed'],
    }


def upload_result_single(task_name: str, algo_name: str, results: list):
    ''' upload the result '''
    # upload txt
    file_name = task_name + ',' + algo_name + '.txt'
    best_result = results[0]
    with open(os.path.join(ResultDir, file_name), 'w') as f:
        f.write(str(best_result['reward_mean']) + "+-" + str(best_result['reward_std']))
        for k, v in best_result['parameter'].items():
            f.write('\n')
            f.write(f'{k} : {v}')

    # upload json
    file_name = task_name + ',' + algo_name + '.json'
    with open(os.path.join(ResultDir, file_name), 'w') as f:
        json.dump(results, f, indent=4)


def flatten_dict(dt, delimiter="/", prevent_delimiter=False):
    dt = copy.deepcopy(dt)
    if prevent_delimiter and any(delimiter in key for key in dt):
        # Raise if delimiter is any of the keys
        raise ValueError(
            "Found delimiter `{}` in key when trying to flatten array."
            "Please avoid using the delimiter in your specification.")
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    if prevent_delimiter and delimiter in subkey:
                        # Raise  if delimiter is in any of the subkeys
                        raise ValueError(
                            "Found delimiter `{}` in key when trying to "
                            "flatten array. Please avoid using the delimiter "
                            "in your specification.")
                    add[delimiter.join([key, str(subkey)])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str)
    parser.add_argument('--level', type=str)
    parser.add_argument('--amount', type=int)
    parser.add_argument('--algo', type=str, help='select from `bc`, `bcq`, `plas`, `cql`, `crr`, `bremen`, `mopo`')
    parser.add_argument('--address', type=str, default=None, help='address of the ray cluster')
    args = parser.parse_args()

    # ray.init(args.address)

    domain = args.domain
    level = args.level
    amount = args.amount
    algo = args.algo

    ''' run and upload result '''
    config = {}
    config["kwargs"] = {
        "exp_name": f'{domain}-{level}-{amount}-{algo}',
        "algo_name": algo,
        "task": domain,
        "task_data_type": level,
        "task_train_num": amount,
    }
    _, _, algo_config = algo_select({"algo_name": algo})

    parameter_names = []
    grid_tune = algo_config["grid_tune"]
    for k, v in grid_tune.items():
        parameter_names.append(k)
        config[k] = v[0]

    config['seed'] = SEEDS[0]
    config['dynamics_root'] = os.path.abspath('dynamics')
    config['behavior_root'] = os.path.abspath('behaviors')

    def get_results_df(config, seed):
        config_training = config
        config_training['seed'] = seed

        return flatten_dict(training_function(config_training), delimiter=".")


    df = pd.DataFrame.from_records(
        [
                get_results_df(config, seed)
                for seed in SEEDS
        ],
    )

    ''' process result '''
    results = {}
    for i in range(len(df)):
        parameter = {}
        for pn in parameter_names:
            parameter[pn] = df[f'parameter.{pn}'][i]
            if type(parameter[pn]) == np.int64:
                parameter[pn] = int(parameter[pn])  # covert to python type
        parameter_string = str(parameter)

        if not parameter_string in results:
            results[parameter_string] = {
                'parameter': parameter,
                'rewards': [0 for x in range(len(SEEDS))],
            }

        results[parameter_string]['rewards'][SEEDS.index(df['seed'][i])] = df['reward'][i]


    def summary_result(single_result):
        single_result.update({
            'reward_mean': np.mean(single_result['rewards']),
            'reward_std': np.std(single_result['rewards']),
        })
        return single_result

    results = [summary_result(single_result) for single_result in results.values()]

    ''' upload result '''
    upload_result_single(f'{domain}-{level}-{amount}', algo, results)
